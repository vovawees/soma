package ru.vovawe.soma;

import com.google.gson.Gson;
import com.mojang.authlib.GameProfile;
import net.minecraft.core.BlockPos;
import net.minecraft.core.Direction;
import net.minecraft.core.registries.BuiltInRegistries;
import net.minecraft.network.Connection;
import net.minecraft.network.chat.Component;
import net.minecraft.network.protocol.PacketFlow;
import net.minecraft.server.MinecraftServer;
import net.minecraft.server.level.ServerLevel;
import net.minecraft.world.InteractionHand;
import net.minecraft.world.entity.Entity;
import net.minecraft.world.entity.MoverType;
import net.minecraft.world.entity.item.ItemEntity;
import net.minecraft.world.entity.monster.Enemy;
import net.minecraft.world.entity.player.Inventory;
import net.minecraft.world.item.Item;
import net.minecraft.world.item.ItemStack;
import net.minecraft.world.level.GameType;
import net.minecraft.world.level.block.state.BlockState;
import net.minecraft.world.phys.AABB;
import net.minecraft.world.phys.BlockHitResult;
import net.minecraft.world.phys.Vec3;
import net.neoforged.bus.api.IEventBus;
import net.neoforged.fml.common.Mod;
import net.neoforged.neoforge.common.NeoForge;
import net.neoforged.neoforge.common.util.FakePlayer;
import net.neoforged.neoforge.event.RegisterCommandsEvent;
import net.neoforged.neoforge.event.server.ServerStartedEvent;
import net.neoforged.neoforge.event.server.ServerStoppingEvent;
import net.neoforged.neoforge.event.tick.ServerTickEvent;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.net.InetSocketAddress;
import java.net.ServerSocket;
import java.net.Socket;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Queue;
import java.util.UUID;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;

/**
 * SOMA environment adapter, server side.
 *
 * The agent is a REAL ServerPlayer: it is in the tab list, mobs target it and
 * it picks up drops natively. It receives numeric frames over TCP, answers with
 * the index of the candidate it wants executed, and never executes anything the
 * brain did not ask for.
 *
 * No policy, no reward shaping, no strategy. Same wire schema as bot/ (Node).
 */
@Mod("soma")
public class SomaMod {

    private static final Gson GSON = new Gson();
    private Agent agent;

    public SomaMod(IEventBus modBus) {
        NeoForge.EVENT_BUS.addListener(this::onServerStarted);
        NeoForge.EVENT_BUS.addListener(this::onServerTick);
        NeoForge.EVENT_BUS.addListener(this::onServerStopping);
        NeoForge.EVENT_BUS.addListener(this::onRegisterCommands);
        System.out.println("[soma] mod constructed");
    }

    private void onServerStarted(ServerStartedEvent ev) {
        agent = new Agent(ev.getServer());
        agent.start();
    }

    private void onServerStopping(ServerStoppingEvent ev) {
        if (agent != null) agent.stop();
    }

    private void onServerTick(ServerTickEvent.Post ev) {
        if (agent != null) agent.tick();
    }

    private void onRegisterCommands(RegisterCommandsEvent ev) {
        ev.getDispatcher().register(
            net.minecraft.commands.Commands.literal("soma")
                .then(net.minecraft.commands.Commands.literal("start")
                    .executes(c -> { if (agent != null) { agent.start(); agent.setEnabled(true); }
                                     c.getSource().sendSuccess(() -> Component.literal("[soma] enabled"), true); return 1; }))
                .then(net.minecraft.commands.Commands.literal("stop")
                    .executes(c -> { if (agent != null) agent.setEnabled(false);
                                     c.getSource().sendSuccess(() -> Component.literal("[soma] disabled"), true); return 1; }))
                .then(net.minecraft.commands.Commands.literal("status")
                    .executes(c -> { if (agent != null) { agent.start(); agent.status(c.getSource()); }
                                     return 1; }))
                .then(net.minecraft.commands.Commands.literal("hurt")
                    .executes(c -> { if (agent != null) agent.debugHurt(3); return 1; }))
                .then(net.minecraft.commands.Commands.literal("cleanup")
                    .executes(c -> { if (agent != null) agent.cleanupStands(true);
                                     return 1; }))
        );
    }

    // =====================================================================
    //  Agent
    // =====================================================================
    static final class Agent {
        static final int GRID_N = 8;
        static final int GRID_STEP = 2;
        static final int GRID_CELLS = GRID_N * GRID_N * GRID_N;
        static final int MAX_CAND = 48;
        static final int MAX_ENT = 12;
        static final int EV_MAX = 10;
        static final int OBS_TICKS = 6;

        static final double REACH = 4.5;
        static final String HOST = "127.0.0.1";
        static final int PORT = Integer.parseInt(System.getenv().getOrDefault("SOMA_IPC_PORT", "8788"));

        final MinecraftServer server;
        ServerLevel level;
        FakePlayer player;

        final AtomicBoolean enabled = new AtomicBoolean(false);
        final AtomicBoolean running = new AtomicBoolean(false);
        final AtomicReference<String> lastError = new AtomicReference<>("not started");

        ServerSocket serverSocket;
        volatile PrintWriter out;
        volatile BufferedReader in;
        final Queue<int[]> incoming = new ArrayDeque<>();

        int tickCounter = 0;
        int seq = 0;
        int actionCount = 0;
        int errorCount = 0;
        int warmup = 20;
        int cleaned = -1;
        boolean playerVisible = false;
        boolean pathMode = false;

        String activeAction = "none";
        BlockPos activeTarget = null;
        int activeTicks = 0;
        boolean activeOk = true;

        final List<Map<String, Object>> events = new ArrayList<>();
        final Map<Integer, String> blockNames = new HashMap<>();
        List<Map<String, Object>> candidates = new ArrayList<>();
        final java.util.Map<Integer, List<Map<String, Object>>> candBySeq = new java.util.LinkedHashMap<>();
        int lastPosTick = 0;
        double lastMoveX = 0, lastMoveZ = 0;

        BlockPos miningPos = null;
        Direction miningDir = Direction.UP;
        int miningTicks = 0;
        float miningProgress = 0.0f;
        double fallTrack = 0.0;
        int hungerTick = 0;
        int regenTick = 0;

        int lastAttackedId = -1;
        String lastAttackedName = "";
        final Map<Integer, String> savedInventory = new HashMap<>();

        // chunk cache
        net.minecraft.world.level.chunk.LevelChunk curChunk = null;
        int curCX = Integer.MIN_VALUE, curCZ = Integer.MIN_VALUE;
        int gridFallbacks = 0;
        final int[] gridCache = new int[GRID_CELLS];

        Agent(MinecraftServer server) { this.server = server; }

        // ------------------------------------------------------------- net --
        void start() {
            Thread prev = netThread.get();
            if (running.get() && prev != null && prev.isAlive()) return;
            running.set(true);
            Thread t = new Thread(this::netLoop, "soma-net");
            t.setDaemon(true);
            netThread.set(t);
            t.start();
            System.out.println("[soma] agent thread started, ipc " + HOST + ":" + PORT);
        }

        final AtomicReference<Thread> netThread = new AtomicReference<>();

        void stop() {
            running.set(false);
            try { if (serverSocket != null) serverSocket.close(); } catch (Exception ignored) {}
        }

        void setEnabled(boolean v) { enabled.set(v); if (v) warmup = 20; }

        void status(net.minecraft.commands.CommandSourceStack src) {
            src.sendSuccess(() -> Component.literal(
                "[soma] enabled=" + enabled.get()
                    + " player=" + (player == null ? "none" : player.getName().getString())
                    + " seq=" + seq + " actions=" + actionCount
                    + " pos=" + (player == null ? "-" : String.format(Locale.ROOT, "%.0f,%.0f,%.0f",
                        player.getX(), player.getY(), player.getZ()))
                    + " hp=" + (player == null ? 0 : (int) player.getHealth())
                    + " food=" + (player == null ? 0 : player.getFoodData().getFoodLevel())
                    + " inv=" + (player == null ? 0 : countItems())
                    + " gm=" + (player == null ? "-" : player.gameMode.getGameModeForPlayer())
                    + " fly=" + (player == null ? "-" : (player.getAbilities().mayfly + "/" + player.getAbilities().flying))
                    + " spec=" + (player == null ? "-" : player.isSpectator())
                    + " players=" + server.getPlayerList().getPlayers().size()
                    + " inTab=" + server.getPlayerList().getPlayers().contains(player)
                    + " errors=" + errorCount + " lastError=" + lastError.get()), true);
        }

        /** Debug: apply damage to the agent so the damage path can be verified. */
        void debugHurt(float amount) {
            if (player == null) return;
            float before = player.getHealth();
            player.setHealth(Math.max(0.5f, before - amount));
            pushEvent("debug_hurt", (int) amount, null);
            System.out.println("[soma] debug hurt " + before + " -> " + player.getHealth());
        }

        int countItems() {
            try {
                int n = 0;
                Inventory inv = player.getInventory();
                for (int i = 0; i < inv.getContainerSize(); i++) if (!inv.getItem(i).isEmpty()) n++;
                return n;
            } catch (Throwable e) { return -1; }
        }

        void netLoop() {
            while (running.get()) {
                try {
                    ServerSocket ss = new ServerSocket();
                    ss.setReuseAddress(true);
                    ss.bind(new InetSocketAddress(HOST, PORT));
                    serverSocket = ss;
                    System.out.println("[soma] listening on " + HOST + ":" + PORT);
                    while (running.get()) {
                        Socket s = ss.accept();
                        s.setTcpNoDelay(true);
                        in = new BufferedReader(new InputStreamReader(s.getInputStream()));
                        out = new PrintWriter(s.getOutputStream(), true);
                        System.out.println("[soma] brain connected");
                        lastError.set("not started");
                        String line;
                        while (running.get() && (line = in.readLine()) != null) {
                            line = line.trim();
                            if (line.isEmpty()) continue;
                            try {
                                Map<?, ?> msg = GSON.fromJson(line, Map.class);
                                Object t = msg.get("t");
                                Object i = msg.get("i");
                                if ("act".equals(t) && i instanceof Number) {
                                    int sq = msg.get("s") instanceof Number ? ((Number) msg.get("s")).intValue() : -1;
                                    synchronized (incoming) {
                                        incoming.clear();
                                        incoming.add(new int[]{((Number) i).intValue(), sq});
                                    }
                                }
                            } catch (Exception e) { errorCount++; }
                        }
                        System.out.println("[soma] brain disconnected");
                    }
                } catch (Throwable e) {
                    lastError.set(String.valueOf(e));
                    System.out.println("[soma] net error: " + e);
                    try { Thread.sleep(2000); } catch (InterruptedException ignored) {}
                }
            }
        }

        // ----------------------------------------------------------- player --
        void ensurePlayer() {
            if (level == null) level = server.getLevel(net.minecraft.world.level.Level.OVERWORLD);
            if (player == null && level != null) {
                GameProfile profile = new GameProfile(UUID.fromString("6dded40e-b7fe-3867-86cf-20f080f1c988"), "SOMA");
                player = new FakePlayer(level, profile) {
                    @Override public boolean isInvulnerableTo(net.minecraft.world.damagesource.DamageSource d) { return false; }
                    @Override public void tick() { /* driven by the agent */ }
                    @Override public boolean isSpectator() { return false; }
                    @Override public boolean isInvisible() { return false; }
                };
                BlockPos spawn = level.getSharedSpawnPos();
                player.setPos(spawn.getX() + 0.5, findGroundY(spawn.getX(), spawn.getZ()) + 1.0, spawn.getZ() + 0.5);
                setSurvival();
                attachToServer();
                // AFTER placeNewPlayer: it reloads the player NBT from disk and wipes
                // anything we put in the inventory before that
                giveStarterKit();
                try {
                    player.getFoodData().setFoodLevel(20);
                    player.getFoodData().setSaturation(5.0f);
                } catch (Throwable ignored) {}
                System.out.println("[soma] fake player created at " + player.getX() + "," + player.getY()
                    + "," + player.getZ() + " visible=" + playerVisible);
            }
            if (player != null && player.isRemoved()) { player = null; playerVisible = false; }
        }

        /** Full survival: no flight, no creative invulnerability, no instant break. */
        void setSurvival() {
            try { player.setGameMode(GameType.SURVIVAL); } catch (Throwable ignored) {}
            try {
                player.getAbilities().instabuild = false;
                player.getAbilities().mayfly = false;
                player.getAbilities().flying = false;
                player.getAbilities().invulnerable = false;
            } catch (Throwable ignored) {}
        }

        /** Bootstrap: without a tool and food the agent cannot survive (and cannot craft). */
        void giveStarterKit() {
            try {
                Inventory inv = player.getInventory();
                inv.setItem(0, new ItemStack(net.minecraft.world.item.Items.WOODEN_PICKAXE));
                inv.setItem(1, new ItemStack(net.minecraft.world.item.Items.BREAD, 3));
                inv.setItem(2, new ItemStack(net.minecraft.world.item.Items.TORCH, 8));
                inv.selected = 0;
                System.out.println("[soma] starter kit given");
            } catch (Throwable e) {
                System.out.println("[soma] starter kit failed: " + e);
            }
        }

        void attachToServer() {
            try {
                Connection conn = new Connection(PacketFlow.SERVERBOUND);
                io.netty.channel.embedded.EmbeddedChannel ch = new io.netty.channel.embedded.EmbeddedChannel();
                boolean wired = false;
                for (java.lang.reflect.Field f : Connection.class.getDeclaredFields()) {
                    if (io.netty.channel.Channel.class.isAssignableFrom(f.getType())) {
                        f.setAccessible(true);
                        f.set(conn, ch);
                        wired = true;
                        break;
                    }
                }
                if (!wired) throw new IllegalStateException("no Channel field on Connection");
                registerAllChannels(conn, ch);

                net.minecraft.server.network.CommonListenerCookie cookie =
                    new net.minecraft.server.network.CommonListenerCookie(
                        player.getGameProfile(), 0,
                        net.minecraft.server.level.ClientInformation.createDefault(), false);
                player.connection = new net.minecraft.server.network.ServerGamePacketListenerImpl(
                    server, conn, player, cookie);
                server.getPlayerList().placeNewPlayer(conn, player, cookie);
                playerVisible = true;
                setSurvival();
                System.out.println("[soma] attached -> tab entry, mob aggro, native pickups");
                return;
            } catch (Throwable e) {
                System.out.println("[soma] placeNewPlayer failed (" + e + "), world-only player");
            }
            try {
                level.addNewPlayer(player);
                playerVisible = true;
                System.out.println("[soma] world player registered (no tab entry)");
            } catch (Throwable e2) {
                playerVisible = false;
                System.out.println("[soma] attach failed: " + e2);
            }
        }

        @SuppressWarnings("unchecked")
        void registerAllChannels(Connection conn, io.netty.channel.embedded.EmbeddedChannel ch) {
            try {
                java.util.Set<net.minecraft.resources.ResourceLocation> all = new java.util.HashSet<>();
                java.lang.reflect.Field f = net.neoforged.neoforge.network.registration.NetworkRegistry.class
                    .getDeclaredField("PAYLOAD_REGISTRATIONS");
                f.setAccessible(true);
                java.util.Map<?, ?> regs = (java.util.Map<?, ?>) f.get(null);
                for (Object v : regs.values()) {
                    java.util.Map<?, ?> byId = (java.util.Map<?, ?>) v;
                    for (Object id : byId.keySet()) all.add((net.minecraft.resources.ResourceLocation) id);
                }
                net.neoforged.neoforge.network.registration.ChannelAttributes
                    .getOrCreateAdHocChannels(conn).addAll(all);
                io.netty.util.Attribute<java.util.Map<net.minecraft.network.ConnectionProtocol,
                    java.util.Set<net.minecraft.resources.ResourceLocation>>> attr =
                    ch.attr(net.neoforged.neoforge.network.registration.ChannelAttributes.COMMON_CHANNELS);
                java.util.Map<net.minecraft.network.ConnectionProtocol,
                    java.util.Set<net.minecraft.resources.ResourceLocation>> m = attr.get();
                if (m == null) { m = new java.util.HashMap<>(); attr.set(m); }
                for (net.minecraft.network.ConnectionProtocol proto : net.minecraft.network.ConnectionProtocol.values()) {
                    m.computeIfAbsent(proto, k -> new java.util.HashSet<>()).addAll(all);
                }
                System.out.println("[soma] claimed " + all.size() + " mod channels");
            } catch (Throwable e) {
                System.out.println("[soma] channel registration failed: " + e);
            }
        }

        // ----------------------------------------------------------- health --
        void checkHealth() {
            if (player == null) return;
            try {
                player.setAirSupply(player.getMaxAirSupply());
                if (player.getHealth() > 0.5f) { snapshotInventory(); return; }
                pushEvent("death", 1, null);
                player.setHealth(player.getMaxHealth());
                try { player.getFoodData().setFoodLevel(20); player.getFoodData().setSaturation(5.0f); } catch (Throwable ignored) {}
                restoreInventory();
                BlockPos sp = level.getSharedSpawnPos();
                player.setPos(sp.getX() + 0.5, findGroundY(sp.getX(), sp.getZ()) + 1.0, sp.getZ() + 0.5);
                player.setDeltaMovement(0, 0, 0);
                System.out.println("[soma] agent died, respawned at spawn");
            } catch (Throwable e) {
                try { player.setHealth(player.getMaxHealth()); } catch (Throwable ignored) {}
            }
        }

        void snapshotInventory() {
            savedInventory.clear();
            Inventory inv = player.getInventory();
            for (int i = 0; i < inv.getContainerSize(); i++) {
                ItemStack st = inv.getItem(i);
                if (!st.isEmpty()) {
                    savedInventory.put(i, BuiltInRegistries.ITEM.getKey(st.getItem()).toString());
                }
            }
        }

        void restoreInventory() {
            Inventory inv = player.getInventory();
            inv.clearContent();
            for (Map.Entry<Integer, String> e : savedInventory.entrySet()) {
                Item item = BuiltInRegistries.ITEM.get(
                    net.minecraft.resources.ResourceLocation.tryParse(e.getValue()));
                if (item == null || item == Items_AIR()) continue;
                inv.setItem(e.getKey(), new ItemStack(item, 1));
            }
        }

        static Item Items_AIR() { return net.minecraft.world.item.Items.AIR; }

        // ------------------------------------------------------- stand cleanup --
        void cleanupStands() { cleanupStands(false); }

        void cleanupStands(boolean verbose) {
            if (cleaned >= 0 && !verbose) return;
            cleaned = 0;
            try {
                AABB box = new AABB(-3000, -64, -3000, 3000, 320, 3000);
                List<net.minecraft.world.entity.decoration.ArmorStand> stands =
                    level.getEntitiesOfClass(net.minecraft.world.entity.decoration.ArmorStand.class, box);
                int total = stands.size();
                int mine = 0;
                for (net.minecraft.world.entity.decoration.ArmorStand st : stands) {
                    Component n = st.getCustomName();
                    if (n != null && "SOMA".equals(n.getString())) { st.discard(); mine++; }
                }
                cleaned = mine;
                System.out.println("[soma] armour stands: " + total + " total, " + mine + " SOMA ones removed");
            } catch (Throwable e) {
                System.out.println("[soma] stand cleanup failed: " + e);
            }
        }

        // -------------------------------------------------------------- tick --
        void tick() {
            if (!enabled.get()) return;
            try {
                ensurePlayer();
                if (player == null) return;
                if (cleaned < 0) cleanupStands();
                else if (tickCounter % 600 == 0) cleanupStands(true);
                checkHealth();
                survivalTick();
                advanceMining();
                if (warmup > 0) { warmup--; return; }
                tickCounter++;
                driveActive();
                if (tickCounter % OBS_TICKS != 0) return;
                sendObservation();
                pollAndExecute();
            } catch (Throwable e) {
                errorCount++;
                lastError.set(String.valueOf(e));
            }
        }

        /**
         * Survival rules the vanilla player tick would normally apply. FakePlayer.tick()
         * is empty, so we drive hunger, regeneration and environmental damage here.
         */
        void survivalTick() {
            if (player == null) return;
            try {
                // fall damage
                if (!player.onGround()) {
                    fallTrack = Math.max(fallTrack, player.fallDistance);
                } else if (fallTrack > 3.0) {
                    float dmg = (float) (fallTrack - 3.0);
                    player.setHealth(Math.max(0.5f, player.getHealth() - dmg));
                    pushEvent("fall", (int) Math.round(dmg), null);
                    fallTrack = 0.0;
                }

                // lava / fire
                if (player.isInLava()) {
                    if (++hungerTick % 10 == 0) {
                        player.setHealth(Math.max(0.5f, player.getHealth() - 1.0f));
                        pushEvent("burn", 1, null);
                    }
                }

                // hunger
                hungerTick++;
                if (hungerTick % 1200 == 0) {
                    net.minecraft.world.food.FoodData fd = player.getFoodData();
                    if (fd.getSaturationLevel() > 0) fd.setSaturation(fd.getSaturationLevel() - 1.0f);
                    else if (fd.getFoodLevel() > 0) fd.setFoodLevel(fd.getFoodLevel() - 1);
                    else { player.setHealth(Math.max(0.5f, player.getHealth() - 1.0f)); pushEvent("starve", 1, null); }
                }

                // regeneration
                regenTick++;
                if (regenTick % 400 == 0 && player.getFoodData().getFoodLevel() >= 18
                        && player.getHealth() < player.getMaxHealth()) {
                    player.setHealth(Math.min(player.getMaxHealth(), player.getHealth() + 0.5f));
                }
            } catch (Throwable e) {
                lastError.set("survival: " + e);
            }
        }

        /** Survival-correct mining: progress depends on the tool and the block. */
        void advanceMining() {
            if (miningPos == null || player == null) return;
            miningTicks++;
            try {
                if (miningTicks > 160) { miningPos = null; pushEvent("dig_fail", 1, null); return; }
                BlockState st = bs(miningPos);
                if (st == null || st.isAir()) {
                    miningPos = null;
                    pushEvent("dig", 1, miningPos);
                    finish(true, "mined");
                    return;
                }
                // vanilla-shaped mining: progress per tick depends on the tool
                float prog = 1.0f;
                try {
                    prog = st.getDestroyProgress(player, level, miningPos);
                } catch (Throwable e) {
                    float hard = st.getDestroySpeed(level, miningPos);
                    prog = hard > 0 ? 1.0f / (hard * 30.0f) : 1.0f;
                }
                miningProgress += prog;
                if (miningProgress >= 1.0f) {
                    miningProgress = 0.0f;
                    BlockPos done = miningPos;
                    boolean ok = player.gameMode.destroyBlock(done);
                    miningPos = null;
                    pushEvent(ok ? "dig" : "dig_fail", 1, done);
                    finish(ok, "mined");
                }
            } catch (Throwable e) {
                miningPos = null;
                pushEvent("dig_fail", 1, null);
            }
        }

        // --------------------------------------------------------- movement --
        List<int[]> route = new ArrayList<>();
        int routeIdx = 0;

        void planRoute(BlockPos goal) {
            route.clear();
            routeIdx = 0;
            pathMode = false;
            try {
                List<int[]> p = aStar(player.blockPosition(), goal);
                if (p != null && !p.isEmpty()) { route.addAll(p); pathMode = true; }
            } catch (Throwable e) {
                System.out.println("[soma] pathfinding failed: " + e);
            }
        }

        void driveActive() {
            if (player == null || activeTarget == null) return;
            if (!"goto".equals(activeAction)) return;
            double dx = (activeTarget.getX() + 0.5) - player.getX();
            double dz = (activeTarget.getZ() + 0.5) - player.getZ();
            if (Math.hypot(dx, dz) < 0.9) { route.clear(); finish(true, "arrived"); return; }
            // stuck watchdog: no progress for a while -> abort so the brain can retry
            if (Math.abs(player.getX() - lastMoveX) < 0.05 && Math.abs(player.getZ() - lastMoveZ) < 0.05) {
                if (++lastPosTick > 40) { route.clear(); lastPosTick = 0; finish(false, "stuck"); return; }
            } else {
                lastPosTick = 0;
                lastMoveX = player.getX();
                lastMoveZ = player.getZ();
            }
            if (pathMode && routeIdx < route.size()) {
                int[] wp = route.get(routeIdx);
                double wdx = wp[0] + 0.5 - player.getX();
                double wdz = wp[2] + 0.5 - player.getZ();
                if (Math.hypot(wdx, wdz) < 0.6) { routeIdx++; }
                else { walkStep(wdx, wdz); activeTicks++; if (activeTicks > 600) { route.clear(); finish(false, "timeout"); } return; }
            }
            walkStep(dx, dz);
            activeTicks++;
            if (activeTicks > 600) finish(false, "timeout");
        }

        void walkStep(double dx, double dz) {
            double dist = Math.hypot(dx, dz);
            if (dist < 1e-4) return;
            if (inWater()) { swimStep(dx, dz); return; }
            faceTowards(dx, dz);
            try { player.setSprinting(true); } catch (Throwable ignored) {}
            Vec3 d = player.getDeltaMovement();
            double speed = 0.14;
            double tx = dx / dist * speed;
            double tz = dz / dist * speed;
            double nvx = (d.x + (tx - d.x) * 0.30) * 0.98;
            double nvz = (d.z + (tz - d.z) * 0.30) * 0.98;
            double nvy = Math.max(d.y - 0.08, -0.7);
            player.setDeltaMovement(nvx, nvy, nvz);
            if (player.horizontalCollision && player.onGround()) {
                player.setDeltaMovement(nvx, 0.42, nvz);
                pushEvent("jump", 0, null);
            }
            player.move(MoverType.SELF, player.getDeltaMovement());
        }

        void faceTowards(double dx, double dz) {
            float yaw = (float) Math.toDegrees(Math.atan2(-dx, dz));
            player.setYRot(yaw);
            player.setYHeadRot(yaw);
            player.yBodyRot = yaw;
        }

        boolean inWater() {
            if (player == null) return false;
            try {
                if (player.isInWater() || player.isInWaterOrBubble()) return true;
                return level.getFluidState(player.blockPosition()).is(net.minecraft.tags.FluidTags.WATER);
            } catch (Throwable e) { return false; }
        }

        void swimStep(double dx, double dz) {
            double dist = Math.hypot(dx, dz);
            if (dist < 1e-4) return;
            faceTowards(dx, dz);
            player.setSwimming(true);
            Vec3 d = player.getDeltaMovement();
            double tx = dx / dist * 0.07;
            double tz = dz / dist * 0.07;
            double nvx = (d.x + (tx - d.x) * 0.25) * 0.85;
            double nvz = (d.z + (tz - d.z) * 0.25) * 0.85;
            boolean submerged = level.getFluidState(player.blockPosition().above()).is(net.minecraft.tags.FluidTags.WATER);
            double nvy = submerged ? 0.09 : 0.015;
            player.setDeltaMovement(nvx, nvy, nvz);
            player.move(MoverType.SELF, player.getDeltaMovement());
        }

        // ------------------------------------------------------------- A* --
        static final int[] DX = {1, -1, 0, 0, 1, 1, -1, -1};
        static final int[] DZ = {0, 0, 1, -1, 1, -1, 1, -1};

        static long key(int x, int z) {
            return (((long) (x + 30000000)) << 32) | ((z + 30000000) & 0xFFFFFFFFL);
        }

        boolean passable(BlockPos p) {
            BlockState st = bs(p);
            if (st == null) return false;
            try { return st.getCollisionShape(level, p).isEmpty(); } catch (Throwable e) { return st.isAir(); }
        }

        boolean standable(BlockPos p) {
            if (!passable(p)) return false;
            if (!passable(p.above())) return false;
            BlockState below = bs(p.below());
            if (below == null) return false;
            try { return below.isFaceSturdy(level, p.below(), Direction.UP); } catch (Throwable e) { return false; }
        }

        /** Compact A* over the block grid. Budgeted so it stays inside one tick. */
        List<int[]> aStar(BlockPos start, BlockPos goal) {
            int limit = 900;
            int maxRange = 48;
            Map<Long, Double> g = new HashMap<>();
            Map<Long, BlockPos> came = new HashMap<>();
            PriorityQueue<double[]> open = new PriorityQueue<>(Comparator.comparingDouble(a -> a[0]));
            g.put(key(start.getX(), start.getZ()), 0.0);
            open.add(new double[]{heur(start, goal), key(start.getX(), start.getZ()), 0, 0, 0});
            Map<Long, BlockPos> node = new HashMap<>();
            node.put(key(start.getX(), start.getZ()), start);
            int expanded = 0;
            boolean found = false;
            while (!open.isEmpty() && expanded < limit) {
                double[] cur = open.poll();
                long ck = (long) cur[1];
                BlockPos cp = node.get(ck);
                if (cp == null) continue;
                if (cp.equals(goal)) { found = true; break; }
                expanded++;
                double cg = g.getOrDefault(ck, Double.MAX_VALUE);
                for (int i = 0; i < 8; i++) {
                    int nx = cp.getX() + DX[i];
                    int nz = cp.getZ() + DZ[i];
                    if (Math.abs(nx - start.getX()) > maxRange || Math.abs(nz - start.getZ()) > maxRange) continue;
                    BlockPos np = new BlockPos(nx, cp.getY(), nz);
                    BlockPos up = np.above();
                    BlockPos down = np.below();
                    boolean ok = standable(np);
                    if (!ok && standable(down)) { np = down; ok = true; }
                    if (!ok && standable(up)) { np = up; ok = true; }
                    if (!ok) continue;
                    long nk = key(np.getX(), np.getZ());
                    double step = (i < 4) ? 1.0 : 1.414;
                    double ng = cg + step;
                    if (ng < g.getOrDefault(nk, Double.MAX_VALUE)) {
                        g.put(nk, ng);
                        came.put(nk, cp);
                        node.put(nk, np);
                        open.add(new double[]{ng + heur(np, goal), nk, np.getX(), np.getY(), np.getZ()});
                    }
                }
            }
            if (!found) return null;
            List<int[]> path = new ArrayList<>();
            BlockPos cur = goal;
            int guard = 0;
            while (cur != null && !cur.equals(start) && guard++ < 200) {
                path.add(new int[]{cur.getX(), cur.getY(), cur.getZ()});
                cur = came.get(key(cur.getX(), cur.getZ()));
            }
            java.util.Collections.reverse(path);
            return path;
        }

        static double heur(BlockPos a, BlockPos b) {
            double dx = Math.abs(a.getX() - b.getX());
            double dz = Math.abs(a.getZ() - b.getZ());
            double dy = Math.abs(a.getY() - b.getY());
            return Math.max(dx, dz) + 0.414 * Math.min(dx, dz) + dy * 0.5;
        }

        // -------------------------------------------------------- execution --
        void pollAndExecute() {
            int idx = -1;
            int sq = -1;
            synchronized (incoming) {
                if (!incoming.isEmpty()) {
                    int[] a = incoming.poll();
                    idx = a[0];
                    sq = a.length > 1 ? a[1] : -1;
                }
            }
            if (idx < 0) return;
            if (activeTarget != null && ("goto".equals(activeAction) || "mine".equals(activeAction))) return;
            // the brain picked an index in the candidate list of frame `sq`; using
            // the CURRENT list would apply it to a completely different action
            if (sq >= 0) {
                List<Map<String, Object>> list = candBySeq.get(sq);
                if (list != null) candidates = list;
            }
            actionCount++;
            try { execute(idx); }
            catch (Throwable e) {
                errorCount++;
                lastError.set(String.valueOf(e));
                pushEvent("act_fail", 1, null);
            }
        }

        void execute(int idx) {
            if (idx < 0 || idx >= candidates.size()) { pushEvent("bad_action", 1, null); return; }
            Map<String, Object> c = candidates.get(idx);
            String a = String.valueOf(c.get("a"));
            switch (a) {
                case "goto" -> {
                    activeAction = "goto";
                    activeTarget = new BlockPos(ip(c, "x"), ip(c, "y"), ip(c, "z"));
                    activeTicks = 0;
                    planRoute(activeTarget);
                }
                case "mine" -> {
                    BlockPos p = new BlockPos(ip(c, "x"), ip(c, "y"), ip(c, "z"));
                    if (!p.closerToCenterThan(player.position(), REACH + 1.5)) { pushEvent("act_fail", 1, null); return; }
                    // look at the block and start digging with the tool in hand
                    net.minecraft.world.phys.Vec3 eye = player.getEyePosition();
                    double dx = p.getX() + 0.5 - eye.x, dy = p.getY() + 0.5 - eye.y, dz = p.getZ() + 0.5 - eye.z;
                    double len = Math.sqrt(dx * dx + dy * dy + dz * dz);
                    if (len > 0) {
                        player.setYRot((float) Math.toDegrees(Math.atan2(-dx, dz)));
                        player.setYHeadRot(player.getYRot());
                        player.setXRot((float) Math.toDegrees(-Math.atan2(dy, Math.sqrt(dx * dx + dz * dz))));
                    }
                    miningPos = p;
                    miningDir = Direction.UP;
                    miningTicks = 0;
                    miningProgress = 0.0f;
                    activeAction = "mine";
                    activeTarget = p;
                    activeTicks = 0;
                }
                case "attack" -> {
                    int eid = ip(c, "eid");
                    Entity e = level.getEntity(eid);
                    if (e == null || !e.isAlive() || e.distanceTo(player) > REACH + 1.5) { pushEvent("act_fail", 1, null); return; }
                    player.attack(e);
                    lastAttackedId = e.getId();
                    lastAttackedName = BuiltInRegistries.ENTITY_TYPE.getKey(e.getType()).toString();
                    pushEvent("attack", 1, null);
                    finish(true, "attacked");
                }
                case "use" -> {
                    BlockPos p = new BlockPos(ip(c, "x"), ip(c, "y"), ip(c, "z"));
                    BlockHitResult hit = new BlockHitResult(
                        new Vec3(p.getX() + 0.5, p.getY() + 0.5, p.getZ() + 0.5),
                        Direction.UP, p, false);
                    player.gameMode.useItemOn(player, level, player.getMainHandItem(), InteractionHand.MAIN_HAND, hit);
                    pushEvent("use", 1, p);
                    finish(true, "used");
                }
                case "place" -> {
                    BlockPos p = new BlockPos(ip(c, "x"), ip(c, "y"), ip(c, "z"));
                    BlockHitResult hit = new BlockHitResult(
                        new Vec3(p.getX() + 0.5, p.getY(), p.getZ() + 0.5),
                        Direction.UP, p.below(), false);
                    player.gameMode.useItemOn(player, level, player.getMainHandItem(), InteractionHand.MAIN_HAND, hit);
                    pushEvent("place", 1, p);
                    finish(true, "placed");
                }
                case "eat" -> {
                    ItemStack food = findFoodItem();
                    if (food == null) { pushEvent("act_fail", 1, null); return; }
                    Inventory inv = player.getInventory();
                    int slot = inv.selected;
                    ItemStack old = inv.getItem(slot);
                    inv.setItem(slot, food.copy());
                    player.gameMode.useItem(player, level, inv.getItem(slot), InteractionHand.MAIN_HAND);
                    if (!inv.getItem(slot).isEmpty()) {
                        for (int i = 0; i < inv.getContainerSize(); i++) {
                            if (inv.getItem(i).isEmpty() && !old.isEmpty()) { inv.setItem(i, old); break; }
                        }
                    }
                    pushEvent("eat", 1, null);
                    finish(true, "ate");
                }
                case "jump" -> {
                    player.setDeltaMovement(player.getDeltaMovement().x, 0.42, player.getDeltaMovement().z);
                    finish(true, "jumped");
                }
                case "wait" -> finish(true, "waited");
                case "stop" -> finish(true, "stopped");
                default -> pushEvent("act_fail", 1, null);
            }
        }

        void finish(boolean ok, String reason) {
            activeAction = "none";
            activeTarget = null;
            activeTicks = 0;
            activeOk = ok;
        }

        void pushEvent(String name, int mag, BlockPos p) {
            if (events.size() >= EV_MAX) return;
            Map<String, Object> m = new HashMap<>();
            m.put("e", name);
            m.put("m", mag);
            if (p != null) { m.put("x", p.getX()); m.put("y", p.getY()); m.put("z", p.getZ()); }
            events.add(m);
        }

        // ------------------------------------------------------- observation --
        BlockState bs(BlockPos p) {
            int cx = p.getX() >> 4, cz = p.getZ() >> 4;
            if (cx != curCX || cz != curCZ) {
                curCX = cx; curCZ = cz;
                try { curChunk = level.getChunkSource().getChunk(cx, cz, false); }
                catch (Throwable e) { curChunk = null; }
            }
            if (curChunk != null) {
                try { return curChunk.getBlockState(p); } catch (Throwable e) { }
            }
            gridFallbacks++;
            try { return level.getBlockState(p); } catch (Throwable e) { return null; }
        }

        void sendObservation() {
            if (player == null || out == null) return;
            long t0 = System.nanoTime();
            Vec3 pos = player.position();
            int bx = (int) Math.floor(pos.x), by = (int) Math.floor(pos.y), bz = (int) Math.floor(pos.z);

            Map<String, Object> self = new HashMap<>();
            Vec3 v = player.getDeltaMovement();
            self.put("x", pos.x); self.put("y", pos.y); self.put("z", pos.z);
            self.put("vx", v.x); self.put("vy", v.y); self.put("vz", v.z);
            self.put("yaw", player.getYRot()); self.put("pitch", player.getXRot());
            self.put("hp", player.getHealth());
            self.put("food", player.getFoodData().getFoodLevel());
            self.put("air", player.getAirSupply());
            self.put("sat", player.getFoodData().getSaturationLevel());
            self.put("light", level.getMaxLocalRawBrightness(new BlockPos(bx, by, bz)));
            self.put("dim", level.dimension().location().toString());
            self.put("on_ground", player.onGround());
            self.put("in_water", inWater());
            self.put("in_lava", player.isInLava());
            self.put("sprinting", player.isSprinting());
            self.put("sneaking", player.isCrouching());
            self.put("flying", player.getAbilities().flying);
            self.put("fall", player.fallDistance);
            self.put("collided", player.horizontalCollision);
            self.put("time", level.getDayTime() % 24000);
            self.put("raining", level.isRaining());
            self.put("thunder", level.isThundering());
            long tSelf = System.nanoTime();

            boolean rebuild = (seq % 2) == 0;
            int[] ids = gridCache;
            if (rebuild) {
                int k = 0;
                for (int ix = 0; ix < GRID_N; ix++) {
                    for (int iy = 0; iy < GRID_N; iy++) {
                        for (int iz = 0; iz < GRID_N; iz++) {
                            BlockPos p = new BlockPos(bx - 7 + ix * GRID_STEP, by - 7 + iy * GRID_STEP, bz - 7 + iz * GRID_STEP);
                            BlockState st = bs(p);
                            int id = 0;
                            if (st != null && !st.isAir()) {
                                id = BuiltInRegistries.BLOCK.getId(st.getBlock());
                                if (!blockNames.containsKey(id)) {
                                    blockNames.put(id, BuiltInRegistries.BLOCK.getKey(st.getBlock()).toString());
                                }
                            }
                            gridCache[k++] = id;
                        }
                    }
                }
            }
            long tGrid = System.nanoTime();

            AABB area = new AABB(bx - 24, by - 24, bz - 24, bx + 24, by + 24, bz + 24);
            List<Entity> ents = new ArrayList<>(level.getEntitiesOfClass(Entity.class, area,
                e -> e != player && e.isAlive()));
            ents.sort(Comparator.comparingDouble(e -> e.distanceToSqr(player)));
            List<List<Object>> entOut = new ArrayList<>();
            for (Entity e : ents) {
                if (entOut.size() >= MAX_ENT) break;
                List<Object> row = new ArrayList<>();
                row.add(e.getId());
                row.add(round(e.getX() - pos.x));
                row.add(round(e.getY() - pos.y));
                row.add(round(e.getZ() - pos.z));
                row.add(round(e.distanceTo(player)));
                row.add(e instanceof net.minecraft.world.entity.LivingEntity le ? le.getHealth() : -1.0);
                row.add(e instanceof Enemy ? 1 : 0);
                entOut.add(row);
            }
            long tEnt = System.nanoTime();

            Inventory inv = player.getInventory();
            List<List<Object>> invOut = new ArrayList<>();
            for (int i = 0; i < inv.getContainerSize(); i++) {
                ItemStack st = inv.getItem(i);
                if (st.isEmpty()) continue;
                List<Object> row = new ArrayList<>();
                row.add(i);
                row.add(BuiltInRegistries.ITEM.getId(st.getItem()));
                row.add(st.getCount());
                row.add(st.get(net.minecraft.core.component.DataComponents.FOOD) != null ? 1 : 0);
                invOut.add(row);
            }
            long tInv = System.nanoTime();

            if (lastAttackedId >= 0 && level.getEntity(lastAttackedId) == null) {
                Map<String, Object> kill = new HashMap<>();
                kill.put("e", "kill"); kill.put("m", 1); kill.put("n", lastAttackedName);
                events.add(kill);
                lastAttackedId = -1;
            }

            List<Map<String, Object>> cands = buildCandidates(bx, by, bz);
            candidates = cands;
            candBySeq.put(seq, cands);
            while (candBySeq.size() > 12) {
                java.util.Iterator<Integer> it = candBySeq.keySet().iterator();
                it.next();
                it.remove();
            }
            long tCand = System.nanoTime();

            Map<String, Object> frame = new HashMap<>();
            frame.put("t", "obs");
            frame.put("seq", ++seq);
            frame.put("tick", tickCounter);
            frame.put("ms", System.currentTimeMillis());
            frame.put("self", self);
            Map<String, Object> grid = new HashMap<>();
            grid.put("ox", bx - 7); grid.put("oy", by - 7); grid.put("oz", bz - 7);
            grid.put("step", GRID_STEP); grid.put("n", GRID_N); grid.put("ids", ids);
            frame.put("grid", grid);
            frame.put("ent", entOut);
            frame.put("inv", invOut);
            frame.put("cand", cands);
            List<Map<String, Object>> ev;
            synchronized (events) { ev = new ArrayList<>(events); events.clear(); }
            frame.put("ev", ev);
            frame.put("bmap", new ArrayList<>());
            frame.put("prev", Map.of("a", activeAction, "ticks", activeTicks, "ok", activeOk));

            long tJson = System.nanoTime();
            try {
                PrintWriter w = out;
                if (w != null) {
                    w.println(GSON.toJson(frame));
                    if (w.checkError()) lastError.set("write error");
                }
            } catch (Throwable e) { errorCount++; }
            long total = System.nanoTime() - t0;
            if (seq % 10 == 0) {
                System.out.println(String.format(Locale.ROOT,
                    "[soma] obs #%d self=%.1fms grid=%.1fms ent=%.1fms inv=%.1fms cand=%.1fms json=%.1fms total=%.1fms cands=%d fb=%d",
                    seq, (tSelf - t0) / 1e6, (tGrid - tSelf) / 1e6, (tEnt - tGrid) / 1e6,
                    (tInv - tEnt) / 1e6, (tCand - tInv) / 1e6, (tJson - tCand) / 1e6,
                    total / 1e6, cands.size(), gridFallbacks));
            }
        }

        // -------------------------------------------------------- candidates --
        List<Map<String, Object>> buildCandidates(int bx, int by, int bz) {
            List<Map<String, Object>> out = new ArrayList<>();

            List<Map<String, Object>> mine = new ArrayList<>();
            for (int dx = -4; dx <= 4; dx++) {
                for (int dy = -3; dy <= 4; dy++) {
                    for (int dz = -4; dz <= 4; dz++) {
                        BlockPos p = new BlockPos(bx + dx, by + dy, bz + dz);
                        BlockState st = bs(p);
                        if (st == null || st.isAir() || st.getFluidState().isSource()) continue;
                        if (st.getDestroySpeed(level, p) < 0) continue;
                        double d = Math.sqrt(dx * dx + dy * dy + dz * dz);
                        if (d > REACH) continue;
                        Map<String, Object> m = new HashMap<>();
                        m.put("a", "mine"); m.put("x", p.getX()); m.put("y", p.getY()); m.put("z", p.getZ());
                        m.put("n", BuiltInRegistries.BLOCK.getKey(st.getBlock()).toString());
                        m.put("d", round(d));
                        mine.add(m);
                    }
                }
            }
            mine.sort(Comparator.comparingDouble(m -> (double) m.get("d")));
            out.addAll(mine.subList(0, Math.min(8, mine.size())));

            AABB area = new AABB(bx - 32, by - 12, bz - 32, bx + 32, by + 12, bz + 32);
            List<Entity> hostiles = new ArrayList<>(level.getEntitiesOfClass(Entity.class, area,
                e -> e != player && e.isAlive() && e instanceof Enemy));
            hostiles.sort(Comparator.comparingDouble(e -> e.distanceToSqr(player)));
            int added = 0;
            for (Entity e : hostiles) {
                if (added >= 3) break;
                if (e.distanceTo(player) > REACH) continue;
                Map<String, Object> m = new HashMap<>();
                m.put("a", "attack"); m.put("eid", e.getId());
                m.put("n", BuiltInRegistries.ENTITY_TYPE.getKey(e.getType()).toString());
                m.put("d", round(e.distanceTo(player)));
                m.put("h", 1);
                out.add(m);
                added++;
            }
            if (!hostiles.isEmpty()) {
                Entity h = hostiles.get(0);
                double dist = h.distanceTo(player);
                if (dist > REACH && dist < 32) {
                    Map<String, Object> m = new HashMap<>();
                    m.put("a", "goto");
                    m.put("x", (int) Math.floor(h.getX()));
                    m.put("y", (int) Math.floor(h.getY()));
                    m.put("z", (int) Math.floor(h.getZ()));
                    m.put("d", round(dist));
                    out.add(m);
                }
            }

            int used = 0;
            for (Map<String, Object> m : mine) {
                if (used >= 3) break;
                String n = String.valueOf(m.get("n"));
                if (!n.matches(".*(crafting_table|furnace|chest|barrel|door|lever|button|bed|anvil|loom|smithing|brewing|composter|beacon|shulker|hopper|dispenser|dropper|campfire|bell|jukebox|enchanting|bookshelf|cauldron).*")) continue;
                Map<String, Object> c = new HashMap<>(m);
                c.put("a", "use");
                out.add(c);
                used++;
            }

            ItemStack held = player.getMainHandItem();
            if (!held.isEmpty()) {
                for (int i = 0; i < 8; i++) {
                    double ang = i * Math.PI / 4;
                    int px = (int) Math.floor(player.getX() + Math.sin(ang) * 2);
                    int pz = (int) Math.floor(player.getZ() + Math.cos(ang) * 2);
                    BlockState tgt = bs(new BlockPos(px, by, pz));
                    if (tgt == null || !tgt.isAir()) continue;
                    BlockState ref = bs(new BlockPos(px, by - 1, pz));
                    if (ref == null || !ref.isFaceSturdy(level, new BlockPos(px, by - 1, pz), Direction.UP)) continue;
                    Map<String, Object> m = new HashMap<>();
                    m.put("a", "place"); m.put("x", px); m.put("y", by); m.put("z", pz);
                    m.put("n", held.getHoverName().getString()); m.put("d", 2);
                    out.add(m);
                    break;
                }
            }

            for (int i = 0; i < 8; i++) {
                double ang = i * Math.PI / 4;
                double sx = Math.sin(ang), sz = Math.cos(ang);
                for (double dist : new double[]{2, 4, 7}) {
                    int gx = (int) Math.floor(player.getX() + sx * dist);
                    int gz = (int) Math.floor(player.getZ() + sz * dist);
                    int gy = standY(gx, by, gz);
                    if (gy == Integer.MIN_VALUE) continue;
                    Map<String, Object> m = new HashMap<>();
                    m.put("a", "goto"); m.put("x", gx); m.put("y", gy); m.put("z", gz); m.put("d", dist);
                    out.add(m);
                }
            }

            if (player.getFoodData().getFoodLevel() < 16 && findFoodItem() != null) {
                Map<String, Object> m = new HashMap<>();
                m.put("a", "eat"); m.put("d", 0);
                out.add(m);
            }
            Map<String, Object> jump = new HashMap<>();
            jump.put("a", "jump"); jump.put("d", 0);
            out.add(jump);
            Map<String, Object> wait = new HashMap<>();
            wait.put("a", "wait"); wait.put("d", 0);
            out.add(wait);
            Map<String, Object> stop = new HashMap<>();
            stop.put("a", "stop"); stop.put("d", 0);
            out.add(stop);

            if (out.size() > MAX_CAND) out = out.subList(0, MAX_CAND);
            return out;
        }

        ItemStack findFoodItem() {
            try {
                Inventory inv = player.getInventory();
                for (int i = 0; i < inv.getContainerSize(); i++) {
                    ItemStack st = inv.getItem(i);
                    if (!st.isEmpty() && st.get(net.minecraft.core.component.DataComponents.FOOD) != null) return st;
                }
            } catch (Throwable ignored) {}
            return null;
        }

        int standY(int x, int nearY, int z) {
            for (int dy = 3; dy >= -4; dy--) {
                int y = nearY + dy;
                BlockPos p = new BlockPos(x, y, z);
                BlockState below = bs(p.below());
                if (below == null || below.isAir()) continue;
                try { if (!below.isFaceSturdy(level, p.below(), Direction.UP)) continue; } catch (Throwable e) { continue; }
                BlockState here = bs(p);
                BlockState above = bs(p.above());
                if (here == null || !here.isAir()) continue;
                if (above == null || !above.isAir()) continue;
                return y;
            }
            return Integer.MIN_VALUE;
        }

        int findGroundY(int x, int z) {
            try {
                for (int y = 320; y > -64; y--) {
                    BlockPos p = new BlockPos(x, y, z);
                    BlockState st = bs(p);
                    if (st == null || st.isAir()) continue;
                    try { if (st.isFaceSturdy(level, p, Direction.UP)) return y + 1; } catch (Throwable ignored) {}
                }
            } catch (Throwable ignored) {}
            return 100;
        }

        static int ip(Map<String, Object> m, String k) {
            Object o = m.get(k);
            return o instanceof Number ? ((Number) o).intValue() : 0;
        }

        static double round(double v) { return Math.round(v * 100000.0) / 100000.0; }
    }
}
