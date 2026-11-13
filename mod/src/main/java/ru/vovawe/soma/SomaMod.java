package ru.vovawe.soma;

import com.google.gson.Gson;
import com.mojang.authlib.GameProfile;
import net.minecraft.core.BlockPos;
import net.minecraft.core.Direction;
import net.minecraft.core.component.DataComponents;
import net.minecraft.core.registries.BuiltInRegistries;
import net.minecraft.network.Connection;
import net.minecraft.network.chat.Component;
import net.minecraft.network.protocol.PacketFlow;
import net.minecraft.server.MinecraftServer;
import net.minecraft.server.level.ServerLevel;
import net.minecraft.tags.BlockTags;
import net.minecraft.tags.FluidTags;
import net.minecraft.world.InteractionHand;
import net.minecraft.world.entity.Entity;
import net.minecraft.world.entity.LivingEntity;
import net.minecraft.world.entity.Mob;
import net.minecraft.world.entity.MobCategory;
import net.minecraft.world.entity.MoverType;
import net.minecraft.world.entity.NeutralMob;
import net.minecraft.world.entity.item.ItemEntity;
import net.minecraft.world.entity.monster.Enemy;
import net.minecraft.world.entity.player.Inventory;
import net.minecraft.world.entity.player.Player;
import net.minecraft.world.entity.projectile.Projectile;
import net.minecraft.world.item.ArmorItem;
import net.minecraft.world.item.BlockItem;
import net.minecraft.world.item.DiggerItem;
import net.minecraft.world.item.Equipable;
import net.minecraft.world.item.Item;
import net.minecraft.world.item.ItemStack;
import net.minecraft.world.item.SwordItem;
import net.minecraft.world.item.TieredItem;
import net.minecraft.world.item.crafting.CraftingRecipe;
import net.minecraft.world.item.crafting.Ingredient;
import net.minecraft.world.item.crafting.RecipeHolder;
import net.minecraft.world.item.crafting.RecipeType;
import net.minecraft.world.level.GameRules;
import net.minecraft.world.level.GameType;
import net.minecraft.world.level.block.BedBlock;
import net.minecraft.world.level.block.Blocks;
import net.minecraft.world.level.block.ButtonBlock;
import net.minecraft.world.level.block.CraftingTableBlock;
import net.minecraft.world.level.block.DoorBlock;
import net.minecraft.world.level.block.FenceGateBlock;
import net.minecraft.world.level.block.FurnaceBlock;
import net.minecraft.world.level.block.LeverBlock;
import net.minecraft.world.level.block.TrapDoorBlock;
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
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Queue;
import java.util.UUID;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;

/**
 * SOMA environment adapter v2 (NeoForge 1.21.1).
 *
 * Full real survival (damage, drowning, hunger, fire, armor, pickups, death/respawn),
 * universal modpack support (dynamic RecipeManager crafting, MenuProvider/BlockEntity
 * interaction, modded entity & item classification), dense 1-block local voxel grid,
 * interruptible A* movement, and high-speed cached observation streaming.
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
        System.out.println("[soma] mod v2 constructed");
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
                .then(net.minecraft.commands.Commands.literal("spawnmob")
                    .executes(c -> { if (agent != null) agent.spawnTestMob(); return 1; }))
                .then(net.minecraft.commands.Commands.literal("hurt")
                    .executes(c -> { if (agent != null) agent.debugHurt(4.0f); return 1; }))
                .then(net.minecraft.commands.Commands.literal("cleanup")
                    .executes(c -> { if (agent != null) agent.cleanupStands(true); return 1; }))
        );
    }

    // =====================================================================
    //  Agent
    // =====================================================================
    static final class Agent {
        static final int GRID_N = 8;
        static final int GRID_STEP = 1;
        static final int GRID_CELLS = GRID_N * GRID_N * GRID_N;
        static final int MAX_CAND = 48;
        static final int MAX_ENT = 16;
        static final int EV_MAX = 10;
        static final int OBS_TICKS = 4;   // 5 Hz decision rate

        static final double REACH = 4.5;
        static final String HOST = "127.0.0.1";
        static final int PORT = Integer.parseInt(System.getenv().getOrDefault("SOMA_IPC_PORT", "8788"));

        final MinecraftServer server;
        ServerLevel level;
        FakePlayer player;

        final AtomicBoolean enabled = new AtomicBoolean(false);
        final AtomicBoolean running = new AtomicBoolean(false);
        final AtomicReference<String> lastError = new AtomicReference<>("not started");
        final AtomicReference<Thread> netThread = new AtomicReference<>();

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
        List<Map<String, Object>> candidates = new ArrayList<>();
        final Map<Integer, List<Map<String, Object>>> candBySeq = new LinkedHashMap<>();
        final Map<String, RecipeHolder<CraftingRecipe>> craftableById = new HashMap<>();

        int lastPosTick = 0;
        double lastMoveX = 0, lastMoveZ = 0;

        BlockPos miningPos = null;
        int miningTicks = 0;
        float miningProgress = 0.0f;

        double fallStartY = -1.0;
        float lastHp = 20.0f;
        int lastFood = 20;
        int lavaTick = 0;
        int drownTick = 0;
        int suffocateTick = 0;

        int lastAttackedId = -1;
        String lastAttackedName = "";
        String lastMobInfo = "";

        // 3x3 Chunk cache
        final net.minecraft.world.level.chunk.LevelChunk[] chunkCache = new net.minecraft.world.level.chunk.LevelChunk[9];
        int cacheCX = Integer.MIN_VALUE, cacheCZ = Integer.MIN_VALUE;
        int gridFallbacks = 0;
        final int[] gridIds = new int[GRID_CELLS];
        final int[] gridFlags = new int[GRID_CELLS];

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
                    + " hp=" + (player == null ? 0 : String.format(Locale.ROOT, "%.1f", player.getHealth()))
                    + " food=" + (player == null ? 0 : player.getFoodData().getFoodLevel())
                    + " air=" + (player == null ? 0 : player.getAirSupply())
                    + " inv=" + (player == null ? 0 : countItems())
                    + " gm=" + (player == null ? "-" : player.gameMode.getGameModeForPlayer())
                    + " mobs=" + (player == null ? 0 : countHostiles())
                    + " inTab=" + (player != null && server.getPlayerList().getPlayers().contains(player))
                    + " errors=" + errorCount + " lastError=" + lastError.get()), true);
        }

        void spawnTestMob() {
            if (player == null || level == null) return;
            try {
                Mob m = net.minecraft.world.entity.EntityType.ZOMBIE.create(level);
                m.moveTo(player.getX() + 2.0, player.getY(), player.getZ());
                level.addFreshEntity(m);
            } catch (Throwable e) {
                System.out.println("[soma] spawn failed: " + e);
            }
        }

        void debugHurt(float amount) {
            if (player == null || level == null) return;
            player.invulnerableTime = 0;
            boolean ok = player.hurt(level.damageSources().generic(), amount);
            System.out.println("[soma] debug hurt ok=" + ok + " hp=" + player.getHealth());
        }

        int countHostiles() {
            try {
                AABB a = player.getBoundingBox().inflate(16.0);
                List<Entity> l = level.getEntitiesOfClass(Entity.class, a,
                    e -> e != player && e.isAlive() && isHostile(e));
                StringBuilder sb = new StringBuilder();
                for (Entity e : l) {
                    Entity tgt = (e instanceof Mob m) ? m.getTarget() : null;
                    sb.append(BuiltInRegistries.ENTITY_TYPE.getKey(e.getType()))
                      .append("->").append(tgt == player ? "SOMA" : (tgt == null ? "none" : "other"))
                      .append(' ');
                }
                lastMobInfo = sb.toString();
                return l.size();
            } catch (Throwable e) { return -1; }
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
                        lastError.set("connected");
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
                    @Override public boolean isInvulnerable() { return false; }
                    @Override public void tick() { /* driven by Agent.tick() */ }
                    @Override public boolean isSpectator() { return false; }
                    @Override public boolean isInvisible() { return false; }
                    @Override public boolean canBeSeenAsEnemy() { return isAlive(); }
                };
                BlockPos spawn = level.getSharedSpawnPos();
                player.setPos(spawn.getX() + 0.5, findGroundY(spawn.getX(), spawn.getZ()), spawn.getZ() + 0.5);
                setSurvival();
                attachToServer();
                player.invulnerableTime = 0;
                giveStarterKitIfEmpty();
                lastHp = player.getHealth();
                lastFood = player.getFoodData().getFoodLevel();
                System.out.println("[soma] player ready at " + player.getX() + "," + player.getY()
                    + "," + player.getZ() + " visible=" + playerVisible);
            }
            if (player != null && player.isRemoved()) { player = null; playerVisible = false; }
        }

        void setSurvival() {
            try { player.setGameMode(GameType.SURVIVAL); } catch (Throwable ignored) {}
            try {
                player.getAbilities().instabuild = false;
                player.getAbilities().mayfly = false;
                player.getAbilities().flying = false;
                player.getAbilities().invulnerable = false;
                player.setInvulnerable(false);
            } catch (Throwable ignored) {}
        }

        void giveStarterKitIfEmpty() {
            try {
                Inventory inv = player.getInventory();
                boolean empty = true;
                for (int i = 0; i < inv.getContainerSize(); i++) {
                    if (!inv.getItem(i).isEmpty()) { empty = false; break; }
                }
                if (empty) {
                    inv.setItem(0, new ItemStack(net.minecraft.world.item.Items.WOODEN_PICKAXE));
                    inv.setItem(1, new ItemStack(net.minecraft.world.item.Items.BREAD, 3));
                    inv.setItem(2, new ItemStack(net.minecraft.world.item.Items.TORCH, 8));
                    inv.selected = 0;
                }
            } catch (Throwable ignored) {}
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
                return;
            } catch (Throwable e) {
                System.out.println("[soma] placeNewPlayer failed (" + e + "), world-only fallback");
            }
            try {
                level.addNewPlayer(player);
                playerVisible = true;
            } catch (Throwable e2) {
                playerVisible = false;
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
            } catch (Throwable e) {
                System.out.println("[soma] channel registration warning: " + e);
            }
        }

        // ----------------------------------------------------------- survival --
        void handleDeathAndRespawn() {
            if (player == null) return;
            if (player.getHealth() > 0.0f && !player.isDeadOrDying()) return;

            pushEvent("death", 1, player.blockPosition());
            boolean keepInv = false;
            try { keepInv = level.getGameRules().getBoolean(GameRules.RULE_KEEPINVENTORY); } catch (Throwable ignored) {}
            if (!keepInv) {
                try { player.getInventory().dropAll(); } catch (Throwable ignored) {}
            }
            try { player.removeAllEffects(); } catch (Throwable ignored) {}
            player.setHealth(player.getMaxHealth());
            player.getFoodData().setFoodLevel(20);
            player.getFoodData().setSaturation(5.0f);
            player.setAirSupply(player.getMaxAirSupply());
            player.clearFire();
            player.invulnerableTime = 20;
            fallStartY = -1.0;
            miningPos = null;
            route.clear();
            finish(false, "died");

            BlockPos sp = player.getRespawnPosition();
            if (sp == null) sp = level.getSharedSpawnPos();
            player.setPos(sp.getX() + 0.5, findGroundY(sp.getX(), sp.getZ()), sp.getZ() + 0.5);
            player.setDeltaMovement(0, 0, 0);
            lastHp = player.getHealth();
            System.out.println("[soma] agent died -> respawned at " + sp);
        }

        void pickupNearbyItems() {
            if (player == null || level == null) return;
            try {
                AABB box = player.getBoundingBox().inflate(1.2, 0.7, 1.2);
                List<ItemEntity> items = level.getEntitiesOfClass(ItemEntity.class, box,
                    ie -> ie.isAlive() && !ie.hasPickUpDelay());
                for (ItemEntity ie : items) {
                    ItemStack st = ie.getItem();
                    if (st.isEmpty()) continue;
                    String nm = BuiltInRegistries.ITEM.getKey(st.getItem()).toString();
                    int before = st.getCount();
                    ie.playerTouch(player);
                    int after = ie.isRemoved() ? 0 : ie.getItem().getCount();
                    if (after < before) {
                        pushEvent("pickup", before - after, ie.blockPosition(), nm);
                        autoEquipArmor(st);
                    }
                }
            } catch (Throwable ignored) {}
        }

        void autoEquipArmor(ItemStack picked) {
            try {
                Inventory inv = player.getInventory();
                for (int i = 0; i < 36; i++) {
                    ItemStack st = inv.getItem(i);
                    if (st.isEmpty()) continue;
                    if (st.getItem() instanceof ArmorItem armor) {
                        int slotIdx = armor.getEquipmentSlot().getIndex();
                        ItemStack cur = inv.armor.get(slotIdx);
                        if (cur.isEmpty()) {
                            inv.armor.set(slotIdx, st.copy());
                            inv.setItem(i, ItemStack.EMPTY);
                        }
                    }
                }
            } catch (Throwable ignored) {}
        }

        void cleanupStands() { cleanupStands(false); }

        void cleanupStands(boolean verbose) {
            if (cleaned >= 0 && !verbose) return;
            cleaned = 0;
            try {
                AABB box = new AABB(-3000, -64, -3000, 3000, 320, 3000);
                List<net.minecraft.world.entity.decoration.ArmorStand> stands =
                    level.getEntitiesOfClass(net.minecraft.world.entity.decoration.ArmorStand.class, box);
                int mine = 0;
                for (net.minecraft.world.entity.decoration.ArmorStand st : stands) {
                    Component n = st.getCustomName();
                    if (n != null && "SOMA".equals(n.getString())) { st.discard(); mine++; }
                }
                cleaned = mine;
            } catch (Throwable ignored) {}
        }

        // -------------------------------------------------------------- tick --
        void tick() {
            if (!enabled.get()) return;
            try {
                ensurePlayer();
                if (player == null) return;
                if (cleaned < 0) cleanupStands();
                else if (tickCounter % 1200 == 0) cleanupStands(false);

                // Drive core entity & player ticks
                int tcBefore = player.tickCount;
                try { player.baseTick(); } catch (Throwable ignored) {}
                if (player.tickCount <= tcBefore) player.tickCount = tcBefore + 1;
                try { player.getFoodData().tick(player); } catch (Throwable ignored) {}
                try { player.attackStrengthTicker++; } catch (Throwable ignored) {}

                survivalTick();
                handleDeathAndRespawn();
                pickupNearbyItems();
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

        void survivalTick() {
            if (player == null) return;
            try {
                // Fall damage via real DamageSource
                if (!player.onGround() && !inWater() && !player.isInLava()) {
                    if (fallStartY < 0) fallStartY = player.getY();
                    else fallStartY = Math.max(fallStartY, player.getY());
                } else if (fallStartY >= 0) {
                    double drop = fallStartY - player.getY();
                    fallStartY = -1.0;
                    if (drop > 3.0 && !inWater()) {
                        float dmg = (float) (drop - 3.0);
                        player.hurt(level.damageSources().fall(), dmg);
                        pushEvent("fall", (int) Math.round(dmg), player.blockPosition());
                    }
                }

                // Lava & Fire
                if (player.isInLava()) {
                    player.setRemainingFireTicks(300);
                    if (++lavaTick % 10 == 0) {
                        player.hurt(level.damageSources().lava(), 4.0f);
                        pushEvent("burn", 4, player.blockPosition());
                    }
                } else {
                    lavaTick = 0;
                }

                // Drowning (real air supply!)
                BlockPos eyePos = BlockPos.containing(player.getX(), player.getEyeY(), player.getZ());
                boolean eyeInWater = level.getFluidState(eyePos).is(FluidTags.WATER);
                if (eyeInWater) {
                    int air = player.getAirSupply() - 1;
                    player.setAirSupply(air);
                    if (air <= -20) {
                        player.setAirSupply(0);
                        player.hurt(level.damageSources().drown(), 2.0f);
                        pushEvent("drown", 2, eyePos);
                    }
                } else {
                    player.setAirSupply(Math.min(player.getMaxAirSupply(), player.getAirSupply() + 4));
                }

                // Suffocation in solid block
                BlockState eyeState = bs(eyePos);
                if (eyeState != null && eyeState.isSuffocating(level, eyePos)) {
                    if (++suffocateTick % 20 == 0) {
                        player.hurt(level.damageSources().inWall(), 1.0f);
                    }
                } else {
                    suffocateTick = 0;
                }

                // Void damage
                if (player.getY() < level.getMinBuildHeight() - 32) {
                    player.hurt(level.damageSources().fellOutOfWorld(), 4.0f);
                }

                // Track real HP changes (from mobs, projectiles, fire, starvation, etc.)
                float hp = player.getHealth();
                if (hp < lastHp - 0.05f) {
                    float dmg = lastHp - hp;
                    pushEvent("hurt", Math.max(1, Math.round(dmg)), player.blockPosition());
                }
                lastHp = hp;
            } catch (Throwable e) {
                lastError.set("survival: " + e);
            }
        }

        // ----------------------------------------------------------- mining --
        void selectBestToolFor(BlockState st) {
            Inventory inv = player.getInventory();
            int bestSlot = inv.selected;
            float bestSpeed = inv.getItem(bestSlot).getDestroySpeed(st);
            for (int i = 0; i < 9; i++) {
                ItemStack s = inv.getItem(i);
                if (s.isEmpty()) continue;
                float spd = s.getDestroySpeed(st);
                if (spd > bestSpeed) {
                    bestSpeed = spd;
                    bestSlot = i;
                }
            }
            inv.selected = bestSlot;
        }

        void selectBestWeapon() {
            Inventory inv = player.getInventory();
            int bestSlot = inv.selected;
            double bestScore = weaponScore(inv.getItem(bestSlot));
            for (int i = 0; i < 9; i++) {
                double sc = weaponScore(inv.getItem(i));
                if (sc > bestScore) { bestScore = sc; bestSlot = i; }
            }
            inv.selected = bestSlot;
        }

        static double weaponScore(ItemStack st) {
            if (st.isEmpty()) return 0.0;
            Item it = st.getItem();
            if (it instanceof SwordItem sw) return 10.0 + sw.getTier().getAttackDamageBonus();
            if (it instanceof DiggerItem dg) return 5.0 + dg.getTier().getAttackDamageBonus();
            return 0.5;
        }

        void advanceMining() {
            if (miningPos == null || player == null) return;
            miningTicks++;
            try {
                if (miningTicks > 160 || !miningPos.closerToCenterThan(player.position(), REACH + 1.5)) {
                    miningPos = null;
                    pushEvent("dig_fail", 1, null);
                    finish(false, "dig_timeout");
                    return;
                }
                BlockState st = bs(miningPos);
                if (st == null || st.isAir()) {
                    miningPos = null;
                    finish(true, "air");
                    return;
                }
                selectBestToolFor(st);
                float prog;
                try { prog = st.getDestroyProgress(player, level, miningPos); }
                catch (Throwable e) {
                    float hard = st.getDestroySpeed(level, miningPos);
                    prog = hard > 0 ? 1.0f / (hard * 30.0f) : 1.0f;
                }
                if (miningTicks % 5 == 0) {
                    try { player.swing(InteractionHand.MAIN_HAND); } catch (Throwable ignored) {}
                }
                miningProgress += prog;
                if (miningProgress >= 1.0f) {
                    miningProgress = 0.0f;
                    BlockPos done = miningPos;
                    String bname = BuiltInRegistries.BLOCK.getKey(st.getBlock()).toString();
                    boolean ok = player.gameMode.destroyBlock(done);
                    miningPos = null;
                    try { player.getFoodData().addExhaustion(0.005f); } catch (Throwable ignored) {}
                    pushEvent(ok ? "dig" : "dig_fail", 1, done, bname);
                    finish(ok, "mined");
                }
            } catch (Throwable e) {
                miningPos = null;
                pushEvent("dig_fail", 1, null);
                finish(false, "dig_err");
            }
        }

        // --------------------------------------------------------- movement --
        final List<int[]> route = new ArrayList<>();
        int routeIdx = 0;

        void planRoute(BlockPos goal) {
            route.clear();
            routeIdx = 0;
            pathMode = false;
            try {
                List<int[]> p = aStar(player.blockPosition(), goal);
                if (p != null && !p.isEmpty()) { route.addAll(p); pathMode = true; }
            } catch (Throwable ignored) {}
        }

        void driveActive() {
            if (player == null || activeTarget == null) return;
            if (!"goto".equals(activeAction)) return;

            double dx = (activeTarget.getX() + 0.5) - player.getX();
            double dz = (activeTarget.getZ() + 0.5) - player.getZ();
            if (Math.hypot(dx, dz) < 0.85 && Math.abs(activeTarget.getY() - player.getY()) < 2.0) {
                route.clear();
                finish(true, "arrived");
                return;
            }
            if (Math.abs(player.getX() - lastMoveX) < 0.04 && Math.abs(player.getZ() - lastMoveZ) < 0.04) {
                if (++lastPosTick > 35) {
                    route.clear();
                    lastPosTick = 0;
                    finish(false, "stuck");
                    return;
                }
            } else {
                lastPosTick = 0;
                lastMoveX = player.getX();
                lastMoveZ = player.getZ();
            }

            if (pathMode && routeIdx < route.size()) {
                int[] wp = route.get(routeIdx);
                double wdx = wp[0] + 0.5 - player.getX();
                double wdz = wp[2] + 0.5 - player.getZ();
                if (Math.hypot(wdx, wdz) < 0.6) {
                    routeIdx++;
                } else {
                    walkStep(wdx, wdz);
                    activeTicks++;
                    if (activeTicks > 400) { route.clear(); finish(false, "timeout"); }
                    return;
                }
            }
            walkStep(dx, dz);
            activeTicks++;
            if (activeTicks > 400) finish(false, "timeout");
        }

        void walkStep(double dx, double dz) {
            double dist = Math.hypot(dx, dz);
            if (dist < 1e-4) return;
            if (inWater()) { swimStep(dx, dz); return; }
            faceTowards(dx, dz);
            boolean sprint = player.getFoodData().getFoodLevel() > 6 && !player.isCrouching();
            try { player.setSprinting(sprint); } catch (Throwable ignored) {}
            Vec3 d = player.getDeltaMovement();
            double speed = player.isCrouching() ? 0.065 : (sprint ? 0.16 : 0.12);
            double tx = dx / dist * speed;
            double tz = dz / dist * speed;
            double nvx = (d.x + (tx - d.x) * 0.35) * 0.91;
            double nvz = (d.z + (tz - d.z) * 0.35) * 0.91;
            double nvy = Math.max(d.y - 0.08, -0.78);
            player.setDeltaMovement(nvx, nvy, nvz);
            if (player.horizontalCollision && player.onGround()) {
                player.setDeltaMovement(nvx, 0.42, nvz);
                try { player.getFoodData().addExhaustion(0.05f); } catch (Throwable ignored) {}
                pushEvent("jump", 0, null);
            }
            player.move(MoverType.SELF, player.getDeltaMovement());
            try { player.getFoodData().addExhaustion(sprint ? 0.01f : 0.002f); } catch (Throwable ignored) {}
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
                return level.getFluidState(player.blockPosition()).is(FluidTags.WATER);
            } catch (Throwable e) { return false; }
        }

        void swimStep(double dx, double dz) {
            double dist = Math.hypot(dx, dz);
            if (dist < 1e-4) return;
            faceTowards(dx, dz);
            player.setSwimming(true);
            Vec3 d = player.getDeltaMovement();
            double tx = dx / dist * 0.08;
            double tz = dz / dist * 0.08;
            double nvx = (d.x + (tx - d.x) * 0.25) * 0.85;
            double nvz = (d.z + (tz - d.z) * 0.25) * 0.85;
            boolean submerged = level.getFluidState(player.blockPosition().above()).is(FluidTags.WATER);
            double nvy = submerged ? 0.09 : 0.02;
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
            if (!st.getFluidState().isEmpty() && st.getFluidState().is(FluidTags.LAVA)) return false;
            if (st.is(Blocks.FIRE) || st.is(Blocks.SOUL_FIRE) || st.is(Blocks.CACTUS)) return false;
            try { return st.getCollisionShape(level, p).isEmpty(); } catch (Throwable e) { return st.isAir(); }
        }

        boolean standable(BlockPos p) {
            if (!passable(p) || !passable(p.above())) return false;
            BlockState below = bs(p.below());
            if (below == null) return false;
            if (!below.getFluidState().isEmpty()) return below.getFluidState().is(FluidTags.WATER);
            try { return below.isFaceSturdy(level, p.below(), Direction.UP); } catch (Throwable e) { return false; }
        }

        List<int[]> aStar(BlockPos start, BlockPos goal) {
            int limit = 700;
            int maxRange = 40;
            Map<Long, Double> g = new HashMap<>();
            Map<Long, BlockPos> came = new HashMap<>();
            PriorityQueue<double[]> open = new PriorityQueue<>(Comparator.comparingDouble(a -> a[0]));
            long sk = key(start.getX(), start.getZ());
            g.put(sk, 0.0);
            open.add(new double[]{heur(start, goal), sk});
            Map<Long, BlockPos> node = new HashMap<>();
            node.put(sk, start);
            int expanded = 0;
            BlockPos bestNode = start;
            double bestH = heur(start, goal);

            while (!open.isEmpty() && expanded < limit) {
                double[] cur = open.poll();
                long ck = (long) cur[1];
                BlockPos cp = node.get(ck);
                if (cp == null) continue;
                double ch = heur(cp, goal);
                if (ch < bestH) { bestH = ch; bestNode = cp; }
                if (cp.getX() == goal.getX() && cp.getZ() == goal.getZ() && Math.abs(cp.getY() - goal.getY()) <= 2) {
                    bestNode = cp;
                    break;
                }
                expanded++;
                double cg = g.getOrDefault(ck, Double.MAX_VALUE);
                for (int i = 0; i < 8; i++) {
                    int nx = cp.getX() + DX[i];
                    int nz = cp.getZ() + DZ[i];
                    if (Math.abs(nx - start.getX()) > maxRange || Math.abs(nz - start.getZ()) > maxRange) continue;
                    // Diagonal corner check
                    if (i >= 4) {
                        if (!passable(new BlockPos(cp.getX() + DX[i], cp.getY() + 1, cp.getZ())) ||
                            !passable(new BlockPos(cp.getX(), cp.getY() + 1, cp.getZ() + DZ[i]))) continue;
                    }
                    BlockPos np = new BlockPos(nx, cp.getY(), nz);
                    boolean ok = standable(np);
                    if (!ok && standable(np.below())) { np = np.below(); ok = true; }
                    if (!ok && standable(np.above())) { np = np.above(); ok = true; }
                    if (!ok && standable(np.below(2))) { np = np.below(2); ok = true; }
                    if (!ok) continue;

                    long nk = key(np.getX(), np.getZ());
                    double step = (i < 4) ? 1.0 : 1.414;
                    double ng = cg + step;
                    if (ng < g.getOrDefault(nk, Double.MAX_VALUE)) {
                        g.put(nk, ng);
                        came.put(nk, cp);
                        node.put(nk, np);
                        open.add(new double[]{ng + heur(np, goal), nk});
                    }
                }
            }
            if (bestNode.equals(start)) return null;
            List<int[]> path = new ArrayList<>();
            BlockPos cur = bestNode;
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
            return Math.max(dx, dz) + 0.414 * Math.min(dx, dz) + dy * 0.6;
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

            if (sq >= 0) {
                List<Map<String, Object>> list = candBySeq.get(sq);
                if (list != null) candidates = list;
            }
            if (idx >= candidates.size()) { pushEvent("bad_action", 1, null); return; }

            Map<String, Object> c = candidates.get(idx);
            String nextAct = String.valueOf(c.get("a"));
            // Allow interrupting goto/mine when brain chooses urgent or different action
            if (activeTarget != null && "mine".equals(activeAction) && "mine".equals(nextAct)) {
                BlockPos p = new BlockPos(ip(c, "x"), ip(c, "y"), ip(c, "z"));
                if (p.equals(miningPos)) return; // keep mining same block
            }
            if (activeTarget != null && "goto".equals(activeAction) && "goto".equals(nextAct)) {
                BlockPos p = new BlockPos(ip(c, "x"), ip(c, "y"), ip(c, "z"));
                if (p.equals(activeTarget)) return; // keep walking to same target
            }
            if (!"mine".equals(nextAct)) miningPos = null;

            actionCount++;
            try { execute(c, nextAct); }
            catch (Throwable e) {
                errorCount++;
                lastError.set(String.valueOf(e));
                pushEvent("act_fail", 1, null);
            }
        }

        void execute(Map<String, Object> c, String a) {
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
                    Vec3 eye = player.getEyePosition();
                    double dx = p.getX() + 0.5 - eye.x, dy = p.getY() + 0.5 - eye.y, dz = p.getZ() + 0.5 - eye.z;
                    player.setYRot((float) Math.toDegrees(Math.atan2(-dx, dz)));
                    player.setYHeadRot(player.getYRot());
                    player.setXRot((float) Math.toDegrees(-Math.atan2(dy, Math.hypot(dx, dz))));
                    miningPos = p;
                    miningTicks = 0;
                    miningProgress = 0.0f;
                    activeAction = "mine";
                    activeTarget = p;
                    activeTicks = 0;
                }
                case "attack" -> {
                    int eid = ip(c, "eid");
                    Entity e = level.getEntity(eid);
                    if (e == null || !e.isAlive() || e.distanceTo(player) > REACH + 1.5) {
                        pushEvent("act_fail", 1, null);
                        return;
                    }
                    selectBestWeapon();
                    faceTowards(e.getX() - player.getX(), e.getZ() - player.getZ());
                    try { player.swing(InteractionHand.MAIN_HAND); } catch (Throwable ignored) {}
                    player.attack(e);
                    player.resetAttackStrengthTicker();
                    try { player.getFoodData().addExhaustion(0.1f); } catch (Throwable ignored) {}
                    lastAttackedId = e.getId();
                    lastAttackedName = BuiltInRegistries.ENTITY_TYPE.getKey(e.getType()).toString();
                    pushEvent("attack", 1, e.blockPosition(), lastAttackedName);
                    finish(true, "attacked");
                }
                case "use" -> {
                    BlockPos p = new BlockPos(ip(c, "x"), ip(c, "y"), ip(c, "z"));
                    if (!p.closerToCenterThan(player.position(), REACH + 1.5)) { pushEvent("act_fail", 1, null); return; }
                    BlockHitResult hit = new BlockHitResult(
                        new Vec3(p.getX() + 0.5, p.getY() + 0.5, p.getZ() + 0.5),
                        Direction.UP, p, false);
                    player.gameMode.useItemOn(player, level, player.getMainHandItem(), InteractionHand.MAIN_HAND, hit);
                    pushEvent("use", 1, p, String.valueOf(c.getOrDefault("n", "")));
                    finish(true, "used");
                }
                case "place" -> {
                    BlockPos p = new BlockPos(ip(c, "x"), ip(c, "y"), ip(c, "z"));
                    int slot = selectBlockSlot();
                    if (slot < 0) { pushEvent("act_fail", 1, null); return; }
                    player.getInventory().selected = slot;
                    BlockHitResult hit = new BlockHitResult(
                        new Vec3(p.getX() + 0.5, p.getY(), p.getZ() + 0.5),
                        Direction.UP, p.below(), false);
                    player.gameMode.useItemOn(player, level, player.getMainHandItem(), InteractionHand.MAIN_HAND, hit);
                    pushEvent("place", 1, p, BuiltInRegistries.ITEM.getKey(player.getMainHandItem().getItem()).toString());
                    finish(true, "placed");
                }
                case "craft" -> {
                    String rid = String.valueOf(c.getOrDefault("rid", ""));
                    RecipeHolder<CraftingRecipe> rh = craftableById.get(rid);
                    if (rh == null || !executeCraft(rh.value())) {
                        pushEvent("act_fail", 1, null);
                        return;
                    }
                    ItemStack res = rh.value().getResultItem(level.registryAccess());
                    String nm = BuiltInRegistries.ITEM.getKey(res.getItem()).toString();
                    pushEvent("craft", res.getCount(), player.blockPosition(), nm);
                    finish(true, "crafted");
                }
                case "equip" -> {
                    int slot = ip(c, "slot");
                    if (slot >= 0 && slot < 9) {
                        player.getInventory().selected = slot;
                        pushEvent("equip", 1, null);
                        finish(true, "equipped");
                    } else {
                        pushEvent("act_fail", 1, null);
                    }
                }
                case "eat" -> {
                    int fslot = findFoodSlot();
                    if (fslot < 0) { pushEvent("act_fail", 1, null); return; }
                    Inventory inv = player.getInventory();
                    if (fslot < 9) {
                        inv.selected = fslot;
                    } else {
                        ItemStack tmp = inv.getItem(inv.selected);
                        inv.setItem(inv.selected, inv.getItem(fslot));
                        inv.setItem(fslot, tmp);
                    }
                    ItemStack held = inv.getItem(inv.selected);
                    var foodProps = held.get(DataComponents.FOOD);
                    if (foodProps != null) {
                        player.getFoodData().eat(foodProps);
                        held.shrink(1);
                        pushEvent("eat", 1, null, BuiltInRegistries.ITEM.getKey(held.getItem()).toString());
                        finish(true, "ate");
                    } else {
                        pushEvent("act_fail", 1, null);
                    }
                }
                case "interact_entity" -> {
                    int eid = ip(c, "eid");
                    Entity e = level.getEntity(eid);
                    if (e == null || !e.isAlive() || e.distanceTo(player) > REACH + 1.5) {
                        pushEvent("act_fail", 1, null);
                        return;
                    }
                    faceTowards(e.getX() - player.getX(), e.getZ() - player.getZ());
                    player.interactOn(e, InteractionHand.MAIN_HAND);
                    pushEvent("interact", 1, e.blockPosition());
                    finish(true, "interacted");
                }
                case "jump" -> {
                    if (player.onGround()) {
                        player.setDeltaMovement(player.getDeltaMovement().x, 0.42, player.getDeltaMovement().z);
                        player.move(MoverType.SELF, player.getDeltaMovement());
                        pushEvent("jump", 0, null);
                    }
                    finish(true, "jumped");
                }
                case "sneak" -> {
                    player.setShiftKeyDown(!player.isCrouching());
                    finish(true, "sneaked");
                }
                case "drop" -> {
                    ItemStack held = player.getMainHandItem();
                    if (!held.isEmpty()) {
                        player.drop(held.split(1), false);
                    }
                    finish(true, "dropped");
                }
                case "wait" -> finish(true, "waited");
                case "stop" -> { route.clear(); miningPos = null; finish(true, "stopped"); }
                default -> pushEvent("act_fail", 1, null);
            }
        }

        void finish(boolean ok, String reason) {
            activeAction = "none";
            activeTarget = null;
            activeTicks = 0;
            activeOk = ok;
        }

        void pushEvent(String name, int mag, BlockPos p) { pushEvent(name, mag, p, null); }

        void pushEvent(String name, int mag, BlockPos p, String itemOrBlock) {
            synchronized (events) {
                if (events.size() >= EV_MAX) return;
                Map<String, Object> m = new HashMap<>();
                m.put("e", name);
                m.put("m", mag);
                if (p != null) { m.put("x", p.getX()); m.put("y", p.getY()); m.put("z", p.getZ()); }
                if (itemOrBlock != null && !itemOrBlock.isEmpty()) m.put("n", itemOrBlock);
                events.add(m);
            }
        }

        // ------------------------------------------------------- crafting --
        boolean canCraftRecipe(CraftingRecipe recipe, Map<Item, Integer> invCounts) {
            List<Ingredient> ings = recipe.getIngredients();
            if (ings.isEmpty()) return false;
            Map<Item, Integer> needed = new HashMap<>();
            for (Ingredient ing : ings) {
                if (ing.isEmpty()) continue;
                boolean matched = false;
                for (Map.Entry<Item, Integer> e : invCounts.entrySet()) {
                    int used = needed.getOrDefault(e.getKey(), 0);
                    if (e.getValue() - used > 0 && ing.test(new ItemStack(e.getKey()))) {
                        needed.put(e.getKey(), used + 1);
                        matched = true;
                        break;
                    }
                }
                if (!matched) return false;
            }
            return true;
        }

        boolean executeCraft(CraftingRecipe recipe) {
            Inventory inv = player.getInventory();
            List<Integer> slotsToConsume = new ArrayList<>();
            Map<Integer, Integer> usedPerSlot = new HashMap<>();
            for (Ingredient ing : recipe.getIngredients()) {
                if (ing.isEmpty()) continue;
                boolean matched = false;
                for (int i = 0; i < 36; i++) {
                    ItemStack st = inv.getItem(i);
                    if (st.isEmpty()) continue;
                    int already = usedPerSlot.getOrDefault(i, 0);
                    if (st.getCount() - already > 0 && ing.test(st)) {
                        usedPerSlot.put(i, already + 1);
                        slotsToConsume.add(i);
                        matched = true;
                        break;
                    }
                }
                if (!matched) return false;
            }
            for (Map.Entry<Integer, Integer> e : usedPerSlot.entrySet()) {
                inv.getItem(e.getKey()).shrink(e.getValue());
            }
            ItemStack outStack = recipe.getResultItem(level.registryAccess()).copy();
            if (!inv.add(outStack)) {
                player.drop(outStack, false);
            }
            return true;
        }

        // ------------------------------------------------------- observation --
        BlockState bs(BlockPos p) {
            int cx = p.getX() >> 4, cz = p.getZ() >> 4;
            int dx = cx - cacheCX + 1, dz = cz - cacheCZ + 1;
            if (dx < 0 || dx > 2 || dz < 0 || dz > 2) {
                cacheCX = cx; cacheCZ = cz;
                for (int ix = 0; ix < 3; ix++) {
                    for (int iz = 0; iz < 3; iz++) {
                        try { chunkCache[ix * 3 + iz] = level.getChunkSource().getChunk(cx + ix - 1, cz + iz - 1, false); }
                        catch (Throwable e) { chunkCache[ix * 3 + iz] = null; }
                    }
                }
                dx = 1; dz = 1;
            }
            net.minecraft.world.level.chunk.LevelChunk ch = chunkCache[dx * 3 + dz];
            if (ch != null) {
                try { return ch.getBlockState(p); } catch (Throwable ignored) {}
            }
            gridFallbacks++;
            try { return level.getBlockState(p); } catch (Throwable e) { return null; }
        }

        static boolean isHostile(Entity e) {
            if (e instanceof Enemy) return true;
            if (e instanceof NeutralMob nm && nm.isAngry()) return true;
            return e.getType().getCategory() == MobCategory.MONSTER;
        }

        boolean isInteractable(BlockState st, BlockPos p) {
            if (st == null || st.isAir()) return false;
            if (st.getMenuProvider(level, p) != null) return true;
            if (level.getBlockEntity(p) != null) return true;
            var b = st.getBlock();
            return b instanceof CraftingTableBlock || b instanceof FurnaceBlock
                || b instanceof DoorBlock || b instanceof TrapDoorBlock
                || b instanceof FenceGateBlock || b instanceof ButtonBlock
                || b instanceof LeverBlock || b instanceof BedBlock
                || st.is(BlockTags.DOORS) || st.is(BlockTags.BEDS) || st.is(BlockTags.BUTTONS);
        }

        void sendObservation() {
            if (player == null || out == null) return;
            Vec3 pos = player.position();
            int bx = (int) Math.floor(pos.x), by = (int) Math.floor(pos.y), bz = (int) Math.floor(pos.z);

            // Inventory summary & scan
            Inventory inv = player.getInventory();
            List<List<Object>> invOut = new ArrayList<>();
            Map<Item, Integer> invCounts = new HashMap<>();
            boolean hasFood = false, hasTool = false, hasWeapon = false, hasBlocks = false;
            for (int i = 0; i < inv.getContainerSize(); i++) {
                ItemStack st = inv.getItem(i);
                if (st.isEmpty()) continue;
                Item it = st.getItem();
                invCounts.merge(it, st.getCount(), Integer::sum);
                int isFood = st.get(DataComponents.FOOD) != null ? 1 : 0;
                int isTool = (it instanceof TieredItem || it instanceof DiggerItem || it instanceof SwordItem) ? 1 : 0;
                int isBlock = (it instanceof BlockItem) ? 1 : 0;
                int isArmor = (it instanceof ArmorItem || it instanceof Equipable) ? 1 : 0;
                double dur = st.isDamageableItem() ? (1.0 - (double) st.getDamageValue() / Math.max(1, st.getMaxDamage())) : 1.0;
                if (isFood == 1) hasFood = true;
                if (it instanceof DiggerItem) hasTool = true;
                if (it instanceof SwordItem) hasWeapon = true;
                if (isBlock == 1) hasBlocks = true;

                List<Object> row = new ArrayList<>(8);
                row.add(i);
                row.add(BuiltInRegistries.ITEM.getId(it));
                row.add(st.getCount());
                row.add(isFood);
                row.add(isTool);
                row.add(isBlock);
                row.add(isArmor);
                row.add(round(dur));
                invOut.add(row);
            }

            // Dense 8x8x8 grid (step=1, gap-free around feet)
            boolean nearCraft = false, nearFurnace = false;
            int k = 0;
            for (int ix = 0; ix < GRID_N; ix++) {
                for (int iy = 0; iy < GRID_N; iy++) {
                    for (int iz = 0; iz < GRID_N; iz++) {
                        BlockPos p = new BlockPos(bx - 3 + ix, by - 2 + iy, bz - 3 + iz);
                        BlockState st = bs(p);
                        int id = 0;
                        int fl = 0;
                        if (st != null && !st.isAir()) {
                            id = BuiltInRegistries.BLOCK.getId(st.getBlock());
                            if (!st.getCollisionShape(level, p).isEmpty()) fl |= 1;
                            if (!st.getFluidState().isEmpty() || st.is(Blocks.FIRE) || st.is(Blocks.CACTUS)) fl |= 2;
                            if (isInteractable(st, p)) fl |= 4;
                            if (st.getBlock() instanceof CraftingTableBlock) nearCraft = true;
                            if (st.getBlock() instanceof FurnaceBlock) nearFurnace = true;
                        }
                        gridIds[k] = id;
                        gridFlags[k] = fl;
                        k++;
                    }
                }
            }

            long dayTime = level.getDayTime() % 24000;
            int curFood = player.getFoodData().getFoodLevel();
            Map<String, Object> self = new HashMap<>();
            Vec3 v = player.getDeltaMovement();
            self.put("x", pos.x); self.put("y", pos.y); self.put("z", pos.z);
            self.put("vx", v.x); self.put("vy", v.y); self.put("vz", v.z);
            self.put("yaw", player.getYRot()); self.put("pitch", player.getXRot());
            self.put("hp", player.getHealth());
            self.put("hp_d", player.getHealth() - lastHp);
            self.put("food", curFood);
            self.put("food_d", curFood - lastFood);
            lastFood = curFood;
            self.put("air", player.getAirSupply());
            self.put("sat", player.getFoodData().getSaturationLevel());
            self.put("light", level.getMaxLocalRawBrightness(new BlockPos(bx, by, bz)));
            self.put("dim", level.dimension().location().toString());
            self.put("on_ground", player.onGround());
            self.put("in_water", inWater());
            self.put("in_lava", player.isInLava());
            self.put("on_fire", player.isOnFire());
            self.put("sprinting", player.isSprinting());
            self.put("sneaking", player.isCrouching());
            self.put("fall", fallStartY >= 0 ? Math.max(0.0, fallStartY - pos.y) : 0.0);
            self.put("collided", player.horizontalCollision);
            self.put("time", dayTime);
            self.put("is_night", dayTime >= 13000 && dayTime <= 23000);
            self.put("raining", level.isRaining());
            self.put("thunder", level.isThundering());
            self.put("armor", player.getArmorValue());
            self.put("atk_cd", player.getAttackStrengthScale(0.5f));
            self.put("near_craft", nearCraft);
            self.put("near_furnace", nearFurnace);
            self.put("sel_slot", inv.selected);
            self.put("has_food", hasFood);
            self.put("has_tool", hasTool);
            self.put("has_weapon", hasWeapon);
            self.put("has_blocks", hasBlocks);

            // Entities (v2 12-field schema)
            AABB area = new AABB(bx - 24, by - 16, bz - 24, bx + 24, by + 16, bz + 24);
            List<Entity> ents = new ArrayList<>(level.getEntitiesOfClass(Entity.class, area,
                e -> e != player && e.isAlive()));
            ents.sort(Comparator.comparingDouble(e -> e.distanceToSqr(player)));
            List<List<Object>> entOut = new ArrayList<>();
            for (Entity e : ents) {
                if (entOut.size() >= MAX_ENT) break;
                int role = 0;
                if (e instanceof Player) role = 2;
                else if (e instanceof ItemEntity) role = 3;
                else if (e instanceof Projectile) role = 4;
                else if (isHostile(e)) role = 1;

                int tgtMe = (e instanceof Mob m && m.getTarget() == player) ? 1 : 0;
                double hp = (e instanceof LivingEntity le) ? le.getHealth() : -1.0;
                double maxHp = (e instanceof LivingEntity le) ? le.getMaxHealth() : 1.0;
                Vec3 ev = e.getDeltaMovement();

                List<Object> row = new ArrayList<>(12);
                row.add(e.getId());
                row.add(BuiltInRegistries.ENTITY_TYPE.getKey(e.getType()).toString());
                row.add(round(e.getX() - pos.x));
                row.add(round(e.getY() - pos.y));
                row.add(round(e.getZ() - pos.z));
                row.add(round(e.distanceTo(player)));
                row.add(round(hp));
                row.add(round(maxHp));
                row.add(role);
                row.add(tgtMe);
                row.add(round(ev.x));
                row.add(round(ev.z));
                entOut.add(row);
            }

            // Kill check
            if (lastAttackedId >= 0) {
                Entity ae = level.getEntity(lastAttackedId);
                if (ae == null || !ae.isAlive()) {
                    pushEvent("kill", 1, player.blockPosition(), lastAttackedName);
                    lastAttackedId = -1;
                }
            }

            int nextSeq = ++seq;
            List<Map<String, Object>> cands = buildCandidates(bx, by, bz, ents, invCounts, nearCraft);
            candidates = cands;
            candBySeq.put(nextSeq, cands);
            while (candBySeq.size() > 16) {
                var it = candBySeq.keySet().iterator();
                it.next();
                it.remove();
            }

            Map<String, Object> frame = new HashMap<>();
            frame.put("t", "obs");
            frame.put("seq", nextSeq);
            frame.put("tick", tickCounter);
            frame.put("self", self);

            Map<String, Object> grid = new HashMap<>();
            grid.put("ox", bx - 3); grid.put("oy", by - 2); grid.put("oz", bz - 3);
            grid.put("step", GRID_STEP); grid.put("n", GRID_N);
            grid.put("ids", gridIds);
            grid.put("flags", gridFlags);
            frame.put("grid", grid);

            frame.put("ent", entOut);
            frame.put("inv", invOut);
            frame.put("cand", cands);

            List<Map<String, Object>> ev;
            synchronized (events) { ev = new ArrayList<>(events); events.clear(); }
            frame.put("ev", ev);
            frame.put("prev", Map.of("a", activeAction, "ticks", activeTicks, "ok", activeOk));

            try {
                PrintWriter w = out;
                if (w != null) {
                    w.println(GSON.toJson(frame));
                    if (w.checkError()) lastError.set("write error");
                }
            } catch (Throwable e) { errorCount++; }
        }

        // -------------------------------------------------------- candidates --
        List<Map<String, Object>> buildCandidates(
                int bx, int by, int bz,
                List<Entity> ents, Map<Item, Integer> invCounts, boolean nearCraft) {

            List<Map<String, Object>> out = new ArrayList<>();

            // 1. Combat & Entity interaction
            for (Entity e : ents) {
                double dist = e.distanceTo(player);
                if (dist <= REACH) {
                    boolean host = isHostile(e);
                    if (host || e instanceof net.minecraft.world.entity.animal.Animal) {
                        Map<String, Object> m = new HashMap<>();
                        m.put("a", "attack"); m.put("eid", e.getId());
                        m.put("n", BuiltInRegistries.ENTITY_TYPE.getKey(e.getType()).toString());
                        m.put("d", round(dist));
                        m.put("h", host ? 1 : 0);
                        m.put("tgt", (e instanceof Mob mb && mb.getTarget() == player) ? 1 : 0);
                        out.add(m);
                    }
                    if (e instanceof Player || e instanceof net.minecraft.world.entity.npc.AbstractVillager) {
                        Map<String, Object> m = new HashMap<>();
                        m.put("a", "interact_entity"); m.put("eid", e.getId());
                        m.put("n", e.getName().getString());
                        m.put("d", round(dist));
                        m.put("player", (e instanceof Player) ? 1 : 0);
                        out.add(m);
                    }
                } else if (dist < 24.0 && (e instanceof ItemEntity || isHostile(e) || e instanceof Player)) {
                    Map<String, Object> m = new HashMap<>();
                    m.put("a", "goto");
                    m.put("x", (int) Math.floor(e.getX()));
                    m.put("y", (int) Math.floor(e.getY()));
                    m.put("z", (int) Math.floor(e.getZ()));
                    m.put("d", round(dist));
                    m.put("h", isHostile(e) ? 1 : 0);
                    m.put("item_drop", (e instanceof ItemEntity) ? 1 : 0);
                    m.put("player", (e instanceof Player) ? 1 : 0);
                    out.add(m);
                }
                if (out.size() >= 10) break;
            }

            // 2. Eat if hungry
            if (player.getFoodData().getFoodLevel() < 18 && findFoodSlot() >= 0) {
                Map<String, Object> m = new HashMap<>();
                m.put("a", "eat"); m.put("d", 0.0); m.put("val", 1.0);
                out.add(m);
            }

            // 3. Craftable recipes (vanilla + all modded recipes, sorted by progression value)
            craftableById.clear();
            try {
                int dim = nearCraft ? 3 : 2;
                List<Map<String, Object>> craftCands = new ArrayList<>();
                for (RecipeHolder<CraftingRecipe> rh : level.getRecipeManager().getAllRecipesFor(RecipeType.CRAFTING)) {
                    CraftingRecipe rec = rh.value();
                    if (!rec.canCraftInDimensions(dim, dim)) continue;
                    if (!canCraftRecipe(rec, invCounts)) continue;
                    ItemStack res = rec.getResultItem(level.registryAccess());
                    if (res.isEmpty()) continue;
                    String rid = rh.id().toString();
                    String rname = BuiltInRegistries.ITEM.getKey(res.getItem()).toString();
                    craftableById.put(rid, rh);
                    double val = 0.2;
                    if (res.getItem() instanceof TieredItem || res.getItem() instanceof ArmorItem) val = 1.0;
                    else if (rname.contains("crafting_table") || rname.contains("furnace") || rname.contains("chest")) val = 0.95;
                    else if (rname.contains("planks") || rname.contains("stick") || rname.contains("torch") || rname.contains("bed")) val = 0.8;
                    else if (res.get(DataComponents.FOOD) != null) val = 0.7;
                    Map<String, Object> m = new HashMap<>();
                    m.put("a", "craft"); m.put("rid", rid); m.put("n", rname);
                    m.put("d", 0.0); m.put("val", val);
                    craftCands.add(m);
                }
                craftCands.sort((a, b) -> Double.compare((double) b.get("val"), (double) a.get("val")));
                out.addAll(craftCands.subList(0, Math.min(6, craftCands.size())));
            } catch (Throwable ignored) {}

            // 4. Mine & Use nearby blocks (prioritize ores/logs/stone over plain dirt)
            List<Map<String, Object>> mine = new ArrayList<>();
            List<Map<String, Object>> useList = new ArrayList<>();
            for (int dx = -4; dx <= 4; dx++) {
                for (int dy = -2; dy <= 4; dy++) {
                    for (int dz = -4; dz <= 4; dz++) {
                        BlockPos p = new BlockPos(bx + dx, by + dy, bz + dz);
                        BlockState st = bs(p);
                        if (st == null || st.isAir() || !st.getFluidState().isEmpty()) continue;
                        if (st.getDestroySpeed(level, p) < 0) continue;
                        double d = Math.sqrt(dx * dx + dy * dy + dz * dz);
                        if (d > REACH) continue;
                        String bname = BuiltInRegistries.BLOCK.getKey(st.getBlock()).toString();
                        double bval = 0.0;
                        if (bname.contains("ore") || bname.contains("diamond") || bname.contains("ancient")) bval = 1.0;
                        else if (bname.contains("log") || bname.contains("wood") || bname.contains("coal")) bval = 0.8;
                        else if (bname.contains("stone") || bname.contains("cobble") || bname.contains("iron")) bval = 0.4;
                        Map<String, Object> m = new HashMap<>();
                        m.put("a", "mine"); m.put("x", p.getX()); m.put("y", p.getY()); m.put("z", p.getZ());
                        m.put("n", bname); m.put("d", round(d)); m.put("val", bval);
                        mine.add(m);
                        if (isInteractable(st, p) && useList.size() < 3) {
                            Map<String, Object> u = new HashMap<>(m);
                            u.put("a", "use");
                            useList.add(u);
                        }
                    }
                }
            }
            mine.sort(Comparator.comparingDouble(m -> (double) m.get("d") - 2.0 * (double) m.get("val")));
            out.addAll(useList);
            out.addAll(mine.subList(0, Math.min(8, mine.size())));

            // 5. Place block
            if (selectBlockSlot() >= 0) {
                for (int i = 0; i < 8; i++) {
                    double ang = i * Math.PI / 4;
                    int px = (int) Math.floor(player.getX() + Math.sin(ang) * 2);
                    int pz = (int) Math.floor(player.getZ() + Math.cos(ang) * 2);
                    BlockPos tp = new BlockPos(px, by, pz);
                    BlockState tgt = bs(tp);
                    if (tgt == null || !tgt.isAir()) continue;
                    BlockState ref = bs(tp.below());
                    if (ref == null || !ref.isFaceSturdy(level, tp.below(), Direction.UP)) continue;
                    Map<String, Object> m = new HashMap<>();
                    m.put("a", "place"); m.put("x", px); m.put("y", by); m.put("z", pz);
                    m.put("n", BuiltInRegistries.ITEM.getKey(player.getInventory().getItem(selectBlockSlot()).getItem()).toString());
                    m.put("d", 2.0);
                    out.add(m);
                    break;
                }
            }

            // 6. Movement directions (8 angles x 2 distances)
            for (int i = 0; i < 8; i++) {
                double ang = i * Math.PI / 4;
                double sx = Math.sin(ang), sz = Math.cos(ang);
                for (double dist : new double[]{3.0, 7.0}) {
                    int gx = (int) Math.floor(player.getX() + sx * dist);
                    int gz = (int) Math.floor(player.getZ() + sz * dist);
                    int gy = standY(gx, by, gz);
                    if (gy == Integer.MIN_VALUE) continue;
                    Map<String, Object> m = new HashMap<>();
                    m.put("a", "goto"); m.put("x", gx); m.put("y", gy); m.put("z", gz); m.put("d", dist);
                    out.add(m);
                }
            }

            // 7. Utility actions
            out.add(Map.of("a", "jump", "d", 0.0));
            out.add(Map.of("a", "wait", "d", 0.0));
            out.add(Map.of("a", "stop", "d", 0.0));

            if (out.size() > MAX_CAND) out = out.subList(0, MAX_CAND);
            return out;
        }

        int findFoodSlot() {
            Inventory inv = player.getInventory();
            for (int i = 0; i < inv.getContainerSize(); i++) {
                ItemStack st = inv.getItem(i);
                if (!st.isEmpty() && st.get(DataComponents.FOOD) != null) return i;
            }
            return -1;
        }

        int selectBlockSlot() {
            Inventory inv = player.getInventory();
            if (!inv.getItem(inv.selected).isEmpty() && inv.getItem(inv.selected).getItem() instanceof BlockItem) {
                return inv.selected;
            }
            for (int i = 0; i < 9; i++) {
                ItemStack st = inv.getItem(i);
                if (!st.isEmpty() && st.getItem() instanceof BlockItem) return i;
            }
            return -1;
        }

        int standY(int x, int nearY, int z) {
            for (int dy = 3; dy >= -4; dy--) {
                int y = nearY + dy;
                BlockPos p = new BlockPos(x, y, z);
                if (standable(p)) return y;
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

        static double round(double v) { return Math.round(v * 10000.0) / 10000.0; }
    }
}
