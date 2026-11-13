package ru.vovawe.soma;

import com.google.gson.Gson;
import com.mojang.authlib.GameProfile;
import net.minecraft.core.BlockPos;
import net.minecraft.core.registries.BuiltInRegistries;
import net.minecraft.nbt.CompoundTag;
import net.minecraft.network.chat.Component;
import net.minecraft.resources.ResourceLocation;
import net.minecraft.server.MinecraftServer;
import net.minecraft.server.level.ServerLevel;
import net.minecraft.server.level.ServerPlayer;
import net.minecraft.world.InteractionHand;
import net.minecraft.world.entity.Entity;
import net.minecraft.world.entity.item.ItemEntity;
import net.minecraft.world.entity.player.Inventory;
import net.minecraft.world.item.ItemStack;
import net.minecraft.world.level.Level;
import net.minecraft.world.level.block.state.BlockState;
import net.minecraft.world.phys.AABB;
import net.minecraft.world.phys.BlockHitResult;
import net.minecraft.world.phys.Vec3;
import net.neoforged.bus.api.IEventBus;
import net.neoforged.bus.api.SubscribeEvent;
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
import java.util.Queue;
import java.util.UUID;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;

/**
 * SOMA environment adapter, server side.
 *
 * Runs inside the NeoForge server: owns one fake player, turns the world into a
 * numeric frame, advertises admissible candidates, executes exactly the
 * candidate the brain picked, and streams events back.
 *
 * No policy, no reward shaping, no strategy. Same wire schema as bot/ (Node).
 */
@Mod("soma")
public class SomaMod {

    public static final String MODID = "soma";
    private static final Gson GSON = new Gson();

    public SomaMod(IEventBus modBus) {
        NeoForge.EVENT_BUS.addListener(this::onServerStarting);
        NeoForge.EVENT_BUS.addListener(this::onServerTick);
        NeoForge.EVENT_BUS.addListener(this::onServerStopping);
        NeoForge.EVENT_BUS.addListener(this::onRegisterCommands);
        System.out.println("[soma] mod constructed");
    }

    private Agent agent;

    private void onServerStarting(ServerStartedEvent ev) {
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
        // /soma start|stop|status|tp x y z
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
        );
    }

    // ================================================================== ===
    //  Agent: fake player + observation + execution + IPC
    // ================================================================== ===
    static final class Agent {
        static final int GRID_N = 8;
        static final int GRID_STEP = 2;
        static final int GRID_CELLS = GRID_N * GRID_N * GRID_N;
        static final int MAX_CAND = 48;
        static final int MAX_ENT = 12;
        static final int EV_MAX = 10;
        static final int OBS_TICKS = 6;          // 300 ms at 20 TPS

        static final double REACH = 4.5;
        static final String HOST = "127.0.0.1";
        static final int PORT = Integer.parseInt(System.getenv().getOrDefault("SOMA_IPC_PORT", "8788"));

        final MinecraftServer server;
        ServerLevel level;
        FakePlayer player;

        final AtomicBoolean enabled = new AtomicBoolean(false);
        final AtomicBoolean running = new AtomicBoolean(false);
        final AtomicReference<Thread> netThread = new AtomicReference<>();
        final AtomicReference<String> lastError = new AtomicReference<>("not started");

        ServerSocket serverSocket;
        volatile PrintWriter out;
        volatile BufferedReader in;
        final Queue<int[]> incoming = new ArrayDeque<>();

        // chunk cache: getBlockState() on ServerLevel resolves the chunk (and joins a
        // future) on every call - at 512 samples that was 33 ms.  Reading the chunk
        // directly is ~30x cheaper.
        net.minecraft.world.level.chunk.LevelChunk curChunk = null;
        int curCX = Integer.MIN_VALUE, curCZ = Integer.MIN_VALUE;
        int gridFallbacks = 0;
        final int[] gridCache = new int[GRID_CELLS];

        BlockState bs(BlockPos p) {
            int cx = p.getX() >> 4, cz = p.getZ() >> 4;
            if (cx != curCX || cz != curCZ) {
                curCX = cx; curCZ = cz;
                try {
                    curChunk = level.getChunkSource().getChunk(cx, cz, false);
                } catch (Throwable e) { curChunk = null; }
            }
            if (curChunk != null) {
                try { return curChunk.getBlockState(p); } catch (Throwable e) { /* fallthrough */ }
            }
            gridFallbacks++;
            try { return level.getBlockState(p); } catch (Throwable e) { return null; }
        }

        /** The fake player is not in the entity tick list, so it never picks items up. */
        void pickupNear(BlockPos center) {
            if (player == null) return;
            try {
                AABB a = new AABB(center).inflate(4.0);
                for (ItemEntity ie : level.getEntitiesOfClass(ItemEntity.class, a, e -> e.isAlive())) {
                    ItemStack st = ie.getItem().copy();
                    if (st.isEmpty()) continue;
                    if (player.getInventory().add(st)) ie.discard();
                }
            } catch (Throwable ignored) {}
        }

        int tickCounter = 0;
        int warmup = 20;
        int seq = 0;
        long frameNanos = 0;
        int actionCount = 0;
        int errorCount = 0;

        // active action
        String activeAction = "none";
        BlockPos activeTarget = null;
        int activeEntity = -1;
        int activeTicks = 0;
        BlockPos lastPos = null;
        boolean activeOk = true;

        // events since last frame
        final List<Map<String, Object>> events = new ArrayList<>();

        // block id -> name, sent to the brain once in a while
        final Map<Integer, String> blockNames = new HashMap<>();

        Agent(MinecraftServer server) {
            this.server = server;
        }

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
                    + " pos=" + (player == null ? "-" : String.format(Locale.ROOT, "%.0f,%.0f,%.0f", player.getX(), player.getY(), player.getZ()))
                    + " inv=" + (player == null ? 0 : player.getInventory().getItem(0).getCount())
                    + " errors=" + errorCount + " lastError=" + lastError.get()), true);
        }

        // ------------------------------------------------------------- net --
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
                        String line;
                        while (running.get() && (line = in.readLine()) != null) {
                            line = line.trim();
                            if (line.isEmpty()) continue;
                            try {
                                Map<?, ?> msg = GSON.fromJson(line, Map.class);
                                Object t = msg.get("t");
                                Object i = msg.get("i");
                                if ("act".equals(t) && i instanceof Number) {
                                    synchronized (incoming) {
                                        incoming.clear();
                                        incoming.add(new int[]{((Number) i).intValue()});
                                    }
                                }
                            } catch (Exception e) {
                                errorCount++;
                            }
                        }
                        System.out.println("[soma] brain disconnected");
                    }
                } catch (Throwable e) {
                    lastError.set(String.valueOf(e));
                    System.out.println("[soma] net error: " + e);
                    e.printStackTrace();
                    try { Thread.sleep(2000); } catch (InterruptedException ignored) {}
                }
            }
        }

        // ------------------------------------------------------------ spawn --
        void ensurePlayer() {
            if (level == null) {
                level = server.getLevel(net.minecraft.world.level.Level.OVERWORLD);
            }
            if (player == null && level != null) {
                GameProfile profile = new GameProfile(UUID.fromString("6dded40e-b7fe-3867-86cf-20f080f1c988"), "SOMA");
                player = new FakePlayer(level, profile) {
                    @Override
                    public boolean isInvulnerableTo(net.minecraft.world.damagesource.DamageSource dmg) { return true; }
                    @Override
                    public void tick() { /* driven by the agent */ }
                };
                player.getAbilities().instabuild = false;
                player.getAbilities().mayfly = true;
                BlockPos spawn = new BlockPos(0, 100, 0);
                try {
                    spawn = level.getSharedSpawnPos();
                } catch (Exception ignored) {}
                player.setPos(spawn.getX() + 0.5, findGroundY(spawn.getX(), spawn.getZ()) + 1.0, spawn.getZ() + 0.5);
                // NOTE: deliberately NOT calling level.addNewPlayer().
                // Once the fake player enters the entity tick list, mods that push
                // packets to players (AppleSkin and friends) block the server
                // thread forever trying to talk to a connection that does not exist.
                System.out.println("[soma] fake player created at " + player.getX() + "," + player.getY() + "," + player.getZ());
            }
            if (player != null && player.isRemoved()) {
                player = null;
            }
        }

        int findGroundY(int x, int z) {
            try {
                for (int y = 320; y > -64; y--) {
                    BlockState st = level.getBlockState(new BlockPos(x, y, z));
                    if (!st.isAir() && !st.getFluidState().isSource() && st.isFaceSturdy(level, new BlockPos(x, y, z), net.minecraft.core.Direction.UP)) {
                        return y + 1;
                    }
                }
            } catch (Exception ignored) {}
            return 100;
        }

        // ------------------------------------------------------------- tick --
        void tick() {
            if (!enabled.get()) return;
            try {
                ensurePlayer();
                if (player == null) return;
                if (warmup > 0) { warmup--; return; }
                tickCounter++;
                if (tickCounter % OBS_TICKS != 0) {
                    // still drive long actions (goto) every tick
                    driveActive();
                    return;
                }
                driveActive();
                sendObservation();
                pollAndExecute();
            } catch (Throwable e) {
                errorCount++;
                lastError.set(String.valueOf(e));
            }
        }

        // ---------------------------------------------------------- driving --
        void driveActive() {
            if (player == null || activeTarget == null) return;
            if (!"goto".equals(activeAction)) return;
            double dx = (activeTarget.getX() + 0.5) - player.getX();
            double dz = (activeTarget.getZ() + 0.5) - player.getZ();
            double d = Math.sqrt(dx * dx + dz * dz);
            if (d < 0.9) { finish(true, "arrived"); return; }
            double step = Math.min(0.18, d);
            double nx = player.getX() + dx / d * step;
            double nz = player.getZ() + dz / d * step;
            double ny = player.getY();
            float yaw = (float) Math.toDegrees(Math.atan2(-dx, dz));
            player.setYRot(yaw);
            player.setYHeadRot(yaw);

            AABB box = player.getBoundingBox().move(nx - player.getX(), 0, nz - player.getZ());
            boolean free = level.noCollision(player, box);
            if (!free) {
                // try step up one block
                AABB up = box.move(0, 1.0, 0);
                if (level.noCollision(player, up)
                    && !level.getBlockState(BlockPos.containing(nx, ny, nz)).isAir()) {
                    player.setPos(nx, ny + 1.0, nz);
                } else {
                    activeTicks++;
                    finish(false, "blocked");
                    return;
                }
            } else {
                player.setPos(nx, ny, nz);
            }
            activeTicks++;
            if (activeTicks > 200) finish(false, "timeout");
        }

        void finish(boolean ok, String reason) {
            activeAction = "none";
            activeTarget = null;
            activeEntity = -1;
            activeTicks = 0;
            activeOk = ok;
            if (player != null && !ok) {
                // no-op; the brain sees the failure through events
            }
        }

        // --------------------------------------------------------- execution --
        void pollAndExecute() {
            int idx = -1;
            synchronized (incoming) {
                if (!incoming.isEmpty()) idx = incoming.poll()[0];
            }
            if (idx < 0) return;
            if (activeTarget != null && "goto".equals(activeAction)) return; // busy walking
            actionCount++;
            try {
                execute(idx);
            } catch (Throwable e) {
                errorCount++;
                lastError.set(String.valueOf(e));
                pushEvent("act_fail", 1, null);
            }
        }

        /** Candidates are rebuilt here and cached by sendObservation(). */
        List<Map<String, Object>> candidates = new ArrayList<>();

        void execute(int idx) {
            if (idx < 0 || idx >= candidates.size()) {
                pushEvent("bad_action", 1, null);
                return;
            }
            Map<String, Object> c = candidates.get(idx);
            String a = String.valueOf(c.get("a"));
            switch (a) {
                case "goto" -> {
                    activeAction = "goto";
                    activeTarget = new BlockPos(ip(c, "x"), ip(c, "y"), ip(c, "z"));
                    activeTicks = 0;
                    lastPos = player.blockPosition();
                }
                case "mine" -> {
                    BlockPos p = new BlockPos(ip(c, "x"), ip(c, "y"), ip(c, "z"));
                    if (!p.closerToCenterThan(player.position(), REACH + 1.5)) { pushEvent("act_fail", 1, null); return; }
                    boolean ok = player.gameMode.destroyBlock(p);
                    if (ok) pickupNear(p);
                    pushEvent(ok ? "dig" : "dig_fail", 1, p);
                    finish(ok, "mined");
                }
                case "attack" -> {
                    int eid = ((Number) c.get("eid")).intValue();
                    Entity e = level.getEntity(eid);
                    if (e == null || !e.isAlive() || e.distanceTo(player) > REACH + 1.5) { pushEvent("act_fail", 1, null); return; }
                    player.attack(e);
                    pickupNear(e.blockPosition());
                    pushEvent("attack", 1, null);
                    finish(true, "attacked");
                }
                case "use" -> {
                    BlockPos p = new BlockPos(ip(c, "x"), ip(c, "y"), ip(c, "z"));
                    BlockHitResult hit = new BlockHitResult(new Vec3(p.getX() + 0.5, p.getY() + 0.5, p.getZ() + 0.5),
                        net.minecraft.core.Direction.UP, p, false);
                    player.gameMode.useItemOn(player, level, player.getMainHandItem(), InteractionHand.MAIN_HAND, hit);
                    pushEvent("use", 1, p);
                    finish(true, "used");
                }
                case "place" -> {
                    BlockPos p = new BlockPos(ip(c, "x"), ip(c, "y"), ip(c, "z"));
                    BlockHitResult hit = new BlockHitResult(new Vec3(p.getX() + 0.5, p.getY(), p.getZ() + 0.5),
                        net.minecraft.core.Direction.UP, p.below(), false);
                    player.gameMode.useItemOn(player, level, player.getMainHandItem(), InteractionHand.MAIN_HAND, hit);
                    pushEvent("place", 1, p);
                    finish(true, "placed");
                }
                case "eat" -> {
                    player.gameMode.useItem(player, level, player.getMainHandItem(), InteractionHand.MAIN_HAND);
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

        void pushEvent(String name, int mag, BlockPos p) {
            if (events.size() >= EV_MAX) return;
            Map<String, Object> m = new HashMap<>();
            m.put("e", name);
            m.put("m", mag);
            if (p != null) {
                m.put("x", p.getX());
                m.put("y", p.getY());
                m.put("z", p.getZ());
            }
            events.add(m);
        }

        // ------------------------------------------------------- observation --
        void sendObservation() {
            if (player == null || out == null) return;
            long t0 = System.nanoTime();
            Vec3 pos = player.position();
            int bx = (int) Math.floor(pos.x);
            int by = (int) Math.floor(pos.y);
            int bz = (int) Math.floor(pos.z);

            long tGrid = 0, tEnt = 0, tInv = 0, tCand = 0, tJson = 0;
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
            self.put("in_water", player.isInWater());
            self.put("in_lava", player.isInLava());
            self.put("sprinting", player.isSprinting());
            self.put("sneaking", player.isCrouching());
            self.put("flying", player.getAbilities().flying);
            self.put("fall", player.fallDistance);
            self.put("collided", player.horizontalCollision);
            self.put("time", level.getDayTime() % 24000);
            self.put("raining", level.isRaining());
            self.put("thunder", level.isThundering());

            // voxel window - rebuilt every other frame, reused in between
            // (512 block reads cost ~18 ms on this pack, half of that is not needed
            //  at a 300 ms decision cadence)
            boolean rebuild = (seq % 2) == 0;
            int[] ids = gridCache;
            int k = 0;
            if (!rebuild) { ids = gridCache; }
            for (int ix = 0; ix < GRID_N; ix++) {
                for (int iy = 0; iy < GRID_N; iy++) {
                    for (int iz = 0; iz < GRID_N; iz++) {
                        if (!rebuild) { k = GRID_CELLS; break; }
                        BlockPos p = new BlockPos(bx - 7 + ix * GRID_STEP, by - 7 + iy * GRID_STEP, bz - 7 + iz * GRID_STEP);
                        BlockState st = bs(p);
                        int id = 0;
                        if (st != null && !st.isAir()) {
                            id = BuiltInRegistries.BLOCK.getId(st.getBlock());
                            if (!blockNames.containsKey(id)) {
                                blockNames.put(id, BuiltInRegistries.BLOCK.getKey(st.getBlock()).toString());
                            }
                        }
                        ids[k++] = id;
                    }
                }
            }

            long tSelf = tGrid = System.nanoTime();
            // entities
            AABB area = new AABB(bx - 24, by - 24, bz - 24, bx + 24, by + 24, bz + 24);
            List<Entity> ents = new ArrayList<>(level.getEntitiesOfClass(Entity.class, area, e -> e != player && e.isAlive()));
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
                row.add(e instanceof net.minecraft.world.entity.monster.Enemy ? 1 : 0);
                entOut.add(row);
            }

            tEnt = System.nanoTime();
            // inventory
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
                if (!blockNames.containsKey(-BuiltInRegistries.ITEM.getId(st.getItem()))) {
                    blockNames.put(-BuiltInRegistries.ITEM.getId(st.getItem()), BuiltInRegistries.ITEM.getKey(st.getItem()).toString());
                }
                invOut.add(row);
            }

            tInv = System.nanoTime();
            List<Map<String, Object>> cands = buildCandidates(bx, by, bz);
            tCand = System.nanoTime();
            candidates = cands;

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
            synchronized (events) {
                ev = new ArrayList<>(events);
                events.clear();
            }
            frame.put("ev", ev);
            frame.put("bmap", blockNames.size() > 512 ? new ArrayList<>(blockNames.entrySet()) : new ArrayList<>());
            frame.put("prev", Map.of("a", activeAction, "ticks", activeTicks, "ok", activeOk));

            tJson = System.nanoTime();
            try {
                String json = GSON.toJson(frame);
                PrintWriter w = out;
                if (w != null) {
                    w.println(json);
                    if (w.checkError()) { lastError.set("write error"); }
                }
            } catch (Throwable e) {
                errorCount++;
                System.out.println("[soma] frame error: " + e);
            }
            frameNanos = System.nanoTime() - t0;
            if (seq % 10 == 0) {
                System.out.println(String.format(Locale.ROOT,
                    "[soma] obs #%d self=%.1fms grid=%.1fms ent=%.1fms inv=%.1fms cand=%.1fms total=%.1fms cands=%d fb=%d",
                    seq, (tSelf - t0) / 1e6, (tGrid - tSelf) / 1e6, (tEnt - tGrid) / 1e6, (tInv - tEnt) / 1e6,
                    (tCand - tInv) / 1e6, (tJson - tCand) / 1e6, frameNanos / 1e6, cands.size(), gridFallbacks));
            }
        }

        // -------------------------------------------------------- candidates --
        List<Map<String, Object>> buildCandidates(int bx, int by, int bz) {
            List<Map<String, Object>> out = new ArrayList<>();

            // mine targets: nearest blocks in reach that drop something
            List<Map<String, Object>> mine = new ArrayList<>();
            for (int dx = -4; dx <= 4 && mine.size() < 12; dx++) {
                for (int dy = -3; dy <= 4 && mine.size() < 12; dy++) {
                    for (int dz = -4; dz <= 4 && mine.size() < 12; dz++) {
                        BlockPos p = new BlockPos(bx + dx, by + dy, bz + dz);
                        BlockState st;
                        try { st = level.getBlockState(p); } catch (Exception e) { continue; }
                        if (st.isAir() || st.getFluidState().isSource()) continue;
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

            // attack targets
            AABB area = new AABB(bx - 6, by - 6, bz - 6, bx + 6, by + 6, bz + 6);
            List<Entity> hostiles = new ArrayList<>(level.getEntitiesOfClass(Entity.class, area, e -> e != player && e.isAlive()));
            hostiles.sort(Comparator.comparingDouble(e -> e.distanceToSqr(player)));
            int added = 0;
            for (Entity e : hostiles) {
                if (added >= 3) break;
                if (e.distanceTo(player) > REACH) continue;
                Map<String, Object> m = new HashMap<>();
                m.put("a", "attack"); m.put("eid", e.getId());
                m.put("n", BuiltInRegistries.ENTITY_TYPE.getKey(e.getType()).toString());
                m.put("d", round(e.distanceTo(player)));
                m.put("h", e instanceof net.minecraft.world.entity.monster.Enemy ? 1 : 0);
                out.add(m);
                added++;
            }

            // interact with the nearest interactive block
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

            // goto: 8 directions x 3 distances
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

            Map<String, Object> wait = new HashMap<>();
            wait.put("a", "wait"); wait.put("d", 0);
            out.add(wait);
            Map<String, Object> stop = new HashMap<>();
            stop.put("a", "stop"); stop.put("d", 0);
            out.add(stop);

            if (out.size() > MAX_CAND) out = out.subList(0, MAX_CAND);
            return out;
        }

        int standY(int x, int nearY, int z) {
            for (int dy = 3; dy >= -4; dy--) {
                int y = nearY + dy;
                BlockPos p = new BlockPos(x, y, z);
                BlockState below = bs(p.below());
                if (below == null || below.isAir() || !below.isFaceSturdy(level, p.below(), net.minecraft.core.Direction.UP)) continue;
                BlockState here = bs(p);
                BlockState above = bs(p.above());
                if (here == null || !here.isAir()) continue;
                if (above == null || !above.isAir()) continue;
                return y;
            }
            return Integer.MIN_VALUE;
        }

        static double round(double v) { return Math.round(v * 100000.0) / 100000.0; }

        /** Gson hands back Integer for whole numbers - never cast blind. */
        static int ip(Map<String, Object> m, String k) {
            Object o = m.get(k);
            return o instanceof Number ? ((Number) o).intValue() : 0;
        }
    }
}