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
import net.neoforged.neoforge.event.ServerChatEvent;
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
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Queue;
import java.util.Set;
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
        NeoForge.EVENT_BUS.addListener(this::onServerChat);
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

    private void onServerChat(net.neoforged.neoforge.event.ServerChatEvent ev) {
        if (agent != null) agent.onPlayerChat(ev.getPlayer(), ev.getRawText());
    }

    private void onRegisterCommands(RegisterCommandsEvent ev) {
        var cmd = net.minecraft.commands.Commands.literal("soma")
            .then(net.minecraft.commands.Commands.literal("start")
                .executes(c -> { if (agent != null) { agent.start(); agent.setEnabled(true); }
                                 c.getSource().sendSuccess(() -> Component.literal("[soma] enabled"), true); return 1; }))
            .then(net.minecraft.commands.Commands.literal("stop")
                .executes(c -> { if (agent != null) agent.setEnabled(false);
                                 c.getSource().sendSuccess(() -> Component.literal("[soma] disabled"), true); return 1; }))
            .then(net.minecraft.commands.Commands.literal("status")
                .executes(c -> { if (agent != null) { agent.start(); agent.status(c.getSource()); } return 1; }))
            .then(net.minecraft.commands.Commands.literal("inv")
                .executes(c -> { if (agent != null) agent.showInventory(c.getSource()); return 1; }))
            .then(net.minecraft.commands.Commands.literal("trust")
                .executes(c -> { if (agent != null) agent.showTrust(c.getSource()); return 1; }))
            .then(net.minecraft.commands.Commands.literal("tp")
                .executes(c -> { if (agent != null) agent.tpAdminToSoma(c.getSource()); return 1; }))
            .then(net.minecraft.commands.Commands.literal("tphere")
                .executes(c -> { if (agent != null) agent.tpSomaToAdmin(c.getSource()); return 1; }))
            .then(net.minecraft.commands.Commands.literal("heal")
                .executes(c -> { if (agent != null) agent.healSoma(c.getSource()); return 1; }))
            .then(net.minecraft.commands.Commands.literal("kit")
                .executes(c -> { if (agent != null) agent.giveStarterKit(c.getSource()); return 1; }))
            .then(net.minecraft.commands.Commands.literal("clearinv")
                .executes(c -> { if (agent != null) agent.clearInventory(c.getSource()); return 1; }))
            .then(net.minecraft.commands.Commands.literal("spawnmob")
                .executes(c -> { if (agent != null) agent.spawnTestMob(); return 1; }))
            .then(net.minecraft.commands.Commands.literal("hurt")
                .executes(c -> { if (agent != null) agent.debugHurt(c.getSource(), 4.0f); return 1; })
                .then(net.minecraft.commands.Commands.argument("amount", com.mojang.brigadier.arguments.FloatArgumentType.floatArg(0.5f, 100.0f))
                    .executes(c -> { if (agent != null) agent.debugHurt(c.getSource(),
                        com.mojang.brigadier.arguments.FloatArgumentType.getFloat(c, "amount")); return 1; })))
            .then(net.minecraft.commands.Commands.literal("god")
                .executes(c -> { if (agent != null) agent.toggleGod(c.getSource()); return 1; }))
            .then(net.minecraft.commands.Commands.literal("say")
                .then(net.minecraft.commands.Commands.argument("msg", com.mojang.brigadier.arguments.StringArgumentType.greedyString())
                    .executes(c -> { if (agent != null) agent.sayChat(
                        com.mojang.brigadier.arguments.StringArgumentType.getString(c, "msg")); return 1; })))
            .then(net.minecraft.commands.Commands.literal("cleanup")
                .executes(c -> { if (agent != null) agent.cleanupStands(true); return 1; }));
        ev.getDispatcher().register(cmd);
    }

    // =====================================================================
    //  SomaPlayer — FakePlayer with real survival damage & cooldowns
    // =====================================================================
    static final class SomaPlayer extends FakePlayer {
        private static final java.lang.reflect.Field SPAWN_INVULN_FIELD;
        static {
            java.lang.reflect.Field f = null;
            try {
                f = net.minecraft.server.level.ServerPlayer.class.getDeclaredField("spawnInvulnerableTime");
                f.setAccessible(true);
            } catch (Throwable ignored) {}
            SPAWN_INVULN_FIELD = f;
        }

        boolean godMode = false;

        SomaPlayer(ServerLevel level, GameProfile profile) {
            super(level, profile);
        }

        void clearSpawnInvuln() {
            if (SPAWN_INVULN_FIELD != null) {
                try { SPAWN_INVULN_FIELD.setInt(this, 0); } catch (Throwable ignored) {}
            }
        }

        public void bumpAttackCooldown() {
            this.attackStrengthTicker++;
        }

        @Override public boolean isInvulnerableTo(net.minecraft.world.damagesource.DamageSource d) { return godMode; }
        @Override public boolean isInvulnerable() { return godMode; }
        @Override public void tick() { /* driven by Agent.tick() */ }
        @Override public boolean isSpectator() { return false; }
        @Override public boolean isInvisible() { return false; }
        @Override public boolean canBeSeenAsEnemy() { return !godMode && isAlive(); }
        @Override public boolean canHarmPlayer(Player other) { return true; }

        @Override
        public boolean hurt(net.minecraft.world.damagesource.DamageSource source, float amount) {
            if (godMode) return false;
            clearSpawnInvuln();
            if (amount <= 0.0f || isDeadOrDying()) return false;
            boolean ok = false;
            try { ok = super.hurt(source, amount); } catch (Throwable ignored) {}
            if (!ok) {
                if (this.invulnerableTime > 10) return false;
                this.actuallyHurt(source, amount);
                this.invulnerableTime = 20;
                this.hurtTime = 10;
                this.hurtDuration = 10;
                this.markHurt();
                if (source.getEntity() instanceof LivingEntity le) {
                    this.setLastHurtByMob(le);
                }
                if (source.getEntity() != null) {
                    double kx = source.getEntity().getX() - this.getX();
                    double kz = source.getEntity().getZ() - this.getZ();
                    this.knockback(0.42, kx, kz);
                }
                this.hurtMarked = true;
                try { level().broadcastDamageEvent(this, source); } catch (Throwable ignored) {}
                try {
                    if (level() instanceof ServerLevel sl) {
                        sl.getChunkSource().broadcast(this,
                            new net.minecraft.network.protocol.game.ClientboundHurtAnimationPacket(this));
                    }
                } catch (Throwable ignored) {}
                try {
                    level().playSound(null, getX(), getY(), getZ(),
                        net.minecraft.sounds.SoundEvents.PLAYER_HURT,
                        net.minecraft.sounds.SoundSource.PLAYERS, 1.0f, 1.0f);
                } catch (Throwable ignored) {}
                if (this.isDeadOrDying()) {
                    this.die(source);
                }
                return true;
            }
            // Even when super.hurt succeeded, ensure knockback & damage broadcast reach watching clients
            this.hurtMarked = true;
            try { level().broadcastDamageEvent(this, source); } catch (Throwable ignored) {}
            try {
                if (level() instanceof ServerLevel sl) {
                    sl.getChunkSource().broadcast(this,
                        new net.minecraft.network.protocol.game.ClientboundHurtAnimationPacket(this));
                }
            } catch (Throwable ignored) {}
            return true;
        }
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
        SomaPlayer player;

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
        int lastCraftHash = Integer.MIN_VALUE;
        List<Map<String, Object>> cachedCraftCands = Collections.emptyList();
        Set<Item> cachedSmeltables = null;
        final BlockPos.MutableBlockPos scanPos = new BlockPos.MutableBlockPos();

        static final double[] SIN8 = new double[8];
        static final double[] COS8 = new double[8];
        static final double[] GOTO_DISTS = {3.0, 7.0};
        static {
            for (int i = 0; i < 8; i++) {
                double ang = i * Math.PI / 4.0;
                SIN8[i] = Math.sin(ang);
                COS8[i] = Math.cos(ang);
            }
        }

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

        // Multiplayer / social interaction state
        final Map<UUID, Integer> playerTrust = new HashMap<>();
        final Map<UUID, Integer> lastSocialTick = new HashMap<>();
        UUID followPlayerUuid = null;
        int lastChatTick = -1000;
        UUID lastChatPlayer = null;
        int lastGlobalChatReplyTick = -1000;
        int lastHurtMobCheckTick = -1;

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
                    + " hp=" + (player == null ? 0 : String.format(Locale.ROOT, "%.1f/%.0f", player.getHealth(), player.getMaxHealth()))
                    + " food=" + (player == null ? 0 : player.getFoodData().getFoodLevel())
                    + " air=" + (player == null ? 0 : player.getAirSupply())
                    + " act=" + activeAction
                    + " inv=" + (player == null ? 0 : countItems())
                    + " gm=" + (player == null ? "-" : player.gameMode.getGameModeForPlayer())
                    + " mobs=" + (player == null ? 0 : countHostiles())
                    + " inTab=" + (player != null && server.getPlayerList().getPlayers().contains(player))
                    + " errors=" + errorCount + " lastError=" + lastError.get()), true);
        }

        void showInventory(net.minecraft.commands.CommandSourceStack src) {
            if (player == null) { src.sendFailure(Component.literal("[soma] player not spawned")); return; }
            Inventory inv = player.getInventory();
            StringBuilder sb = new StringBuilder("[soma] Inventory (selected=" + inv.selected + "): ");
            int count = 0;
            for (int i = 0; i < inv.getContainerSize(); i++) {
                ItemStack st = inv.getItem(i);
                if (st.isEmpty()) continue;
                String nm = BuiltInRegistries.ITEM.getKey(st.getItem()).toString().replace("minecraft:", "");
                String tag = (i >= 36 && i <= 39) ? "A" + (i - 36) : (i == 40 ? "OFF" : String.valueOf(i));
                sb.append("[").append(tag).append(":").append(nm).append("x").append(st.getCount()).append("] ");
                count++;
            }
            if (count == 0) sb.append("(empty)");
            src.sendSuccess(() -> Component.literal(sb.toString()), false);
        }

        void showTrust(net.minecraft.commands.CommandSourceStack src) {
            if (server == null) return;
            StringBuilder sb = new StringBuilder("[soma] Trust:");
            if (playerTrust.isEmpty()) {
                sb.append(" (neutral to all)");
            } else {
                for (var entry : playerTrust.entrySet()) {
                    var sp = server.getPlayerList().getPlayer(entry.getKey());
                    String nm = (sp != null) ? sp.getName().getString() : entry.getKey().toString().substring(0, 8);
                    sb.append(' ').append(nm).append('=').append(entry.getValue());
                }
            }
            if (followPlayerUuid != null) {
                var fp = server.getPlayerList().getPlayer(followPlayerUuid);
                sb.append(" | following=").append(fp != null ? fp.getName().getString() : followPlayerUuid.toString().substring(0, 8));
            }
            src.sendSuccess(() -> Component.literal(sb.toString()), false);
        }

        void tpAdminToSoma(net.minecraft.commands.CommandSourceStack src) {
            if (player == null) { src.sendFailure(Component.literal("[soma] player not spawned")); return; }
            try {
                var admin = src.getPlayerOrException();
                admin.teleportTo(level, player.getX(), player.getY(), player.getZ(), player.getYRot(), player.getXRot());
                src.sendSuccess(() -> Component.literal("[soma] teleported to SOMA"), false);
            } catch (Throwable e) { src.sendFailure(Component.literal("[soma] tp failed: " + e)); }
        }

        void tpSomaToAdmin(net.minecraft.commands.CommandSourceStack src) {
            if (player == null) { src.sendFailure(Component.literal("[soma] player not spawned")); return; }
            try {
                var admin = src.getPlayerOrException();
                player.setPos(admin.getX(), admin.getY(), admin.getZ());
                player.setDeltaMovement(0, 0, 0);
                fallStartY = -1.0;
                src.sendSuccess(() -> Component.literal("[soma] teleported SOMA to you"), true);
            } catch (Throwable e) { src.sendFailure(Component.literal("[soma] tphere failed: " + e)); }
        }

        void healSoma(net.minecraft.commands.CommandSourceStack src) {
            if (player == null) { src.sendFailure(Component.literal("[soma] player not spawned")); return; }
            player.setHealth(player.getMaxHealth());
            player.getFoodData().setFoodLevel(20);
            player.getFoodData().setSaturation(5.0f);
            player.setAirSupply(player.getMaxAirSupply());
            player.clearFire();
            lastHp = player.getHealth();
            src.sendSuccess(() -> Component.literal("[soma] healed SOMA to full"), true);
        }

        void giveStarterKit(net.minecraft.commands.CommandSourceStack src) {
            if (player == null) { src.sendFailure(Component.literal("[soma] player not spawned")); return; }
            Inventory inv = player.getInventory();
            inv.add(new ItemStack(net.minecraft.world.item.Items.WOODEN_PICKAXE));
            inv.add(new ItemStack(net.minecraft.world.item.Items.BREAD, 5));
            inv.add(new ItemStack(net.minecraft.world.item.Items.TORCH, 16));
            src.sendSuccess(() -> Component.literal("[soma] gave starter kit"), true);
        }

        void clearInventory(net.minecraft.commands.CommandSourceStack src) {
            if (player == null) { src.sendFailure(Component.literal("[soma] player not spawned")); return; }
            player.getInventory().clearContent();
            src.sendSuccess(() -> Component.literal("[soma] cleared inventory"), true);
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

        void debugHurt(net.minecraft.commands.CommandSourceStack src, float amount) {
            if (player == null || level == null) return;
            player.invulnerableTime = 0;
            player.clearSpawnInvuln();
            float before = player.getHealth();
            boolean ok = player.hurt(level.damageSources().generic(), amount);
            src.sendSuccess(() -> Component.literal(String.format(Locale.ROOT,
                "[soma] hurt(%.1f) ok=%s hp: %.1f -> %.1f", amount, ok, before, player.getHealth())), true);
        }

        void toggleGod(net.minecraft.commands.CommandSourceStack src) {
            if (player == null) { src.sendFailure(Component.literal("[soma] player not spawned")); return; }
            player.godMode = !player.godMode;
            src.sendSuccess(() -> Component.literal("[soma] godMode=" + player.godMode), true);
        }

        void sayChat(String msg) {
            if (server == null || msg == null || msg.isEmpty()) return;
            server.getPlayerList().broadcastSystemMessage(Component.literal("<SOMA> " + msg), false);
        }

        void onPlayerChat(net.minecraft.server.level.ServerPlayer sender, String msg) {
            if (player == null || sender == null || sender == player || msg == null || level == null) return;
            if (sender.level() != level) return;
            String low = msg.toLowerCase(Locale.ROOT);
            boolean mentioned = low.contains("soma") || low.contains("сома");
            if (!mentioned && sender.distanceTo(player) > 32.0) return;

            lastChatTick = tickCounter;
            lastChatPlayer = sender.getUUID();
            pushEvent("interact", 1, sender.blockPosition(), "chat");

            UUID uid = sender.getUUID();
            String name = sender.getName().getString();
            if (low.contains("follow") || low.contains("come") || low.contains("иди")
                    || low.contains("ко мне") || low.contains("сюда") || low.contains("за мной")) {
                followPlayerUuid = uid;
                int tr = playerTrust.getOrDefault(uid, 0);
                if (tr < 5) playerTrust.put(uid, Math.max(0, tr + 1));
                if (tickCounter - lastGlobalChatReplyTick >= 80) {
                    sayChat("Иду, " + name + "!");
                    lastGlobalChatReplyTick = tickCounter;
                }
            } else if (low.contains("stop") || low.contains("stay") || low.contains("стой")
                    || low.contains("жди") || low.contains("хватит")) {
                if (uid.equals(followPlayerUuid)) followPlayerUuid = null;
                if (tickCounter - lastGlobalChatReplyTick >= 80) {
                    sayChat("Ок, остаюсь тут.");
                    lastGlobalChatReplyTick = tickCounter;
                }
            } else if (low.contains("hi") || low.contains("hello") || low.contains("привет")
                    || low.contains("ку") || low.contains("здаров")) {
                playerTrust.put(uid, Math.min(10, playerTrust.getOrDefault(uid, 0) + 1));
                if (tickCounter - lastGlobalChatReplyTick >= 100) {
                    sayChat("Привет, " + name + "!");
                    lastGlobalChatReplyTick = tickCounter;
                }
            }
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
                player = new SomaPlayer(level, profile);
                BlockPos spawn = level.getSharedSpawnPos();
                player.setPos(spawn.getX() + 0.5, findGroundY(spawn.getX(), spawn.getZ()), spawn.getZ() + 0.5);
                setSurvival();
                attachToServer();
                player.invulnerableTime = 0;
                player.clearSpawnInvuln();
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
            } catch (Throwable e) {
                errorCount++;
                lastError.set("pickup: " + e);
            }
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
                    } else if (st.getItem() instanceof net.minecraft.world.item.ShieldItem && player.getOffhandItem().isEmpty()) {
                        player.setItemInHand(InteractionHand.OFF_HAND, st.copy());
                        inv.setItem(i, ItemStack.EMPTY);
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
                player.clearSpawnInvuln();
                if (player.hurtTime > 0) player.hurtTime--;
                try { player.getFoodData().tick(player); } catch (Throwable ignored) {}
                player.bumpAttackCooldown();

                survivalTick();
                handleDeathAndRespawn();
                pickupNearbyItems();
                advanceMining();

                if (warmup > 0) { physicsTick(); warmup--; return; }
                tickCounter++;
                if ("goto".equals(activeAction) && activeTarget != null) {
                    driveActive();
                } else {
                    physicsTick();
                }

                if (tickCounter % OBS_TICKS != 0) return;
                sendObservation();
                pollAndExecute();
            } catch (Throwable e) {
                errorCount++;
                lastError.set(String.valueOf(e));
            }
        }

        /** Always-on gravity & ground friction so SOMA never floats in mid-air. */
        void physicsTick() {
            if (player == null) return;
            Vec3 d = player.getDeltaMovement();
            double nvx = d.x * (player.onGround() ? 0.6 : 0.91);
            double nvz = d.z * (player.onGround() ? 0.6 : 0.91);
            double nvy;
            if (inWater()) {
                nvy = (d.y + 0.02) * 0.85;
            } else if (player.onClimbable()) {
                nvy = Math.max(d.y * 0.6, -0.15);
            } else if (player.onGround() && d.y <= 0.0) {
                nvy = -0.04;
            } else {
                nvy = Math.max(d.y - 0.08, -0.78);
            }
            if (Math.abs(nvx) < 1e-4) nvx = 0.0;
            if (Math.abs(nvz) < 1e-4) nvz = 0.0;
            player.setDeltaMovement(nvx, nvy, nvz);
            player.move(MoverType.SELF, player.getDeltaMovement());

            // Natural idle head look: face nearest player within 10 blocks if idle
            if ("none".equals(activeAction) || "wait".equals(activeAction)) {
                Player nearest = null;
                double bestD2 = 100.0;
                for (Player p : level.players()) {
                    if (p == player || !p.isAlive() || p.isSpectator()) continue;
                    double d2 = p.distanceToSqr(player);
                    if (d2 < bestD2) { bestD2 = d2; nearest = p; }
                }
                if (nearest != null) {
                    if (tickCounter % 4 == 0) {
                        double dx = nearest.getX() - player.getX();
                        double dz = nearest.getZ() - player.getZ();
                        faceTowards(dx, dz);
                        double hdist = Math.max(0.1, Math.hypot(dx, dz));
                        float targetPitch = (float) -Math.toDegrees(Math.atan2(nearest.getEyeY() - player.getEyeY(), hdist));
                        player.setXRot(net.minecraft.util.Mth.clamp(targetPitch, -45.0f, 45.0f));
                    }
                } else if (tickCounter % 20 == 0) {
                    float pitch = (float) (Math.sin(tickCounter * 0.05) * 6.0);
                    player.setXRot(net.minecraft.util.Mth.clamp(pitch, -30.0f, 30.0f));
                    player.setYHeadRot(player.getYRot());
                }
            }
        }

        void survivalTick() {
            if (player == null) return;
            try {
                // Track player attackers -> decrease trust and stop following them
                LivingEntity lastAttacker = player.getLastHurtByMob();
                if (lastAttacker instanceof Player ap && ap != player
                        && player.getLastHurtByMobTimestamp() != lastHurtMobCheckTick) {
                    lastHurtMobCheckTick = player.getLastHurtByMobTimestamp();
                    UUID auid = ap.getUUID();
                    playerTrust.put(auid, Math.max(-10, playerTrust.getOrDefault(auid, 0) - 2));
                    if (auid.equals(followPlayerUuid)) followPlayerUuid = null;
                }
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
            if (player.hurtTime > 6) {
                physicsTick();
                return;
            }
            boolean sprint = player.getFoodData().getFoodLevel() > 6 && !player.isCrouching();
            try { player.setSprinting(sprint); } catch (Throwable ignored) {}
            Vec3 d = player.getDeltaMovement();
            double speed = player.isCrouching() ? 0.065 : (sprint ? 0.16 : 0.12);
            double tx = dx / dist * speed;
            double tz = dz / dist * speed;
            double nvx = (d.x + (tx - d.x) * 0.35) * 0.91;
            double nvz = (d.z + (tz - d.z) * 0.35) * 0.91;
            double nvy = player.onClimbable() ? (player.horizontalCollision ? 0.2 : Math.max(d.y * 0.6, -0.15)) : Math.max(d.y - 0.08, -0.78);
            player.setDeltaMovement(nvx, nvy, nvz);
            if (player.horizontalCollision && player.onGround() && !player.onClimbable()) {
                player.setDeltaMovement(nvx, 0.42, nvz);
                try { player.getFoodData().addExhaustion(0.05f); } catch (Throwable ignored) {}
                pushEvent("jump", 0, null);
            }
            player.move(MoverType.SELF, player.getDeltaMovement());
            try { player.getFoodData().addExhaustion(sprint ? 0.01f : 0.002f); } catch (Throwable ignored) {}
        }

        void faceTowards(double dx, double dz) {
            float target = (float) Math.toDegrees(Math.atan2(-dx, dz));
            float cur = player.getYRot();
            float diff = net.minecraft.util.Mth.wrapDegrees(target - cur);
            float step = net.minecraft.util.Mth.clamp(diff, -28.0f, 28.0f);
            float yaw = cur + step;
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
            boolean urgent = "attack".equals(nextAct) || "eat".equals(nextAct) || "stop".equals(nextAct) || player.hurtTime >= 7;
            if (!urgent && activeTarget != null) {
                if ("mine".equals(activeAction) && miningPos != null && miningTicks < 120) {
                    return; // let block mining finish instead of cancelling every 4 ticks
                }
                if ("goto".equals(activeAction) && activeTicks < 16) {
                    return; // commit to movement for at least 16 ticks (0.8s)
                }
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
                    try {
                        player.swing(InteractionHand.MAIN_HAND);
                        level.getChunkSource().broadcast(player,
                            new net.minecraft.network.protocol.game.ClientboundAnimatePacket(player, 0));
                    } catch (Throwable ignored) {}
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
                    BlockState st = bs(p);
                    var be = level.getBlockEntity(p);
                    if (st != null && st.getBlock() instanceof BedBlock) {
                        player.setRespawnPosition(level.dimension(), p, player.getYRot(), false, true);
                        long dt = level.getDayTime() % 24000;
                        if (dt >= 12541 && dt <= 23458) {
                            player.heal(2.0f);
                        }
                        pushEvent("use", 1, p, "bed");
                        finish(true, "bed_used");
                    } else if (be instanceof net.minecraft.world.level.block.entity.AbstractFurnaceBlockEntity furn) {
                        Inventory inv = player.getInventory();
                        boolean did = false;
                        ItemStack outStack = furn.getItem(2);
                        if (!outStack.isEmpty()) {
                            String oname = BuiltInRegistries.ITEM.getKey(outStack.getItem()).toString();
                            int cnt = outStack.getCount();
                            ItemStack outCopy = outStack.copy();
                            inv.add(outCopy);
                            int taken = cnt - outCopy.getCount();
                            if (taken > 0) {
                                furn.setItem(2, outCopy.isEmpty() ? ItemStack.EMPTY : outCopy);
                                pushEvent("pickup", taken, p, oname);
                                did = true;
                            }
                        }
                        if (furn.getItem(0).isEmpty()) {
                            for (int i = 0; i < 36; i++) {
                                ItemStack ist = inv.getItem(i);
                                if (!ist.isEmpty() && isSmeltable(ist)) {
                                    furn.setItem(0, ist.split(Math.min(ist.getCount(), 8)));
                                    did = true;
                                    break;
                                }
                            }
                        }
                        if (furn.getItem(1).isEmpty()) {
                            for (int i = 0; i < 36; i++) {
                                ItemStack fst = inv.getItem(i);
                                if (!fst.isEmpty() && net.minecraft.world.level.block.entity.AbstractFurnaceBlockEntity.isFuel(fst)) {
                                    furn.setItem(1, fst.split(Math.min(fst.getCount(), 8)));
                                    did = true;
                                    break;
                                }
                            }
                        }
                        if (did) furn.setChanged();
                        pushEvent("use", 1, p, "furnace");
                        finish(true, "furnace_used");
                    } else if (be instanceof net.minecraft.world.Container cont) {
                        Inventory inv = player.getInventory();
                        int looted = 0;
                        for (int i = 0; i < cont.getContainerSize() && looted < 4; i++) {
                            ItemStack st2 = cont.getItem(i);
                            if (st2.isEmpty()) continue;
                            String nm = BuiltInRegistries.ITEM.getKey(st2.getItem()).toString();
                            int cnt = st2.getCount();
                            ItemStack copy = st2.copy();
                            inv.add(copy);
                            int taken = cnt - copy.getCount();
                            if (taken > 0) {
                                cont.setItem(i, copy.isEmpty() ? ItemStack.EMPTY : copy);
                                pushEvent("pickup", taken, p, nm);
                                looted++;
                            }
                        }
                        if (looted > 0) cont.setChanged();
                        pushEvent("use", 1, p, String.valueOf(c.getOrDefault("n", "container")));
                        finish(true, "looted");
                    } else {
                        BlockHitResult hit = new BlockHitResult(
                            new Vec3(p.getX() + 0.5, p.getY() + 0.5, p.getZ() + 0.5),
                            Direction.UP, p, false);
                        player.gameMode.useItemOn(player, level, player.getMainHandItem(), InteractionHand.MAIN_HAND, hit);
                        pushEvent("use", 1, p, String.valueOf(c.getOrDefault("n", "")));
                        finish(true, "used");
                    }
                }
                case "shoot" -> {
                    int eid = ip(c, "eid");
                    Entity e = level.getEntity(eid);
                    int bowSlot = findItemSlot(net.minecraft.world.item.BowItem.class);
                    int arrSlot = findItemSlot(net.minecraft.world.item.ArrowItem.class);
                    if (e == null || !e.isAlive() || bowSlot < 0 || arrSlot < 0 || e.distanceTo(player) > 24.0) {
                        pushEvent("act_fail", 1, null);
                        return;
                    }
                    Inventory inv = player.getInventory();
                    if (bowSlot < 9) inv.selected = bowSlot;
                    ItemStack bowStack = inv.getItem(bowSlot);
                    ItemStack arrStack = inv.getItem(arrSlot);
                    Vec3 eye = player.getEyePosition();
                    double dx = e.getX() - eye.x;
                    double dy = (e.getY() + e.getBbHeight() * 0.5) - eye.y;
                    double dz = e.getZ() - eye.z;
                    double horiz = Math.hypot(dx, dz);
                    player.setYRot((float) Math.toDegrees(Math.atan2(-dx, dz)));
                    player.setYHeadRot(player.getYRot());
                    player.setXRot((float) Math.toDegrees(-Math.atan2(dy + horiz * 0.12, horiz)));
                    try {
                        var arrow = ((net.minecraft.world.item.ArrowItem) arrStack.getItem())
                            .createArrow(level, arrStack, player, bowStack);
                        arrow.shootFromRotation(player, player.getXRot(), player.getYRot(), 0.0f, 2.8f, 1.0f);
                        level.addFreshEntity(arrow);
                        arrStack.shrink(1);
                        level.playSound(null, player.getX(), player.getY(), player.getZ(),
                            net.minecraft.sounds.SoundEvents.ARROW_SHOOT,
                            net.minecraft.sounds.SoundSource.PLAYERS, 1.0f, 1.0f);
                        pushEvent("attack", 1, e.blockPosition(), BuiltInRegistries.ENTITY_TYPE.getKey(e.getType()).toString());
                        finish(true, "shot");
                    } catch (Throwable ex) {
                        pushEvent("act_fail", 1, null);
                    }
                }
                case "pillar" -> {
                    int bslot = selectBlockSlot();
                    if (bslot < 0 || !player.onGround()) {
                        pushEvent("act_fail", 1, null);
                        return;
                    }
                    BlockPos feet = player.blockPosition();
                    player.getInventory().selected = bslot;
                    player.setDeltaMovement(0.0, 0.42, 0.0);
                    player.move(MoverType.SELF, new Vec3(0.0, 1.05, 0.0));
                    BlockHitResult hit = new BlockHitResult(
                        new Vec3(feet.getX() + 0.5, feet.getY(), feet.getZ() + 0.5),
                        Direction.UP, feet.below(), false);
                    player.gameMode.useItemOn(player, level, player.getMainHandItem(), InteractionHand.MAIN_HAND, hit);
                    pushEvent("place", 1, feet, BuiltInRegistries.ITEM.getKey(player.getMainHandItem().getItem()).toString());
                    finish(true, "pillared");
                }
                case "place" -> {
                    BlockPos p = new BlockPos(ip(c, "x"), ip(c, "y"), ip(c, "z"));
                    int slot = selectBlockSlot();
                    if (slot < 0) { pushEvent("act_fail", 1, null); return; }
                    player.getInventory().selected = slot;
                    BlockPos support = null;
                    Direction face = Direction.UP;
                    for (Direction dir : Direction.values()) {
                        BlockPos nb = p.relative(dir);
                        BlockState nbs = bs(nb);
                        if (nbs != null && nbs.isFaceSturdy(level, nb, dir.getOpposite())) {
                            support = nb;
                            face = dir.getOpposite();
                            break;
                        }
                    }
                    if (support == null) {
                        support = p.below();
                        face = Direction.UP;
                    }
                    BlockHitResult hit = new BlockHitResult(
                        new Vec3(p.getX() + 0.5, p.getY() + 0.5, p.getZ() + 0.5),
                        face, support, false);
                    String placedName = BuiltInRegistries.ITEM.getKey(player.getMainHandItem().getItem()).toString();
                    player.gameMode.useItemOn(player, level, player.getMainHandItem(), InteractionHand.MAIN_HAND, hit);
                    pushEvent("place", 1, p, placedName);
                    finish(true, "placed");
                }
                case "craft" -> {
                    String rid = String.valueOf(c.getOrDefault("rid", ""));
                    RecipeHolder<CraftingRecipe> rh = craftableById.get(rid);
                    int repeats = (ip(c, "shift") > 0) ? Math.max(1, Math.min(16, ip(c, "cnt") > 0 ? ip(c, "cnt") : 16)) : 1;
                    int craftedCount = (rh != null) ? executeCraft(rh.value(), repeats) : 0;
                    if (craftedCount <= 0) {
                        pushEvent("act_fail", 1, null);
                        return;
                    }
                    ItemStack res = rh.value().getResultItem(level.registryAccess());
                    String nm = BuiltInRegistries.ITEM.getKey(res.getItem()).toString();
                    pushEvent("craft", craftedCount, player.blockPosition(), nm);
                    finish(true, repeats > 1 ? "shift_crafted" : "crafted");
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
                        String foodName = BuiltInRegistries.ITEM.getKey(held.getItem()).toString();
                        player.getFoodData().eat(foodProps);
                        held.shrink(1);
                        try {
                            level.playSound(null, player.getX(), player.getY(), player.getZ(),
                                net.minecraft.sounds.SoundEvents.GENERIC_EAT,
                                net.minecraft.sounds.SoundSource.PLAYERS, 1.0f, 1.0f);
                        } catch (Throwable ignored) {}
                        pushEvent("eat", 1, null, foodName);
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
                    double dx = e.getX() - player.getX();
                    double dz = e.getZ() - player.getZ();
                    faceTowards(dx, dz);
                    player.swing(InteractionHand.MAIN_HAND);
                    if (e instanceof Player p) {
                        double hd = Math.max(0.1, Math.hypot(dx, dz));
                        float pitch = (float) -Math.toDegrees(Math.atan2(p.getEyeY() - player.getEyeY(), hd));
                        player.setXRot(net.minecraft.util.Mth.clamp(pitch, -45.0f, 45.0f));
                        UUID uid = p.getUUID();
                        playerTrust.put(uid, Math.min(10, playerTrust.getOrDefault(uid, 0) + 1));
                        int lastGreet = lastSocialTick.getOrDefault(uid, -1000);
                        if (tickCounter - lastGreet >= 200 && tickCounter - lastGlobalChatReplyTick >= 80) {
                            lastSocialTick.put(uid, tickCounter);
                            lastGlobalChatReplyTick = tickCounter;
                            sayChat("Привет, " + p.getName().getString() + "!");
                        }
                        pushEvent("interact", 1, e.blockPosition(), "player");
                    } else {
                        player.interactOn(e, InteractionHand.MAIN_HAND);
                        pushEvent("interact", 1, e.blockPosition(), "npc");
                    }
                    finish(true, "interacted");
                }
                case "give" -> {
                    int eid = ip(c, "eid");
                    Entity e = level.getEntity(eid);
                    if (!(e instanceof Player p) || !e.isAlive() || e.distanceTo(player) > REACH + 1.5) {
                        pushEvent("act_fail", 1, null);
                        return;
                    }
                    int sslot = findShareSlot();
                    if (sslot < 0) {
                        pushEvent("act_fail", 1, null);
                        return;
                    }
                    ItemStack st = player.getInventory().getItem(sslot);
                    if (st.isEmpty()) {
                        pushEvent("act_fail", 1, null);
                        return;
                    }
                    double dx = p.getX() - player.getX();
                    double dz = p.getZ() - player.getZ();
                    faceTowards(dx, dz);
                    player.swing(InteractionHand.MAIN_HAND);
                    ItemStack gift = st.split(1);
                    ItemEntity drop = player.drop(gift, false);
                    if (drop != null) {
                        double hd = Math.max(0.1, Math.hypot(dx, dz));
                        drop.setDeltaMovement((dx / hd) * 0.25, 0.15, (dz / hd) * 0.25);
                    }
                    UUID uid = p.getUUID();
                    playerTrust.put(uid, Math.min(10, playerTrust.getOrDefault(uid, 0) + 2));
                    int lastGreet = lastSocialTick.getOrDefault(uid, -1000);
                    if (tickCounter - lastGreet >= 200 && tickCounter - lastGlobalChatReplyTick >= 80) {
                        lastSocialTick.put(uid, tickCounter);
                        lastGlobalChatReplyTick = tickCounter;
                        sayChat("Держи, " + p.getName().getString() + "!");
                    }
                    pushEvent("interact", 2, p.blockPosition(), "give");
                    finish(true, "gave");
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
        int maxCraftCount(CraftingRecipe recipe, Map<Item, Integer> invCounts, int cap) {
            List<Ingredient> ings = recipe.getIngredients();
            if (ings.isEmpty()) return 0;
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
                if (!matched) return 0;
            }
            if (needed.isEmpty()) return 0;
            int maxTimes = cap;
            for (Map.Entry<Item, Integer> e : needed.entrySet()) {
                int avail = invCounts.getOrDefault(e.getKey(), 0);
                maxTimes = Math.min(maxTimes, avail / e.getValue());
            }
            return Math.max(0, maxTimes);
        }

        boolean canCraftRecipe(CraftingRecipe recipe, Map<Item, Integer> invCounts) {
            return maxCraftCount(recipe, invCounts, 1) >= 1;
        }

        int executeCraft(CraftingRecipe recipe, int maxRepeats) {
            ItemStack baseOut = recipe.getResultItem(level.registryAccess());
            if (baseOut.isEmpty()) return 0;
            int perBatch = baseOut.getCount();
            int stackCap = Math.max(1, baseOut.getMaxStackSize() / perBatch);
            int limit = Math.max(1, Math.min(maxRepeats, stackCap));
            Inventory inv = player.getInventory();
            int totalCrafted = 0;

            for (int r = 0; r < limit; r++) {
                Map<Integer, Integer> usedPerSlot = new HashMap<>();
                boolean ok = true;
                for (Ingredient ing : recipe.getIngredients()) {
                    if (ing.isEmpty()) continue;
                    boolean matched = false;
                    for (int i = 0; i < 36; i++) {
                        ItemStack st = inv.getItem(i);
                        if (st.isEmpty()) continue;
                        int already = usedPerSlot.getOrDefault(i, 0);
                        if (st.getCount() - already > 0 && ing.test(st)) {
                            usedPerSlot.put(i, already + 1);
                            matched = true;
                            break;
                        }
                    }
                    if (!matched) { ok = false; break; }
                }
                if (!ok) break;

                List<ItemStack> remainders = new ArrayList<>();
                for (Map.Entry<Integer, Integer> e : usedPerSlot.entrySet()) {
                    int slot = e.getKey();
                    ItemStack st = inv.getItem(slot);
                    ItemStack rem = st.getItem().hasCraftingRemainingItem() ? new ItemStack(st.getItem().getCraftingRemainingItem()) : ItemStack.EMPTY;
                    if (!rem.isEmpty()) {
                        for (int k = 0; k < e.getValue(); k++) remainders.add(rem.copy());
                    }
                    st.shrink(e.getValue());
                    if (st.isEmpty()) inv.setItem(slot, ItemStack.EMPTY);
                }
                ItemStack outStack = baseOut.copy();
                if (!inv.add(outStack) && !outStack.isEmpty()) {
                    player.drop(outStack, false);
                }
                for (ItemStack rem : remainders) {
                    if (!inv.add(rem) && !rem.isEmpty()) {
                        player.drop(rem, false);
                    }
                }
                totalCrafted += perBatch;
            }
            return totalCrafted;
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
            boolean hasBow = false, hasArrow = false, hasShield = player.getOffhandItem().getItem() instanceof net.minecraft.world.item.ShieldItem;
            int freeSlots = 0;
            for (int i = 0; i < inv.getContainerSize(); i++) {
                ItemStack st = inv.getItem(i);
                if (st.isEmpty()) {
                    if (i < 36) freeSlots++;
                    continue;
                }
                Item it = st.getItem();
                if (it instanceof net.minecraft.world.item.ShieldItem && player.getOffhandItem().isEmpty() && i < 36) {
                    player.setItemInHand(InteractionHand.OFF_HAND, st.copy());
                    inv.setItem(i, ItemStack.EMPTY);
                    hasShield = true;
                    freeSlots++;
                    continue;
                }
                if (i < 36) invCounts.merge(it, st.getCount(), Integer::sum);
                int isFood = st.get(DataComponents.FOOD) != null ? 1 : 0;
                int isTool = (it instanceof TieredItem || it instanceof DiggerItem || it instanceof SwordItem) ? 1 : 0;
                int isBlock = (it instanceof BlockItem) ? 1 : 0;
                int isArmor = (it instanceof ArmorItem || it instanceof Equipable) ? 1 : 0;
                double dur = st.isDamageableItem() ? (1.0 - (double) st.getDamageValue() / Math.max(1, st.getMaxDamage())) : 1.0;
                if (isFood == 1) hasFood = true;
                if (it instanceof DiggerItem) hasTool = true;
                if (it instanceof SwordItem) hasWeapon = true;
                if (isBlock == 1) hasBlocks = true;
                if (it instanceof net.minecraft.world.item.BowItem) hasBow = true;
                if (it instanceof net.minecraft.world.item.ArrowItem) hasArrow = true;
                if (it instanceof net.minecraft.world.item.ShieldItem) hasShield = true;

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
            double minLavaDist = 99.0;
            int k = 0;
            BlockPos.MutableBlockPos p = scanPos;
            for (int ix = 0; ix < GRID_N; ix++) {
                int gx = bx - 3 + ix;
                for (int iy = 0; iy < GRID_N; iy++) {
                    int gy = by - 2 + iy;
                    for (int iz = 0; iz < GRID_N; iz++) {
                        p.set(gx, gy, bz - 3 + iz);
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
                            if (st.getFluidState().is(FluidTags.LAVA) || st.is(Blocks.FIRE) || st.is(Blocks.SOUL_FIRE)) {
                                double ld = Math.sqrt(p.distToCenterSqr(pos));
                                if (ld < minLavaDist) minLavaDist = ld;
                            }
                        }
                        gridIds[k] = id;
                        gridFlags[k] = fl;
                        k++;
                    }
                }
            }

            boolean sky = false;
            try { sky = level.canSeeSky(new BlockPos(bx, by + 1, bz)); } catch (Throwable ignored) {}
            int walls = 0;
            if (!passable(new BlockPos(bx + 1, by, bz)) || !passable(new BlockPos(bx + 1, by + 1, bz))) walls++;
            if (!passable(new BlockPos(bx - 1, by, bz)) || !passable(new BlockPos(bx - 1, by + 1, bz))) walls++;
            if (!passable(new BlockPos(bx, by, bz + 1)) || !passable(new BlockPos(bx, by + 1, bz + 1))) walls++;
            if (!passable(new BlockPos(bx, by, bz - 1)) || !passable(new BlockPos(bx, by + 1, bz - 1))) walls++;
            int roof = (!sky || !passable(new BlockPos(bx, by + 2, bz))) ? 1 : 0;
            double shelter = (walls + roof) / 5.0;

            boolean debuff = false, buffProt = false, buffAtk = false;
            try {
                for (var inst : player.getActiveEffects()) {
                    var eff = inst.getEffect();
                    if (eff.value().getCategory() == net.minecraft.world.effect.MobEffectCategory.HARMFUL) debuff = true;
                    if (eff.is(net.minecraft.world.effect.MobEffects.DAMAGE_RESISTANCE)
                        || eff.is(net.minecraft.world.effect.MobEffects.REGENERATION)
                        || eff.is(net.minecraft.world.effect.MobEffects.FIRE_RESISTANCE)
                        || eff.is(net.minecraft.world.effect.MobEffects.ABSORPTION)) buffProt = true;
                    if (eff.is(net.minecraft.world.effect.MobEffects.DAMAGE_BOOST)
                        || eff.is(net.minecraft.world.effect.MobEffects.DIG_SPEED)
                        || eff.is(net.minecraft.world.effect.MobEffects.MOVEMENT_SPEED)) buffAtk = true;
                }
            } catch (Throwable ignored) {}

            Vec3 look = player.getLookAngle();
            BlockPos ahead = BlockPos.containing(pos.x + look.x * 1.5, pos.y, pos.z + look.z * 1.5);
            boolean cliffAhead = passable(ahead) && passable(ahead.below()) && passable(ahead.below(2)) && passable(ahead.below(3));
            boolean rayBlk = false;
            try { rayBlk = player.pick(REACH, 0.0f, false).getType() == net.minecraft.world.phys.HitResult.Type.BLOCK; } catch (Throwable ignored) {}

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
            self.put("sky", sky);
            self.put("shelter", round(shelter));
            self.put("debuff", debuff);
            self.put("buff_prot", buffProt);
            self.put("buff_atk", buffAtk);
            self.put("climb", player.onClimbable());
            self.put("has_bow", hasBow && hasArrow);
            self.put("has_shield", hasShield);
            self.put("lava_near", minLavaDist < 6.0 ? round(1.0 - minLavaDist / 6.0) : 0.0);
            self.put("cliff_ahead", cliffAhead);
            self.put("inv_free", round(freeSlots / 36.0));
            self.put("xp", player.experienceLevel);
            self.put("ray_blk", rayBlk);
            self.put("chat_ping", (tickCounter - lastChatTick) < 60);

            // Entities (v2 12-field schema)
            AABB area = new AABB(bx - 24, by - 16, bz - 24, bx + 24, by + 16, bz + 24);
            List<Entity> ents = new ArrayList<>(level.getEntitiesOfClass(Entity.class, area,
                e -> e != player && e.isAlive()));
            ents.sort(Comparator.comparingDouble(e -> e.distanceToSqr(player)));
            boolean rayEnt = false;
            Vec3 eyePos = player.getEyePosition();
            Vec3 rayEnd = eyePos.add(look.scale(REACH));
            List<List<Object>> entOut = new ArrayList<>();
            boolean seenNearestPlayer = false;
            double nearTrust = 0.0;
            for (Entity e : ents) {
                if (!rayEnt && e.distanceTo(player) <= REACH) {
                    if (e.getBoundingBox().inflate(0.3).clip(eyePos, rayEnd).isPresent()) rayEnt = true;
                }
                if (!seenNearestPlayer && e instanceof Player pl) {
                    seenNearestPlayer = true;
                    nearTrust = playerTrust.getOrDefault(pl.getUUID(), 0) / 10.0;
                }
                if (entOut.size() >= MAX_ENT) continue;
                int role = 0;
                if (e instanceof Player) role = 2;
                else if (e instanceof ItemEntity) role = 3;
                else if (e instanceof Projectile) role = 4;
                else if (isHostile(e)) role = 1;

                int tgtMe = (e instanceof Mob m && m.getTarget() == player) ? 1
                    : (e instanceof Player pl && playerTrust.getOrDefault(pl.getUUID(), 0) < 0 ? 1 : 0);
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
            self.put("ray_ent", rayEnt);
            self.put("near_trust", round(nearTrust));

            // Kill check
            if (lastAttackedId >= 0) {
                Entity ae = level.getEntity(lastAttackedId);
                if (ae == null || !ae.isAlive()) {
                    pushEvent("kill", 1, player.blockPosition(), lastAttackedName);
                    lastAttackedId = -1;
                }
            }

            int nextSeq = ++seq;
            List<Map<String, Object>> cands = buildCandidates(bx, by, bz, ents, invCounts, nearCraft, hasBow && hasArrow);
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
                List<Entity> ents, Map<Item, Integer> invCounts, boolean nearCraft, boolean canShoot) {

            List<Map<String, Object>> out = new ArrayList<>();
            Entity nearestHostile = null;
            double nearestHostileDist = 999.0;

            // 1. Combat & Entity interaction
            int shareSlot = findShareSlot();
            String shareName = (shareSlot >= 0)
                ? BuiltInRegistries.ITEM.getKey(player.getInventory().getItem(shareSlot).getItem()).toString()
                : "";
            boolean sawFollowPlayer = false;

            for (Entity e : ents) {
                double dist = e.distanceTo(player);
                boolean host = isHostile(e);
                if (host && dist < nearestHostileDist) {
                    nearestHostile = e;
                    nearestHostileDist = dist;
                }
                if (dist <= REACH) {
                    if (host || e instanceof net.minecraft.world.entity.animal.Animal) {
                        Map<String, Object> m = new HashMap<>();
                        m.put("a", "attack"); m.put("eid", e.getId());
                        m.put("n", BuiltInRegistries.ENTITY_TYPE.getKey(e.getType()).toString());
                        m.put("d", round(dist));
                        m.put("h", host ? 1 : 0);
                        boolean tgtMe = (e instanceof Mob mb && mb.getTarget() == player);
                        m.put("tgt", tgtMe ? 1 : 0);
                        if (host && !tgtMe) {
                            // Prioritize hostiles threatening nearby friendly players
                            boolean nearFriend = false;
                            Entity mobTgt = (e instanceof Mob mb) ? mb.getTarget() : null;
                            if (mobTgt instanceof Player fp && playerTrust.getOrDefault(fp.getUUID(), 0) >= 0) {
                                nearFriend = true;
                            } else {
                                for (Player fp : level.players()) {
                                    if (fp != player && fp.isAlive() && playerTrust.getOrDefault(fp.getUUID(), 0) >= 0
                                            && fp.distanceTo(e) < 4.5) {
                                        nearFriend = true;
                                        break;
                                    }
                                }
                            }
                            if (nearFriend) m.put("val", 0.9);
                        }
                        out.add(m);
                    }
                    if (e instanceof Player p) {
                        int trust = playerTrust.getOrDefault(p.getUUID(), 0);
                        if (p.getUUID().equals(followPlayerUuid)) sawFollowPlayer = true;
                        if (trust < 0) {
                            Map<String, Object> atk = new HashMap<>();
                            atk.put("a", "attack"); atk.put("eid", p.getId());
                            atk.put("n", p.getName().getString());
                            atk.put("d", round(dist));
                            atk.put("h", 1); atk.put("tgt", 1); atk.put("player", 1);
                            out.add(atk);
                        } else {
                            Map<String, Object> m = new HashMap<>();
                            m.put("a", "interact_entity"); m.put("eid", p.getId());
                            m.put("n", p.getName().getString());
                            m.put("d", round(dist));
                            m.put("player", 1);
                            out.add(m);
                            if (shareSlot >= 0) {
                                Map<String, Object> g = new HashMap<>();
                                g.put("a", "give"); g.put("eid", p.getId());
                                g.put("n", shareName);
                                g.put("d", round(dist));
                                g.put("player", 1);
                                g.put("val", 0.65);
                                out.add(g);
                            }
                        }
                    } else if (e instanceof net.minecraft.world.entity.npc.AbstractVillager) {
                        Map<String, Object> m = new HashMap<>();
                        m.put("a", "interact_entity"); m.put("eid", e.getId());
                        m.put("n", e.getName().getString());
                        m.put("d", round(dist));
                        m.put("player", 0);
                        out.add(m);
                    }
                } else if (canShoot && host && dist <= 18.0) {
                    Map<String, Object> m = new HashMap<>();
                    m.put("a", "shoot"); m.put("eid", e.getId());
                    m.put("n", BuiltInRegistries.ENTITY_TYPE.getKey(e.getType()).toString());
                    m.put("d", round(dist));
                    m.put("h", 1);
                    m.put("val", 0.9);
                    out.add(m);
                } else if (dist < 24.0 && (e instanceof ItemEntity || host || e instanceof Player)) {
                    boolean isFollow = (e instanceof Player fp && fp.getUUID().equals(followPlayerUuid));
                    if (isFollow) sawFollowPlayer = true;
                    Map<String, Object> m = new HashMap<>();
                    m.put("a", "goto");
                    m.put("x", (int) Math.floor(e.getX()));
                    m.put("y", (int) Math.floor(e.getY()));
                    m.put("z", (int) Math.floor(e.getZ()));
                    m.put("d", round(dist));
                    m.put("h", host ? 1 : 0);
                    m.put("item_drop", (e instanceof ItemEntity) ? 1 : 0);
                    m.put("player", (e instanceof Player) ? 1 : 0);
                    if (isFollow && dist > 3.5) m.put("val", 0.9);
                    out.add(m);
                }
                if (out.size() >= 12) break;
            }

            if (!sawFollowPlayer && followPlayerUuid != null) {
                Player fp = level.getPlayerByUUID(followPlayerUuid);
                if (fp != null && fp.isAlive() && fp.level() == level) {
                    double fd = fp.distanceTo(player);
                    if (fd > 3.5 && fd < 48.0) {
                        Map<String, Object> m = new HashMap<>();
                        m.put("a", "goto");
                        m.put("x", (int) Math.floor(fp.getX()));
                        m.put("y", (int) Math.floor(fp.getY()));
                        m.put("z", (int) Math.floor(fp.getZ()));
                        m.put("d", round(fd));
                        m.put("player", 1);
                        m.put("val", 0.9);
                        out.add(m);
                    }
                }
            }

            // 1b. Tactical flee / kite when low HP & hostile nearby
            if (player.getHealth() < 10.0f && nearestHostile != null && nearestHostileDist < 8.0) {
                double fdx = player.getX() - nearestHostile.getX();
                double fdz = player.getZ() - nearestHostile.getZ();
                double fd = Math.max(0.1, Math.hypot(fdx, fdz));
                int fx = (int) Math.floor(player.getX() + (fdx / fd) * 8.0);
                int fz = (int) Math.floor(player.getZ() + (fdz / fd) * 8.0);
                int fy = standY(fx, by, fz);
                if (fy != Integer.MIN_VALUE) {
                    Map<String, Object> m = new HashMap<>();
                    m.put("a", "goto"); m.put("x", fx); m.put("y", fy); m.put("z", fz);
                    m.put("d", 8.0); m.put("val", 0.95);
                    out.add(m);
                }
            }

            // 2. Eat if hungry
            if (player.getFoodData().getFoodLevel() < 18 && findFoodSlot() >= 0) {
                Map<String, Object> m = new HashMap<>();
                m.put("a", "eat"); m.put("d", 0.0); m.put("val", 1.0);
                out.add(m);
            }

            // 3. Craftable recipes (vanilla + all modded recipes + shift-craft for bulk items)
            int craftHash = invCounts.hashCode() * 31 + (nearCraft ? 1 : 0);
            if (craftHash != lastCraftHash) {
                craftableById.clear();
                List<Map<String, Object>> craftCands = new ArrayList<>();
                try {
                    int dim = nearCraft ? 3 : 2;
                    for (RecipeHolder<CraftingRecipe> rh : level.getRecipeManager().getAllRecipesFor(RecipeType.CRAFTING)) {
                        CraftingRecipe rec = rh.value();
                        if (!rec.canCraftInDimensions(dim, dim)) continue;
                        int maxC = maxCraftCount(rec, invCounts, 16);
                        if (maxC <= 0) continue;
                        ItemStack res = rec.getResultItem(level.registryAccess());
                        if (res.isEmpty()) continue;
                        String rid = rh.id().toString();
                        String rname = BuiltInRegistries.ITEM.getKey(res.getItem()).toString();
                        craftableById.put(rid, rh);
                        double val = 0.2;
                        if (res.getItem() instanceof TieredItem || res.getItem() instanceof ArmorItem) val = 1.0;
                        else if (rname.contains("crafting_table") || rname.contains("furnace") || rname.contains("chest")) val = 0.95;
                        else if (rname.contains("planks") || rname.contains("stick") || rname.contains("torch") || rname.contains("bed") || rname.contains("bow") || rname.contains("arrow")) val = 0.85;
                        else if (res.get(DataComponents.FOOD) != null) val = 0.72;
                        boolean isBulk = res.getMaxStackSize() > 1 && maxC >= 2 && (
                            rname.contains("planks") || rname.contains("stick") || rname.contains("torch")
                            || rname.contains("arrow") || rname.contains("bread") || rname.contains("charcoal")
                            || rname.contains("bone_meal") || rname.contains("nugget") || rname.contains("ingot")
                        );
                        Map<String, Object> m = new HashMap<>(10);
                        m.put("a", "craft"); m.put("rid", rid); m.put("n", rname);
                        m.put("d", 0.0); m.put("val", isBulk ? Math.min(1.0, val + 0.08) : val);
                        if (isBulk) {
                            m.put("shift", 1);
                            m.put("cnt", maxC);
                        }
                        craftCands.add(m);
                    }
                    craftCands.sort((a, b) -> Double.compare((double) b.get("val"), (double) a.get("val")));
                    if (craftCands.size() > 6) {
                        craftCands = new ArrayList<>(craftCands.subList(0, 6));
                    }
                } catch (Throwable ignored) {}
                cachedCraftCands = craftCands;
                lastCraftHash = craftHash;
            }
            out.addAll(cachedCraftCands);

            // 4. Mine & Use nearby blocks (prioritize ores/logs/stone over plain dirt)
            List<Map<String, Object>> mine = new ArrayList<>(32);
            List<Map<String, Object>> useList = new ArrayList<>(3);
            BlockPos.MutableBlockPos mp = scanPos;
            for (int dx = -4; dx <= 4; dx++) {
                int px = bx + dx;
                for (int dy = -2; dy <= 4; dy++) {
                    int py = by + dy;
                    for (int dz = -4; dz <= 4; dz++) {
                        int d2 = dx * dx + dy * dy + dz * dz;
                        if (d2 > REACH * REACH) continue;
                        mp.set(px, py, bz + dz);
                        BlockState st = bs(mp);
                        if (st == null || st.isAir() || !st.getFluidState().isEmpty()) continue;
                        if (st.getDestroySpeed(level, mp) < 0) continue;
                        double d = Math.sqrt(d2);
                        String bname = BuiltInRegistries.BLOCK.getKey(st.getBlock()).toString();
                        double bval = 0.0;
                        if (bname.contains("ore") || bname.contains("diamond") || bname.contains("ancient")) bval = 1.0;
                        else if (bname.contains("log") || bname.contains("wood") || bname.contains("coal")) bval = 0.8;
                        else if (bname.contains("stone") || bname.contains("cobble") || bname.contains("iron")) bval = 0.4;
                        Map<String, Object> m = new HashMap<>(8);
                        m.put("a", "mine"); m.put("x", mp.getX()); m.put("y", mp.getY()); m.put("z", mp.getZ());
                        m.put("n", bname); m.put("d", round(d)); m.put("val", bval);
                        mine.add(m);
                        if (useList.size() < 3 && isInteractable(st, mp)) {
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

            // 5. Smart Building / Place candidates (utility, torch, shelter wall/roof, bridge, pillar)
            int bslot = selectBlockSlot();
            if (bslot >= 0) {
                String bItemName = BuiltInRegistries.ITEM.getKey(player.getInventory().getItem(bslot).getItem()).toString();
                boolean isUtil = bItemName.contains("crafting_table") || bItemName.contains("furnace") || bItemName.contains("chest") || bItemName.contains("bed");
                boolean isTorch = bItemName.contains("torch");
                boolean needShelter = level.isNight() || nearestHostile != null || player.getHealth() < 12.0f;
                int lightLv = level.getMaxLocalRawBrightness(player.blockPosition());
                double placeVal = isUtil ? 0.95 : (isTorch ? (lightLv < 8 ? 0.88 : 0.35) : (needShelter ? 0.80 : 0.40));
                int placedCands = 0;

                // Overhead shelter roof block when exposed to sky at night or under threat
                if (needShelter && !isUtil && !isTorch && level.canSeeSky(player.blockPosition())) {
                    BlockPos roof = new BlockPos(bx, by + 2, bz);
                    BlockState rSt = bs(roof);
                    if (rSt != null && rSt.isAir()) {
                        for (Direction d4 : new Direction[]{Direction.NORTH, Direction.SOUTH, Direction.EAST, Direction.WEST}) {
                            BlockPos nb = roof.relative(d4);
                            BlockState nbs = bs(nb);
                            if (nbs != null && nbs.isFaceSturdy(level, nb, d4.getOpposite())) {
                                Map<String, Object> m = new HashMap<>(8);
                                m.put("a", "place"); m.put("x", roof.getX()); m.put("y", roof.getY()); m.put("z", roof.getZ());
                                m.put("n", bItemName); m.put("d", 2.0); m.put("val", 0.86);
                                out.add(m);
                                placedCands++;
                                break;
                            }
                        }
                    }
                }

                for (int i = 0; i < 8 && placedCands < 4; i++) {
                    int px = (int) Math.floor(player.getX() + SIN8[i] * 2.0);
                    int pz = (int) Math.floor(player.getZ() + COS8[i] * 2.0);
                    // Ground / wall placement at feet or head height
                    for (int dy = 0; dy <= 1; dy++) {
                        BlockPos tp = new BlockPos(px, by + dy, pz);
                        BlockState tgt = bs(tp);
                        if (tgt == null || !tgt.isAir()) continue;
                        BlockState ref = bs(tp.below());
                        if (ref != null && ref.isFaceSturdy(level, tp.below(), Direction.UP)) {
                            Map<String, Object> m = new HashMap<>(8);
                            m.put("a", "place"); m.put("x", px); m.put("y", by + dy); m.put("z", pz);
                            m.put("n", bItemName); m.put("d", 2.0); m.put("val", placeVal);
                            out.add(m);
                            placedCands++;
                            break;
                        }
                    }
                    // Bridge placement over air gap
                    if (placedCands < 4) {
                        BlockPos bp = new BlockPos(px, by - 1, pz);
                        BlockState bt = bs(bp);
                        if (bt != null && bt.isAir()) {
                            BlockPos bRef = new BlockPos(bx, by - 1, bz);
                            BlockState br = bs(bRef);
                            if (br != null && br.isFaceSturdy(level, bRef, Direction.UP)) {
                                Map<String, Object> m = new HashMap<>(8);
                                m.put("a", "place"); m.put("x", px); m.put("y", by - 1); m.put("z", pz);
                                m.put("n", bItemName); m.put("d", 2.2); m.put("val", 0.72);
                                out.add(m);
                                placedCands++;
                            }
                        }
                    }
                }

                // Tactical pillar when hostile melee mob is close
                if (player.onGround() && !isUtil && !isTorch && nearestHostile != null && nearestHostileDist < 4.5) {
                    Map<String, Object> pm = new HashMap<>(6);
                    pm.put("a", "pillar"); pm.put("n", bItemName); pm.put("d", 0.0); pm.put("val", 0.82);
                    out.add(pm);
                }
            }

            // 6. Movement directions (8 angles x 2 distances)
            for (int i = 0; i < 8; i++) {
                double sx = SIN8[i], sz = COS8[i];
                for (double dist : GOTO_DISTS) {
                    int gx = (int) Math.floor(player.getX() + sx * dist);
                    int gz = (int) Math.floor(player.getZ() + sz * dist);
                    int gy = standY(gx, by, gz);
                    if (gy == Integer.MIN_VALUE) continue;
                    Map<String, Object> m = new HashMap<>(8);
                    m.put("a", "goto"); m.put("x", gx); m.put("y", gy); m.put("z", gz); m.put("d", dist);
                    out.add(m);
                }
            }

            // 7. Utility actions
            if (bslot >= 0 && player.onGround()) {
                out.add(Map.of("a", "pillar", "d", 0.0, "val", 0.35));
            }
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

        int findShareSlot() {
            Inventory inv = player.getInventory();
            for (int i = 0; i < inv.getContainerSize(); i++) {
                ItemStack st = inv.getItem(i);
                if (!st.isEmpty() && st.getCount() > 2 && st.get(DataComponents.FOOD) != null) return i;
            }
            for (int i = 0; i < inv.getContainerSize(); i++) {
                ItemStack st = inv.getItem(i);
                if (!st.isEmpty() && st.getCount() > 4 && !(st.getItem() instanceof TieredItem)
                        && !(st.getItem() instanceof ArmorItem)) {
                    return i;
                }
            }
            return -1;
        }

        int findItemSlot(Class<?> cls) {
            Inventory inv = player.getInventory();
            for (int i = 0; i < inv.getContainerSize(); i++) {
                ItemStack st = inv.getItem(i);
                if (!st.isEmpty() && cls.isInstance(st.getItem())) return i;
            }
            return -1;
        }

        boolean isSmeltable(ItemStack st) {
            if (st == null || st.isEmpty()) return false;
            if (cachedSmeltables == null) {
                Set<Item> set = new HashSet<>();
                try {
                    for (var rh : level.getRecipeManager().getAllRecipesFor(RecipeType.SMELTING)) {
                        for (Ingredient ing : rh.value().getIngredients()) {
                            for (ItemStack cand : ing.getItems()) {
                                if (!cand.isEmpty()) set.add(cand.getItem());
                            }
                        }
                    }
                } catch (Throwable ignored) {}
                cachedSmeltables = set;
            }
            return cachedSmeltables.contains(st.getItem());
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
