"""Frame encoder: wire frame (JSON from mod/bot) -> flat float32 vector.

v2 upgrades:
  * deterministic CRC32 hashing for string IDs (consistent across restarts);
  * proper degrees -> radians conversion for yaw/pitch;
  * vectorized NumPy encoding for the 512-cell dense voxel grid (~10x faster);
  * fixed entity schema + velocity/role/targeting features;
  * rich inventory features (food, tool, block, armor, durability);
  * episodic memory / POI context in self[32:48];
  * rich 24-dim candidate features for all survival/crafting/social actions.
"""

import math
import zlib
from functools import lru_cache
import numpy as np

import config as C

# ------------------------------------------------------- deterministic maps --
_RS = np.random.RandomState(20261113)
BLOCK_EMB = _RS.normal(0.0, 0.6, (4096, 3)).astype(np.float32)
BLOCK_EMB[0] = 0.0
ITEM_EMB = _RS.normal(0.0, 0.6, (2048, 2)).astype(np.float32)
ITEM_EMB[0] = 0.0
ENT_EMB = _RS.normal(0.0, 0.6, (1024, 3)).astype(np.float32)

_EV_IDX = {
    'dig': 0, 'hurt': 1, 'death': 2, 'attack': 3, 'use': 4,
    'place': 5, 'eat': 6, 'act_fail': 7, 'dig_fail': 8, 'bad_action': 9,
    'pickup': 10, 'craft': 11, 'kill': 12, 'equip': 13, 'fall': 14,
    'burn': 15, 'starve': 16, 'drown': 17, 'interact': 18, 'info': 19,
}

_ACT_IDX = {
    'mine': 0, 'goto': 1, 'attack': 2, 'use': 3, 'place': 4,
    'craft': 5, 'equip': 6, 'eat': 7, 'interact_entity': 8, 'give': 8,
    'drop': 9, 'jump': 10, 'sneak': 11, 'wait': 12, 'stop': 13,
    'shoot': 14, 'pillar': 15, 'none': 16,
}

_TWO_PI = 2.0 * math.pi


def _clip(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else (hi if x > hi else x)


@lru_cache(maxsize=4096)
def _crc(s: str) -> int:
    return zlib.crc32(s.encode('utf-8')) & 0x7FFFFFFF


@lru_cache(maxsize=4096)
def _ib(t):
    if isinstance(t, (int, np.integer)):
        return ITEM_EMB[int(t) % 2048]
    return ITEM_EMB[_crc(str(t)) % 2048]


@lru_cache(maxsize=4096)
def block_vec(bid):
    if isinstance(bid, (int, np.integer)):
        return BLOCK_EMB[int(bid) % 4096]
    return BLOCK_EMB[_crc(str(bid)) % 4096]


@lru_cache(maxsize=4096)
def ent_vec(name):
    if isinstance(name, (int, np.integer)):
        return ENT_EMB[int(name) % 1024]
    return ENT_EMB[_crc(str(name)) % 1024]


def encode(frame: dict, mem_ctx: dict = None, last_rew: float = 0.0, last_ret: float = 0.0) -> np.ndarray:
    """One frame -> float32 vector of length C.FRAME_DIM."""
    v = np.zeros(C.FRAME_DIM, dtype=np.float32)
    o = 0

    s = frame.get('self', {})
    yaw = math.radians(float(s.get('yaw', 0.0)))
    pitch = math.radians(float(s.get('pitch', 0.0)))
    sy, cy = math.sin(yaw), math.cos(yaw)
    vx, vy, vz = float(s.get('vx', 0.0)), float(s.get('vy', 0.0)), float(s.get('vz', 0.0))
    bx, by, bz = float(s.get('x', 0.0)), float(s.get('y', 0.0)), float(s.get('z', 0.0))

    # ---- self (64) ----
    # Note: MC yaw=0 faces +Z, yaw=90 faces -X: forward = (-sin(yaw), cos(yaw)), right = (-cos(yaw), -sin(yaw))
    fwd_x, fwd_z = -sy, cy
    rgt_x, rgt_z = -cy, -sy
    v[o + 0] = _clip((vx * fwd_x + vz * fwd_z) / 1.5, -1.0, 1.0)
    v[o + 1] = _clip((vx * rgt_x + vz * rgt_z) / 1.5, -1.0, 1.0)
    v[o + 2] = _clip(vy / 1.5, -1.0, 1.0)
    v[o + 3] = sy
    v[o + 4] = cy
    v[o + 5] = math.sin(pitch)
    v[o + 6] = math.cos(pitch)
    v[o + 7] = _clip(float(s.get('hp', 20.0)) / 20.0, 0.0, 2.0)
    v[o + 8] = _clip(float(s.get('food', 20.0)) / 20.0, 0.0, 1.0)
    v[o + 9] = _clip(float(s.get('air', 300.0)) / 300.0, 0.0, 1.0)
    v[o + 10] = _clip(float(s.get('sat', 0.0)) / 20.0, 0.0, 1.0)
    v[o + 11] = _clip(float(s.get('light', 0.0)) / 15.0, 0.0, 1.0)
    v[o + 12] = 1.0 if s.get('on_ground') else 0.0
    v[o + 13] = 1.0 if s.get('in_water') else 0.0
    v[o + 14] = 1.0 if s.get('in_lava') else 0.0
    v[o + 15] = 1.0 if s.get('on_fire') else 0.0
    v[o + 16] = 1.0 if s.get('sprinting') else 0.0
    v[o + 17] = 1.0 if s.get('sneaking') else 0.0
    v[o + 18] = _clip(float(s.get('fall', 0.0)) / 10.0, 0.0, 1.0)
    v[o + 19] = 1.0 if s.get('collided') else 0.0
    tday = (float(s.get('time', 0.0)) % 24000.0) / 24000.0 * _TWO_PI
    v[o + 20] = math.sin(tday)
    v[o + 21] = math.cos(tday)
    v[o + 22] = 1.0 if s.get('raining') else 0.0
    v[o + 23] = 1.0 if s.get('thunder') else 0.0
    dim = s.get('dim', 'minecraft:overworld')
    v[o + 24] = 1.0 if dim == 'minecraft:overworld' else 0.0
    v[o + 25] = 1.0 if dim == 'minecraft:the_nether' else 0.0
    v[o + 26] = 1.0 if dim == 'minecraft:the_end' else 0.0
    v[o + 27] = _clip(float(s.get('hp_d', 0.0)) / 5.0, -1.0, 1.0)
    v[o + 28] = _clip(float(s.get('food_d', 0.0)) / 5.0, -1.0, 1.0)
    v[o + 29] = _clip(math.sqrt(vx * vx + vy * vy + vz * vz) / 1.5, 0.0, 1.0)
    v[o + 30] = _clip(float(s.get('armor', 0.0)) / 20.0, 0.0, 1.0)
    v[o + 31] = _clip(float(s.get('atk_cd', 1.0)), 0.0, 1.0)
    v[o + 32] = 1.0 if s.get('near_craft') else 0.0
    v[o + 33] = 1.0 if s.get('near_furnace') else 0.0
    v[o + 34] = _clip(float(s.get('sel_slot', 0)) / 8.0, 0.0, 1.0)
    v[o + 35] = 1.0 if s.get('has_food') else 0.0
    v[o + 36] = 1.0 if s.get('has_tool') else 0.0
    v[o + 37] = 1.0 if s.get('has_weapon') else 0.0
    v[o + 38] = 1.0 if s.get('has_blocks') else 0.0
    v[o + 39] = 1.0 if s.get('is_night') else 0.0

    # Memory / POI context (slots 40..47): relative directions to remembered POIs
    if mem_ctx:
        v[o + 40] = float(mem_ctx.get('home_r', 0.0))
        v[o + 41] = float(mem_ctx.get('home_f', 0.0))
        v[o + 42] = float(mem_ctx.get('home_d', 0.0))
        v[o + 43] = float(mem_ctx.get('craft_r', 0.0))
        v[o + 44] = float(mem_ctx.get('craft_f', 0.0))
        v[o + 45] = float(mem_ctx.get('player_r', 0.0))
        v[o + 46] = float(mem_ctx.get('player_f', 0.0))
        v[o + 47] = float(mem_ctx.get('danger_mem', 0.0))
    # Rich survival & environmental awareness (slots 48..63)
    v[o + 48] = 1.0 if s.get('sky') else 0.0
    v[o + 49] = _clip(float(s.get('shelter', 0.0)), 0.0, 1.0)
    v[o + 50] = 1.0 if s.get('debuff') else 0.0
    v[o + 51] = 1.0 if s.get('buff_prot') else 0.0
    v[o + 52] = 1.0 if s.get('buff_atk') else 0.0
    v[o + 53] = 1.0 if s.get('climb') else 0.0
    v[o + 54] = 1.0 if s.get('has_bow') else 0.0
    v[o + 55] = 1.0 if s.get('has_shield') else 0.0
    v[o + 56] = _clip(float(s.get('lava_near', 0.0)), 0.0, 1.0)
    v[o + 57] = 1.0 if s.get('cliff_ahead') else 0.0
    v[o + 58] = _clip(float(s.get('inv_free', 1.0)), 0.0, 1.0)
    v[o + 59] = _clip(float(s.get('xp', 0.0)) / 30.0, 0.0, 1.0)
    v[o + 60] = 1.0 if s.get('ray_ent') else 0.0
    v[o + 61] = 1.0 if s.get('ray_blk') else 0.0
    v[o + 62] = 1.0 if s.get('chat_ping') else 0.0
    v[o + 63] = _clip(float(s.get('near_trust', 0.0)), -1.0, 1.0)
    o += C.SELF_DIM

    # ---- vectorized voxel window (512 x 6) ----
    g = frame.get('grid', {})
    ids = g.get('ids')
    flags = g.get('flags')
    grid_view = v[o:o + C.GRID_DIM].reshape(C.GRID_CELLS, 6)
    if ids is not None and len(ids) == C.GRID_CELLS:
        ids_arr = np.asarray(ids, dtype=np.int32)
        nz = ids_arr != 0
        if np.any(nz):
            grid_view[nz, :3] = BLOCK_EMB[ids_arr[nz] & 4095]
            grid_view[nz, 3] = 1.0
        if flags is not None and len(flags) == C.GRID_CELLS:
            fl_arr = np.asarray(flags, dtype=np.int32)
            grid_view[:, 3] = fl_arr & 1
            grid_view[:, 4] = (fl_arr >> 1) & 1
            grid_view[:, 5] = (fl_arr >> 2) & 1
    o += C.GRID_DIM

    # ---- entities (16 x 16) ----
    ents = frame.get('ent') or []
    for i, e in enumerate(ents[:C.ENT_MAX]):
        base = o + i * C.ENT_DIM
        if len(e) >= 10:
            # v2 schema: [id, name, dx, dy, dz, dist, hp, max_hp, role, targeting_me, vx, vz]
            nm = e[1]
            dx, dy, dz = float(e[2]), float(e[3]), float(e[4])
            dist = max(float(e[5]), 0.0)
            hp = float(e[6])
            max_hp = max(float(e[7]), 1.0)
            role = int(e[8])
            tgt_me = float(e[9])
            evx = float(e[10]) if len(e) > 10 else 0.0
            evz = float(e[11]) if len(e) > 11 else 0.0
        else:
            # legacy 7-field fallback: [id, dx, dy, dz, dist, hp, hostile]
            nm = e[0]
            dx, dy, dz = float(e[1]), float(e[2]), float(e[3])
            dist = max(float(e[4]), 0.0)
            hp = float(e[5])
            max_hp = 20.0
            role = 1 if (len(e) > 6 and e[6]) else 0
            tgt_me = 0.0
            evx, evz = 0.0, 0.0

        ev = ent_vec(nm)
        v[base + 0] = ev[0]
        v[base + 1] = ev[1]
        v[base + 2] = ev[2]
        v[base + 3] = _clip((dx * rgt_x + dz * rgt_z) / 16.0, -1.0, 1.0)
        v[base + 4] = _clip(dy / 8.0, -1.0, 1.0)
        v[base + 5] = _clip((dx * fwd_x + dz * fwd_z) / 16.0, -1.0, 1.0)
        v[base + 6] = _clip(dist / 24.0, 0.0, 1.0)
        v[base + 7] = 1.0 / (1.0 + dist)
        v[base + 8] = _clip(hp / max_hp, 0.0, 1.0) if hp >= 0 else 0.0
        v[base + 9] = 1.0 if role == 1 else 0.0   # hostile
        v[base + 10] = 1.0 if role == 2 else 0.0  # player
        v[base + 11] = 1.0 if role == 3 else 0.0  # item drop
        v[base + 12] = 1.0 if role in (0, 4) else 0.0  # passive / projectile
        v[base + 13] = tgt_me
        v[base + 14] = _clip((evx * rgt_x + evz * rgt_z) / 1.0, -1.0, 1.0)
        v[base + 15] = _clip((evx * fwd_x + evz * fwd_z) / 1.0, -1.0, 1.0)
    o += C.ENT_TOTAL

    # ---- inventory (41 x 8) ----
    inv = frame.get('inv') or []
    for it in inv:
        slot = int(it[0])
        if slot < 0 or slot >= C.INV_SLOTS:
            continue
        base = o + slot * C.INV_DIM
        em = _ib(it[1])
        v[base + 0] = em[0]
        v[base + 1] = em[1]
        v[base + 2] = _clip(float(it[2]) / 64.0, 0.0, 1.0)
        v[base + 3] = float(it[3]) if len(it) > 3 else 0.0   # is_food
        v[base + 4] = float(it[4]) if len(it) > 4 else 0.0   # is_tool/weapon
        v[base + 5] = float(it[5]) if len(it) > 5 else 0.0   # is_block
        v[base + 6] = float(it[6]) if len(it) > 6 else 0.0   # is_armor
        v[base + 7] = float(it[7]) if len(it) > 7 else 1.0   # durability frac
    o += C.INV_TOTAL

    # ---- events (10 x 6) ----
    evs = frame.get('ev') or []
    for i, e in enumerate(evs[:C.EV_MAX]):
        base = o + i * C.EV_DIM
        ename = str(e.get('e') if isinstance(e, dict) else e[0])
        v[base + 0] = _EV_IDX.get(ename, 19) / 20.0
        mag = float(e.get('m', 0) if isinstance(e, dict) else (e[1] if len(e) > 1 else 0))
        v[base + 1] = _clip(mag, -10.0, 10.0) / 10.0
        if isinstance(e, dict) and 'x' in e:
            v[base + 2] = _clip((float(e['x']) - bx) / 8.0, -1.0, 1.0)
            v[base + 3] = _clip((float(e['y']) - by) / 8.0, -1.0, 1.0)
            v[base + 4] = _clip((float(e['z']) - bz) / 8.0, -1.0, 1.0)
        v[base + 5] = 1.0
    o += C.EV_TOTAL

    # ---- meta (16) ----
    prev = frame.get('prev') or {}
    v[o + 0] = _ACT_IDX.get(str(prev.get('a', 'none')), 15) / 16.0
    v[o + 1] = _clip(float(prev.get('ticks', 0)) / 40.0, 0.0, 1.0)
    v[o + 2] = 0.5 if prev.get('ok') is None else (1.0 if prev.get('ok') else 0.0)
    v[o + 3] = _clip(float(frame.get('tick', 0)) / 4000.0, 0.0, 1.0)
    v[o + 4] = len(frame.get('cand') or []) / float(C.MAX_CAND)
    v[o + 5] = _clip(last_rew / 3.0, -1.0, 1.0)
    v[o + 6] = _clip(last_ret / 10.0, -1.0, 1.0)
    v[o + 7] = _clip(float(frame.get('seq', 0)) / 20000.0, 0.0, 1.0)
    v[o + 8] = _clip(bx / 5000.0, -1.0, 1.0)
    v[o + 9] = _clip(bz / 5000.0, -1.0, 1.0)
    v[o + 10] = _clip(by / 320.0, -1.0, 1.0)
    v[o + 11] = 1.0 if len(evs) > 0 else 0.0
    o += C.META_DIM

    assert o == C.FRAME_DIM, (o, C.FRAME_DIM)
    return v


def candidate_features(cand: dict, frame: dict) -> np.ndarray:
    """One candidate action -> fixed-length feature vector (C.CAND_DIM = 24)."""
    f = np.zeros(C.CAND_DIM, dtype=np.float32)
    a = str(cand.get('a', 'wait'))
    idx = _ACT_IDX.get(a, 14)
    f[0] = idx / 16.0
    d = max(float(cand.get('d', 0.0)), 0.0)
    f[1] = _clip(d / 16.0, 0.0, 1.0)
    f[2] = 1.0 / (1.0 + d)

    if 'x' in cand and 'z' in cand:
        s = frame.get('self', {})
        dx = float(cand.get('x', 0)) - float(s.get('x', 0))
        dy = float(cand.get('y', 0)) - float(s.get('y', 0))
        dz = float(cand.get('z', 0)) - float(s.get('z', 0))
        yaw = math.radians(float(s.get('yaw', 0.0)))
        cy, sy = math.cos(yaw), math.sin(yaw)
        fwd_x, fwd_z = -sy, cy
        rgt_x, rgt_z = -cy, -sy
        f[3] = _clip((dx * rgt_x + dz * rgt_z) / 16.0, -1.0, 1.0)
        f[4] = _clip(dy / 8.0, -1.0, 1.0)
        f[5] = _clip((dx * fwd_x + dz * fwd_z) / 16.0, -1.0, 1.0)

    # Action type one-hot (slots 6..16)
    if a == 'mine': f[6] = 1.0
    elif a == 'goto': f[7] = 1.0
    elif a == 'attack': f[8] = 1.0
    elif a == 'shoot': f[8] = 1.0; f[16] = 1.0
    elif a == 'use': f[9] = 1.0
    elif a == 'place': f[10] = 1.0
    elif a == 'pillar': f[10] = 1.0; f[16] = 1.0
    elif a == 'craft': f[11] = 1.0
    elif a == 'equip': f[12] = 1.0
    elif a == 'eat': f[13] = 1.0
    elif a in ('interact_entity', 'give'): f[14] = 1.0
    elif a in ('wait', 'stop'): f[15] = 1.0
    elif a in ('jump', 'sneak', 'drop'): f[16] = 1.0

    # Target name embedding (slots 17..18)
    nm = cand.get('n', '')
    if nm:
        em = _ib(nm)
        f[17] = em[0]
        f[18] = em[1]

    # Semantic hints from adapter
    f[19] = float(cand.get('h', 0.0))          # hostile
    f[20] = float(cand.get('tgt', 0.0))        # targeting player
    f[21] = float(cand.get('item_drop', 0.0))  # goto item drop
    f[22] = float(cand.get('player', 0.0))     # target is player
    f[23] = _clip(float(cand.get('val', 0.0)), -1.0, 1.0)  # tool/tier/craft value hint
    return f


def encode_candidates(cands: list, frame: dict) -> tuple:
    """Fast batched candidate encoding -> (Mx (MAX_CAND, CAND_DIM), n)."""
    Mx = np.zeros((C.MAX_CAND, C.CAND_DIM), dtype=np.float32)
    if not cands:
        return Mx, 0
    n = min(len(cands), C.MAX_CAND)
    s = frame.get('self', {})
    sx = float(s.get('x', 0.0))
    sy_pos = float(s.get('y', 0.0))
    sz = float(s.get('z', 0.0))
    yaw = math.radians(float(s.get('yaw', 0.0)))
    cy, sy = math.cos(yaw), math.sin(yaw)
    fwd_x, fwd_z = -sy, cy
    rgt_x, rgt_z = -cy, -sy

    for i in range(n):
        c = cands[i]
        row = Mx[i]
        a = str(c.get('a', 'wait'))
        row[0] = _ACT_IDX.get(a, 14) / 16.0
        d = max(float(c.get('d', 0.0)), 0.0)
        row[1] = min(d / 16.0, 1.0)
        row[2] = 1.0 / (1.0 + d)

        if 'x' in c and 'z' in c:
            dx = float(c['x']) - sx
            dy = float(c.get('y', sy_pos)) - sy_pos
            dz = float(c['z']) - sz
            row[3] = max(-1.0, min(1.0, (dx * rgt_x + dz * rgt_z) / 16.0))
            row[4] = max(-1.0, min(1.0, dy / 8.0))
            row[5] = max(-1.0, min(1.0, (dx * fwd_x + dz * fwd_z) / 16.0))

        if a == 'mine': row[6] = 1.0
        elif a == 'goto': row[7] = 1.0
        elif a == 'attack': row[8] = 1.0
        elif a == 'shoot': row[8] = 1.0; row[16] = 1.0
        elif a == 'use': row[9] = 1.0
        elif a == 'place': row[10] = 1.0
        elif a == 'pillar': row[10] = 1.0; row[16] = 1.0
        elif a == 'craft': row[11] = 1.0
        elif a == 'equip': row[12] = 1.0
        elif a == 'eat': row[13] = 1.0
        elif a in ('interact_entity', 'give'): row[14] = 1.0
        elif a in ('wait', 'stop'): row[15] = 1.0
        elif a in ('jump', 'sneak', 'drop'): row[16] = 1.0

        nm = c.get('n', '')
        if nm:
            em = _ib(nm)
            row[17] = em[0]
            row[18] = em[1]

        row[19] = float(c.get('h', 0.0))
        row[20] = float(c.get('tgt', 0.0))
        row[21] = float(c.get('item_drop', 0.0))
        row[22] = float(c.get('player', 0.0))
        val = float(c.get('val', 0.0))
        row[23] = -1.0 if val < -1.0 else (1.0 if val > 1.0 else val)

    return Mx, n