"""Frame encoder: wire frame (JSON from the bot) -> flat float32 vector.

Design rules (from idea.txt section 5):
  * everything is relative to the player and rotated into the player's frame,
    so the representation is translation- and rotation-invariant;
  * categorical ids go through a *fixed* hashed embedding (no learned table,
    so the encoder never has to re-learn an id space that mods keep growing);
  * every feature is normalised to roughly [-1, 1].

The exact layout is frozen: config.FRAME_DIM and the offsets below must match.
"""

import numpy as np

import config as C

# ------------------------------------------------------- deterministic maps --
_RS = np.random.RandomState(20261113)
BLOCK_EMB = _RS.normal(0.0, 0.6, (4096, 3)).astype(np.float32)
ITEM_EMB = _RS.normal(0.0, 0.6, (2048, 2)).astype(np.float32)
ENT_EMB = _RS.normal(0.0, 0.6, (1024, 3)).astype(np.float32)

_EV_IDX = {
    'dig': 0, 'hurt': 1, 'death': 2, 'attack': 3, 'use': 4,
    'place': 5, 'eat': 6, 'act_fail': 7, 'dig_fail': 8, 'bad_action': 9,
    'pickup': 10, 'craft': 11, 'kill': 12, 'info': 13, 'info_end': 14,
}

# item id -> 0..2047 bucket
def _ib(t):
    return ITEM_EMB[int(t) % 2048]


def block_vec(bid):
    return BLOCK_EMB[int(bid) % 4096]


def ent_vec(name):
    h = (hash(name) & 0x7FFFFFFF) % 1024 if not isinstance(name, int) else int(name) % 1024
    return ENT_EMB[h]


def encode(frame: dict) -> np.ndarray:
    """One frame -> float32 vector of length C.FRAME_DIM."""
    v = np.zeros(C.FRAME_DIM, dtype=np.float32)
    o = 0

    s = frame.get('self', {})
    yaw = float(s.get('yaw', 0.0))
    sy, cy = np.sin(yaw), np.cos(yaw)
    vx, vy, vz = float(s.get('vx', 0.0)), float(s.get('vy', 0.0)), float(s.get('vz', 0.0))

    # ---- self (32) ----
    v[o + 0] = np.clip((vx * sy + vz * cy) / 8.0, -1, 1)     # forward speed
    v[o + 1] = np.clip((vx * cy - vz * sy) / 8.0, -1, 1)     # right speed
    v[o + 2] = np.clip(vy / 8.0, -1, 1)
    v[o + 3] = sy
    v[o + 4] = cy
    pitch = float(s.get('pitch', 0.0))
    v[o + 5] = np.sin(pitch)
    v[o + 6] = np.cos(pitch)
    v[o + 7] = float(s.get('hp', 20)) / 20.0
    v[o + 8] = float(s.get('food', 20)) / 20.0
    v[o + 9] = np.clip(float(s.get('air', 20)) / 20.0, 0, 1)
    v[o + 10] = np.clip(float(s.get('sat', 0)) / 20.0, 0, 1)
    v[o + 11] = np.clip(float(s.get('light', 0)) / 15.0, 0, 1)
    v[o + 12] = 1.0 if s.get('on_ground') else 0.0
    v[o + 13] = 1.0 if s.get('in_water') else 0.0
    v[o + 14] = 1.0 if s.get('in_lava') else 0.0
    v[o + 15] = 1.0 if s.get('sprinting') else 0.0
    v[o + 16] = 1.0 if s.get('sneaking') else 0.0
    v[o + 17] = np.clip(float(s.get('fall', 0)) / 10.0, 0, 1)
    v[o + 18] = 1.0 if s.get('collided') else 0.0
    tday = float(s.get('time', 0)) / 24000.0 * 2 * np.pi
    v[o + 19] = np.sin(tday)
    v[o + 20] = np.cos(tday)
    v[o + 21] = 1.0 if s.get('raining') else 0.0
    v[o + 22] = 1.0 if s.get('thunder') else 0.0
    dim = s.get('dim', 'minecraft:overworld')
    v[o + 23] = 1.0 if dim == 'minecraft:overworld' else 0.0
    v[o + 24] = 1.0 if dim == 'minecraft:the_nether' else 0.0
    v[o + 25] = 1.0 if dim == 'minecraft:the_end' else 0.0
    v[o + 26] = np.clip(float(s.get('hp_d', 0)) / 5.0, -1, 1)
    v[o + 27] = np.clip(float(s.get('food_d', 0)) / 5.0, -1, 1)
    v[o + 28] = 0.0
    v[o + 29] = np.clip((vx * vx + vy * vy + vz * vz) ** 0.5 / 8.0, 0, 1)
    v[o + 30] = 0.0
    v[o + 31] = 0.0
    o += C.SELF_DIM

    # ---- voxel window (512 x 6) ----
    g = frame.get('grid', {})
    ids = g.get('ids') or [0] * C.GRID_CELLS
    ox, oy, oz = g.get('ox', 0), g.get('oy', 0), g.get('oz', 0)
    bx, by, bz = int(s.get('x', 0)), int(s.get('y', 0)), int(s.get('z', 0))
    n = C.GRID_N
    px = (bx - ox) // C.GRID_STEP
    py = (by - oy) // C.GRID_STEP
    pz = (bz - oz) // C.GRID_STEP
    for ix in range(n):
        for iy in range(n):
            for iz in range(n):
                bid = int(ids[ix * n * n + iy * n + iz])
                base = o + (ix * n * n + iy * n + iz) * 6
                if bid:
                    e = BLOCK_EMB[bid % 4096]
                    v[base] = e[0]
                    v[base + 1] = e[1]
                    v[base + 2] = e[2]
                    v[base + 3] = 1.0
                # liquid / interactable flags are inferred on the bot side later;
                # left at 0 for now (schema v1)
    o += C.GRID_DIM

    # ---- entities (12 x 10) ----
    ents = frame.get('ent') or []
    for i, e in enumerate(ents[:C.ENT_MAX]):
        base = o + i * C.ENT_DIM
        nm = str(e[1]) if len(e) > 1 else 'unknown'
        ev = ent_vec(nm)
        # rotate into the player frame so orientation does not matter
        dx, dz = float(e[2]), float(e[4])
        v[base + 0] = ev[0]
        v[base + 1] = ev[1]
        v[base + 2] = ev[2]
        v[base + 3] = np.clip((dx * cy - dz * sy) / 8.0, -1, 1)   # right
        v[base + 4] = np.clip(float(e[3]) / 8.0, -1, 1)           # up
        v[base + 5] = np.clip((dx * sy + dz * cy) / 8.0, -1, 1)   # forward
        v[base + 6] = np.clip(max(float(e[5]), 0.0) / 16.0, 0, 1)           # distance
        v[base + 7] = np.clip(float(e[6]) / 20.0, 0, 1) if len(e) > 6 and e[6] >= 0 else 0.0
        v[base + 8] = float(e[7]) if len(e) > 7 else 0.0          # hostility class
        v[base + 9] = 1.0 / (1.0 + max(float(e[5]), 0.0))
    o += C.ENT_TOTAL

    # ---- inventory (41 x 4) ----
    inv = frame.get('inv') or []
    present = np.zeros(C.INV_SLOTS, dtype=np.float32)
    for it in inv:
        slot = int(it[0])
        if slot < 0 or slot >= C.INV_SLOTS:
            continue
        present[slot] = 1.0
        base = o + slot * C.INV_DIM
        e = _ib(it[1])
        v[base] = e[0]
        v[base + 1] = e[1]
        v[base + 2] = np.clip(float(it[2]) / 64.0, 0, 1)
        v[base + 3] = float(it[3]) if len(it) > 3 else 0.0
    # occupancy flags are appended after the slot block by using the last slot's
    # spare dims; instead we encode "empty" implicitly (all-zero slot = empty).
    o += C.INV_TOTAL

    # ---- events (10 x 5) ----
    evs = frame.get('ev') or []
    for i, e in enumerate(evs[:C.EV_MAX]):
        base = o + i * C.EV_DIM
        v[base] = _EV_IDX.get(str(e.get('e') if isinstance(e, dict) else e[0]), 15) / 16.0
        mag = float(e.get('m', 0) if isinstance(e, dict) else (e[1] if len(e) > 1 else 0))
        v[base + 1] = np.clip(mag, -4, 4) / 4.0
        if isinstance(e, dict) and 'x' in e:
            v[base + 2] = np.clip((e['x'] - bx) / 8.0, -1, 1)
            v[base + 3] = np.clip((e['y'] - by) / 8.0, -1, 1)
            v[base + 4] = np.clip((e['z'] - bz) / 8.0, -1, 1)
    o += C.EV_TOTAL

    # ---- meta (12) ----
    prev = frame.get('prev') or {}
    v[o + 0] = _ACT_IDX.get(str(prev.get('a', 'none')), 15) / 16.0
    v[o + 1] = np.clip(float(prev.get('ticks', 0)) / 40.0, 0, 1)
    v[o + 2] = 0.5 if prev.get('ok') is None else (1.0 if prev.get('ok') else 0.0)
    v[o + 3] = np.clip(float(frame.get('tick', 0)) / 4000.0, 0, 1)
    v[o + 4] = len(frame.get('cand') or []) / float(C.MAX_CAND)
    v[o + 5] = 0.0     # last reward   (filled by the brain)
    v[o + 6] = 0.0     # last discounted return
    v[o + 7] = np.clip(float(frame.get('seq', 0)) / 20000.0, 0, 1)
    v[o + 8] = np.clip(abs(bx) / 5000.0, 0, 1)
    v[o + 9] = np.clip(abs(bz) / 5000.0, 0, 1)
    v[o + 10] = np.clip(by / 320.0, 0, 1)
    v[o + 11] = 0.0
    o += C.META_DIM

    assert o == C.FRAME_DIM, (o, C.FRAME_DIM)
    return v


_ACT_IDX = {
    'mine': 0, 'goto': 1, 'attack': 2, 'use': 3, 'place': 4,
    'equip': 5, 'eat': 6, 'drop': 7, 'jump': 8, 'wait': 9, 'stop': 10, 'none': 15,
}


def candidate_features(cand: dict, frame: dict) -> np.ndarray:
    """One candidate action -> fixed-length feature vector (C.CAND_DIM)."""
    f = np.zeros(C.CAND_DIM, dtype=np.float32)
    a = str(cand.get('a', 'wait'))
    f[0] = _ACT_IDX.get(a, 12) / 16.0
    f[1] = np.clip(float(cand.get('d', 0)) / 8.0, 0, 1)
    if a in ('mine', 'goto', 'place', 'use'):
        s = frame.get('self', {})
        dx = float(cand.get('x', 0)) - float(s.get('x', 0))
        dy = float(cand.get('y', 0)) - float(s.get('y', 0))
        dz = float(cand.get('z', 0)) - float(s.get('z', 0))
        yaw = float(s.get('yaw', 0.0))
        cy, sy = np.cos(yaw), np.sin(yaw)
        f[2] = np.clip((dx * cy - dz * sy) / 8.0, -1, 1)
        f[3] = np.clip(dy / 8.0, -1, 1)
        f[4] = np.clip((dx * sy + dz * cy) / 8.0, -1, 1)
    f[5] = 1.0 if a == 'attack' else 0.0
    f[6] = float(cand.get('h', 0))
    f[7] = 1.0 if a == 'eat' else 0.0
    f[8] = 1.0 if a == 'goto' else 0.0
    f[9] = 1.0 if a == 'mine' else 0.0
    f[10] = 1.0 if a == 'use' else 0.0
    f[11] = 1.0 if a == 'place' else 0.0
    f[12] = 1.0 if a in ('wait', 'stop') else 0.0
    f[13] = 1.0 if a == 'jump' else 0.0
    f[14] = np.clip(float(cand.get('d', 0)) / 4.0, 0, 1)
    f[15] = 1.0 / (1.0 + float(cand.get('d', 0)))
    return f