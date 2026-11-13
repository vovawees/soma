"""Reward model v2 — multi-signal survival, progression, combat, curiosity & social.

Designed so the agent learns:
  1. Self-preservation (avoid lava/drowning/falls/starvation, eat when hungry, fight or flee).
  2. Resource gathering & tech progression (mining new blocks, picking up drops, crafting tools/stations).
  3. Modpack-agnostic discovery (curiosity + generic milestone tags).
  4. Combat & social awareness (fight hostiles, don't grief passive mobs/players, interact peacefully).
"""

import math

CLIP = 5.0

EVT = {
    'attack': 0.14,
    'kill': 1.8,
    'use': 0.06,
    'place': 0.03,      # placing torches/blocks/tables is useful in survival
    'pickup': 0.12,
    'equip': 0.04,
    'interact': 0.15,   # interacting with villager/player/tameable
    'fall': -0.10,
    'burn': -0.25,
    'starve': -0.25,
    'drown': -0.25,
    'jump': 0.0,
    'act_fail': -0.06,
    'dig_fail': -0.08,
    'bad_action': -0.10,
    'death': -5.0,
    'info': 0.0,
}

# Generic substring milestones (works across vanilla + modpacks)
MILESTONES = [
    ('log', 0.6), ('planks', 0.6), ('stick', 0.4),
    ('crafting_table', 1.0), ('furnace', 1.0), ('chest', 0.6),
    ('wooden_pickaxe', 0.8), ('stone_pickaxe', 1.2), ('iron_pickaxe', 1.8), ('diamond_pickaxe', 2.5),
    ('sword', 0.8), ('axe', 0.6), ('torch', 0.5), ('bed', 0.8),
    ('coal', 0.4), ('iron', 0.7), ('gold', 0.7), ('diamond', 2.0),
    ('helmet', 0.7), ('chestplate', 0.9), ('leggings', 0.8), ('boots', 0.7),
]


class Reward:
    def __init__(self, state=None):
        self.seen_blocks = set()
        self.seen_items = set()
        self.seen_crafts = set()
        self.milestones = set()
        self.chunks = {}
        self.hp = 20.0
        self.food = 20.0
        self.dead = 0
        self.kills = 0
        self.crafts = 0
        self.idle = 0
        self.steps = 0
        self.total = 0.0
        self.return_ = 0.0
        self.episode = 0
        self.last_pos = None
        self.last_time = 0.0
        if state:
            self.load(state)

    def _chunk_key(self, x, z):
        return f"{int(x) // 16},{int(z) // 16}"

    def _visit(self, x, z):
        k = self._chunk_key(x, z)
        c = self.chunks.get(k, 0) + 1
        if len(self.chunks) < 8192:
            self.chunks[k] = c
        return c

    def _check_milestone(self, name: str) -> float:
        low = name.lower()
        bonus = 0.0
        for key, val in MILESTONES:
            if key in low and key not in self.milestones:
                self.milestones.add(key)
                bonus += val
        return bonus

    def step(self, prev: dict, cur: dict, events, chosen_cand=None, curiosity: float = 0.0) -> tuple:
        r = 0.0
        info = {}

        s = cur.get('self', {})
        ps = prev.get('self', {}) if prev else s

        cur_hp = float(s.get('hp', 20.0))
        prev_hp = float(ps.get('hp', 20.0))
        cur_food = float(s.get('food', 20.0))
        prev_food = float(ps.get('food', 20.0))

        # --- health & food deltas ---
        dhp = cur_hp - prev_hp
        dfd = cur_food - prev_food
        if dhp < -0.05:
            # Losing HP at low health hurts more -> encourages fleeing/eating
            low_hp_mult = 1.5 if cur_hp < 8.0 else 1.0
            r -= 0.18 * min(-dhp, 12.0) * low_hp_mult
        elif dhp > 0.05:
            r += 0.06 * min(dhp, 4.0)

        if dfd < -0.05:
            r -= 0.015

        # --- hazard awareness ---
        if s.get('in_lava') or s.get('on_fire'):
            r -= 0.12
        if float(s.get('air', 300.0)) < 60.0:
            r -= 0.08

        # --- events ---
        for e in (events or []):
            name = e.get('e') if isinstance(e, dict) else str(e)
            mag = float(e.get('m', 0)) if isinstance(e, dict) else 0.0
            ename = str(e.get('n', '')) if isinstance(e, dict) else ''

            if name == 'dig':
                info['digs'] = info.get('digs', 0) + 1
                if ename and ename not in self.seen_blocks:
                    self.seen_blocks.add(ename)
                    r += 0.50
                    info['new_block'] = ename
                else:
                    r += 0.08
                r += self._check_milestone(ename)

            elif name == 'craft':
                self.crafts += 1
                info['craft'] = ename or 'item'
                if ename and ename not in self.seen_crafts:
                    self.seen_crafts.add(ename)
                    r += 1.20
                else:
                    r += 0.35
                r += self._check_milestone(ename)

            elif name == 'pickup':
                r += EVT['pickup'] * min(max(mag, 1.0), 4.0)
                if ename:
                    if ename not in self.seen_items:
                        self.seen_items.add(ename)
                        r += 0.60
                        info['new_item'] = ename
                    r += self._check_milestone(ename)

            elif name == 'eat':
                if prev_food < 16.0:
                    r += 0.25
                elif prev_food < 19.5:
                    r += 0.08
                else:
                    r -= 0.03

            elif name == 'attack':
                # Reward attacking hostiles; penalize attacking non-hostiles
                if chosen_cand and chosen_cand.get('a') == 'attack' and chosen_cand.get('h', 1) == 0:
                    r -= 0.20
                else:
                    r += EVT['attack']

            elif name == 'kill':
                self.kills += 1
                info['kill'] = 1
                r += EVT['kill']

            elif name == 'fall':
                r += EVT['fall'] * min(max(mag, 1.0), 15.0)

            elif name == 'death':
                self.dead += 1
                self.episode += 1
                info['death'] = 1
                r += EVT['death']

            elif name == 'hurt':
                pass  # already accounted via dhp

            else:
                r += EVT.get(name, 0.0)

        # --- inventory delta fallback (catches items from chests/crafting/smelting) ---
        inv_now = {}
        for it in (cur.get('inv') or []):
            key = str(it[1])
            inv_now[key] = inv_now.get(key, 0) + int(it[2])
        if prev:
            inv_prev = {}
            for it in (prev.get('inv') or []):
                key = str(it[1])
                inv_prev[key] = inv_prev.get(key, 0) + int(it[2])
            gained = {k: v - inv_prev.get(k, 0) for k, v in inv_now.items() if v > inv_prev.get(k, 0)}
            tot = sum(gained.values())
            if tot > 0:
                r += 0.025 * min(tot, 8)
                info['gained'] = tot
            for k in gained:
                if k not in self.seen_items:
                    self.seen_items.add(k)
                    r += 0.55
                    info['new_item'] = k

        # --- spatial novelty ---
        x, y, z = float(s.get('x', 0)), float(s.get('y', 0)), float(s.get('z', 0))
        visits = self._visit(x, z)
        if visits == 1:
            r += 0.50
            info['new_chunk'] = 1
        else:
            r += 0.06 / (1.0 + math.log(visits))

        # --- movement vs idling (bug fixed: compute dist BEFORE updating last_pos) ---
        pos = (round(x, 2), round(y, 2), round(z, 2))
        if self.last_pos is not None:
            moved = math.dist(pos, self.last_pos)
            if moved > 0.08:
                r += min(0.05, moved * 0.02)
                self.idle = 0
            elif not events:
                self.idle += 1
        self.last_pos = pos

        if self.idle > 20:
            r -= 0.03

        # --- social / player proximity (future player interaction) ---
        for e in (cur.get('ent') or []):
            if len(e) >= 9 and int(e[8]) == 2:  # role == 2 (Player)
                dist = float(e[5])
                if 1.5 <= dist <= 6.0 and not (chosen_cand and chosen_cand.get('a') == 'attack'):
                    r += 0.015
                break

        # --- intrinsic curiosity (forward dynamics surprise) ---
        if curiosity > 0.0:
            r += min(0.15, curiosity * 0.10)

        # --- dawn survival bonus (survived a night!) ---
        tday = float(s.get('time', 0.0)) % 24000.0
        if self.last_time > 22000.0 and tday < 2000.0:
            r += 0.80
            info['dawn'] = 1
        self.last_time = tday

        # --- small time cost to prevent pure passivity ---
        r -= 0.004

        self.steps += 1
        self.total += r
        self.return_ = r + 0.99 * self.return_
        self.hp = cur_hp
        self.food = cur_food

        r = max(-CLIP, min(CLIP, r))
        return float(r), info

    def state(self):
        return {
            'seen_items': sorted(self.seen_items),
            'seen_blocks': sorted(self.seen_blocks),
            'seen_crafts': sorted(self.seen_crafts),
            'milestones': sorted(self.milestones),
            'chunks': self.chunks,
            'steps': self.steps,
            'total': self.total,
            'episodes': self.episode,
            'deaths': self.dead,
            'kills': self.kills,
            'crafts': self.crafts,
        }

    def load(self, st):
        self.seen_items = set(st.get('seen_items', []))
        self.seen_blocks = set(st.get('seen_blocks', []))
        self.seen_crafts = set(st.get('seen_crafts', []))
        self.milestones = set(st.get('milestones', []))
        self.chunks = st.get('chunks', {})
        self.steps = st.get('steps', 0)
        self.total = st.get('total', 0.0)
        self.episode = st.get('episodes', 0)
        self.dead = st.get('deaths', 0)
        self.kills = st.get('kills', 0)
        self.crafts = st.get('crafts', 0)