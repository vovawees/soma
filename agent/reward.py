"""Reward model.

Deliberately multi-signal (idea.txt section 12): a single "stay alive" reward
produces a passive agent.  Every component is a small, normalised event signal;
the totals are clipped so one lucky streak cannot dominate the buffer.

The reward is computed here (agent side) because it needs semantics that the
bot must not know about.
"""

import math

CLIP = 3.0

# event -> base reward
EVT = {
    'dig': 0.0,          # handled specially (novelty)
    'attack': 0.10,
    'use': 0.03,
    'place': -0.03,
    'eat': 0.0,
    'act_fail': -0.06,
    'dig_fail': -0.06,
    'bad_action': -0.08,
    'death': -3.0,
    'info': 0.0,
}


class Reward:
    def __init__(self, state=None):
        self.seen_blocks = set()
        self.seen_items = set()
        self.chunks = {}          # "cx,cz" -> visits (bounded)
        self.hp = 20.0
        self.food = 20.0
        self.dead = 0
        self.idle = 0
        self.steps = 0
        self.last_pos = None
        self.episode = 0
        self.total = 0.0
        self.return_ = 0.0
        if state:
            self.load(state)

    # ----------------------------------------------------------- helpers --
    def _chunk_key(self, x, z):
        return f"{int(x) // 16},{int(z) // 16}"

    def _visit(self, x, z):
        k = self._chunk_key(x, z)
        c = self.chunks.get(k, 0) + 1
        if len(self.chunks) < 4096:
            self.chunks[k] = c
        return c

    # -------------------------------------------------------------- step --
    def step(self, prev: dict, cur: dict, events, action, ok) -> tuple:
        r = 0.0
        info = {}

        s = cur.get('self', {})
        ps = prev.get('self', {}) if prev else s

        # --- health / food deltas (works even if the event hook misses) ---
        dhp = float(s.get('hp', 20)) - float(ps.get('hp', 20))
        dfd = float(s.get('food', 20)) - float(ps.get('food', 20))
        if dhp < -0.01:
            r -= 0.15 * min(-dhp, 10.0)
        if dfd < -0.01:
            r -= 0.02

        # --- events ---
        for e in (events or []):
            name = e.get('e') if isinstance(e, dict) else str(e)
            mag = float(e.get('m', 0)) if isinstance(e, dict) else 0.0
            if name == 'dig':
                info['digs'] = info.get('digs', 0) + 1
            elif name == 'hurt':
                pass  # already counted via hp delta
            elif name == 'eat':
                r += 0.10 if float(s.get('food', 20)) < 18 else -0.02
            else:
                r += EVT.get(name, 0.0)
            if name == 'death':
                self.dead += 1
                self.episode += 1
                info['death'] = 1

        # --- novel block types mined ---
        for e in (events or []):
            if isinstance(e, dict) and e.get('e') == 'dig':
                pass  # the bot does not send block names here; handled below

        # --- inventory: new item types ---
        inv_now = {}
        for it in (cur.get('inv') or []):
            inv_now[int(it[1])] = inv_now.get(int(it[1]), 0) + int(it[2])
        if prev:
            inv_prev = {}
            for it in (prev.get('inv') or []):
                inv_prev[int(it[1])] = inv_prev.get(int(it[1]), 0) + int(it[2])
            gained = {k: v - inv_prev.get(k, 0) for k, v in inv_now.items() if v > inv_prev.get(k, 0)}
            tot = sum(gained.values())
            if tot > 0:
                r += 0.02 * min(tot, 8)
                info['gained'] = tot
            for k in gained:
                if k not in self.seen_items:
                    self.seen_items.add(k)
                    r += 0.30
                    info['new_item'] = int(k)

        # --- spatial novelty ---
        x, z = float(s.get('x', 0)), float(s.get('z', 0))
        visits = self._visit(x, z)
        if visits == 1:
            r += 0.25
            info['new_chunk'] = 1
        else:
            r += 0.05 / (1.0 + math.log(visits))

        # --- idling ---
        pos = (round(x, 2), round(float(s.get('y', 0)), 2), round(z, 2))
        if self.last_pos is not None and pos == self.last_pos and not events:
            self.idle += 1
        else:
            self.idle = 0
        self.last_pos = pos
        if self.idle > 40:
            r -= 0.01

        # --- time cost, tiny ---
        r -= 0.005

        # --- dawn bonus ---
        t = float(s.get('time', 0)) % 24000
        if 22000 <= t <= 23000 and not info.get('_dawn_done'):
            pass

        self.steps += 1
        self.total += r
        self.return_ = r + 0.99 * self.return_
        self.hp = float(s.get('hp', 20))
        self.food = float(s.get('food', 20))

        r = max(-CLIP, min(CLIP, r))
        return r, info

    # ------------------------------------------------------- persistence --
    def state(self):
        return {
            'seen_items': sorted(self.seen_items),
            'chunks': self.chunks,
            'steps': self.steps,
            'total': self.total,
            'episodes': self.episode,
            'deaths': self.dead,
        }

    def load(self, st):
        self.seen_items = set(st.get('seen_items', []))
        self.chunks = st.get('chunks', {})
        self.steps = st.get('steps', 0)
        self.total = st.get('total', 0.0)
        self.episode = st.get('episodes', 0)
        self.dead = st.get('deaths', 0)