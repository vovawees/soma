"""SOMA brain v2.1 — fast inference, multi-scale EMA memory, persistent world memory & action-failure memory."""

import json
import math
import os
import socket
import time
from collections import deque

import numpy as np

import config as C
import enc
import model as M
import buf as B
import reward as R

HOST = os.environ.get('SOMA_IPC_HOST', '127.0.0.1')
PORT = int(os.environ.get('SOMA_IPC_PORT', '8787'))


class SpatialMemory:
    """Persistent episodic memory for POIs, danger spots, and other players."""

    def __init__(self, state=None):
        self.home = None
        self.craft_tables = []
        self.furnaces = []
        self.chests = []
        self.beds = []
        self.resources = []  # [x, y, z, kind]
        self.deaths = []
        self.players = {}
        if state:
            self.load(state)

    def update(self, fr: dict, ev: list):
        s = fr.get('self', {})
        px, py, pz = float(s.get('x', 0)), float(s.get('y', 0)), float(s.get('z', 0))
        if self.home is None:
            self.home = [px, py, pz]

        for c in (fr.get('cand') or []):
            act = c.get('a')
            if act == 'use':
                nm = str(c.get('n', '')).lower()
                pos = [float(c.get('x', px)), float(c.get('y', py)), float(c.get('z', pz))]
                if 'crafting' in nm:
                    self._add_poi(self.craft_tables, pos, cap=16)
                elif 'furnace' in nm or 'smoker' in nm or 'blast' in nm:
                    self._add_poi(self.furnaces, pos, cap=16)
                elif 'chest' in nm or 'barrel' in nm:
                    self._add_poi(self.chests, pos, cap=24)
                elif 'bed' in nm:
                    self._add_poi(self.beds, pos, cap=8)
                    self.home = pos
            elif act == 'mine':
                nm = str(c.get('n', '')).lower()
                if any(k in nm for k in ('ore', 'diamond', 'ancient', 'iron', 'coal', 'log', 'wood')):
                    pos = [float(c.get('x', px)), float(c.get('y', py)), float(c.get('z', pz))]
                    self._add_poi(self.resources, pos, cap=32)

        now = time.time()
        for e in (fr.get('ent') or []):
            if len(e) >= 9 and int(e[8]) == 2:
                tgt_me = float(e[9]) if len(e) >= 10 else 0.0
                self.players[str(e[0])] = [px + float(e[2]), py + float(e[3]), pz + float(e[4]), now, tgt_me]

        for e in (ev or []):
            if isinstance(e, dict) and e.get('e') in ('death', 'hurt'):
                self._add_poi(self.deaths, [px, py, pz], cap=16)

    @staticmethod
    def _add_poi(lst, pos, cap=16):
        for p in lst:
            if (p[0] - pos[0]) ** 2 + (p[1] - pos[1]) ** 2 + (p[2] - pos[2]) ** 2 < 9.0:
                return
        lst.append(pos)
        if len(lst) > cap:
            lst.pop(0)

    def context(self, fr: dict) -> dict:
        s = fr.get('self', {})
        px, py, pz = float(s.get('x', 0)), float(s.get('y', 0)), float(s.get('z', 0))
        yaw = math.radians(float(s.get('yaw', 0.0)))
        cy, sy = math.cos(yaw), math.sin(yaw)
        fwd_x, fwd_z = -sy, cy
        rgt_x, rgt_z = -cy, -sy

        def rel(target):
            if target is None:
                return 0.0, 0.0, 0.0
            dx, dz = target[0] - px, target[2] - pz
            d = math.hypot(dx, dz)
            if d < 1e-4:
                return 0.0, 0.0, 0.0
            return (dx * rgt_x + dz * rgt_z) / d, (dx * fwd_x + dz * fwd_z) / d, min(d / 64.0, 1.0)

        hr, hf, hd = rel(self.home)
        nearest_ct = min(self.craft_tables, key=lambda p: (p[0] - px) ** 2 + (p[2] - pz) ** 2, default=None)
        cr, cf, _ = rel(nearest_ct)

        now = time.time()
        active_players = [p[:3] for p in self.players.values() if now - p[3] < 60.0 and (len(p) < 5 or p[4] == 0.0)]
        nearest_pl = min(active_players, key=lambda p: (p[0] - px) ** 2 + (p[2] - pz) ** 2, default=None)
        pr, pf, _ = rel(nearest_pl)

        danger = 0.0
        for dp in self.deaths:
            d = math.hypot(dp[0] - px, dp[2] - pz)
            if d < 24.0:
                danger = max(danger, 1.0 - d / 24.0)

        return {
            'home_r': hr, 'home_f': hf, 'home_d': hd,
            'craft_r': cr, 'craft_f': cf,
            'player_r': pr, 'player_f': pf,
            'danger_mem': danger,
        }

    def state(self):
        return {
            'home': self.home,
            'craft_tables': self.craft_tables,
            'furnaces': self.furnaces,
            'chests': self.chests,
            'beds': self.beds,
            'resources': self.resources,
            'deaths': self.deaths,
        }

    def load(self, st):
        self.home = st.get('home')
        self.craft_tables = st.get('craft_tables', [])
        self.furnaces = st.get('furnaces', [])
        self.chests = st.get('chests', [])
        self.beds = st.get('beds', [])
        self.resources = st.get('resources', [])
        self.deaths = st.get('deaths', [])


class GoalGenerator:
    """Hierarchical survival & progression goal generator."""

    def __init__(self, state=None):
        self.goal = 'gather_wood'
        self.completed = set()
        if state:
            self.goal = state.get('goal', 'gather_wood')
            self.completed = set(state.get('completed', []))

    def update(self, fr: dict, mem: SpatialMemory) -> str:
        s = fr.get('self', {})
        hp = float(s.get('hp', 20.0))
        food = float(s.get('food', 20.0))
        inv = fr.get('inv') or []
        ents = fr.get('ent') or []

        inv_names = {}
        for it in inv:
            nm = str(it[1]).lower()
            inv_names[nm] = inv_names.get(nm, 0) + int(it[2])

        has_any = lambda *keys: any(any(k in n for k in keys) for n in inv_names)
        hostile_near = any(len(e) >= 9 and int(e[8]) == 1 and float(e[5]) < 12.0 for e in ents)
        hostile_close = any(len(e) >= 9 and int(e[8]) == 1 and float(e[5]) < 5.5 for e in ents)

        if hp < 8.0 or s.get('in_lava') or s.get('on_fire') or hostile_close:
            self.goal = 'survive_danger'
        elif food < 15.0 and s.get('has_food'):
            self.goal = 'eat_food'
        elif s.get('is_night') and (float(s.get('shelter', 1.0)) < 0.5 or float(s.get('light', 15.0)) < 7.0) and has_any('torch', 'cobble', 'planks', 'dirt', 'stone', 'bed'):
            self.goal = 'build_shelter'
        elif hostile_near and (s.get('has_weapon') or s.get('has_bow')):
            self.goal = 'fight_hostile'
        elif has_any('crafting_table') and not s.get('near_craft'):
            self.goal = 'place_table'
        elif not has_any('wooden_pickaxe', 'stone_pickaxe', 'iron_pickaxe', 'diamond_pickaxe'):
            if has_any('log', 'planks'):
                self.goal = 'craft_basics'
            else:
                self.goal = 'gather_wood'
        elif not has_any('stone_pickaxe', 'iron_pickaxe', 'diamond_pickaxe'):
            if sum(cnt for n, cnt in inv_names.items() if 'cobble' in n or 'stone' in n) >= 3:
                self.goal = 'upgrade_stone'
            else:
                self.goal = 'mine_stone'
        elif has_any('raw_iron', 'iron_ore') or (has_any('furnace') and has_any('coal', 'charcoal')):
            self.goal = 'smelt_iron'
        else:
            self.goal = 'explore_social'
        return self.goal

    def score_candidate(self, c: dict, fr: dict) -> float:
        a = str(c.get('a', ''))
        nm = str(c.get('n', '')).lower()
        val = float(c.get('val', 0.0))
        s = fr.get('self', {})
        score = 0.35 * val

        if self.goal == 'survive_danger':
            if a == 'eat' and float(s.get('food', 20.0)) < 18.0:
                score += 0.90
            elif a in ('flee', 'pillar'):
                score += 0.85
            elif a == 'goto' and val >= 0.9:
                score += 0.80
            elif a in ('shoot', 'attack') and c.get('h', 0):
                score += 0.65
            elif a == 'place' and val >= 0.8:
                score += 0.55
        elif self.goal == 'eat_food':
            if a == 'eat':
                score += 1.00
        elif self.goal == 'build_shelter':
            if a == 'place' and ('torch' in nm or val >= 0.8):
                score += 0.85
            elif a == 'use' and 'bed' in nm:
                score += 0.90
        elif self.goal == 'fight_hostile':
            if a in ('shoot', 'attack') and c.get('h', 0):
                score += 0.90
            elif a == 'goto' and c.get('h', 0):
                score += 0.45
        elif self.goal == 'place_table':
            if a == 'place' and 'crafting_table' in nm:
                score += 0.95
        elif self.goal == 'gather_wood':
            if a == 'mine' and ('log' in nm or 'wood' in nm):
                score += 0.85
            elif a == 'goto' and c.get('item_drop'):
                score += 0.70
        elif self.goal == 'craft_basics':
            if a == 'craft' and any(k in nm for k in ('planks', 'stick', 'crafting_table', 'wooden_pickaxe', 'sword')):
                score += 0.90
        elif self.goal == 'mine_stone':
            if a == 'mine' and any(k in nm for k in ('stone', 'cobble', 'coal', 'ore')):
                score += 0.85
            elif a == 'goto' and c.get('item_drop'):
                score += 0.70
        elif self.goal == 'upgrade_stone':
            if a == 'craft' and any(k in nm for k in ('stone_pickaxe', 'stone_sword', 'furnace', 'torch')):
                score += 0.95
        elif self.goal == 'smelt_iron':
            if a == 'use' and 'furnace' in nm:
                score += 0.85
            elif a == 'place' and 'furnace' in nm:
                score += 0.85
            elif a == 'craft' and 'iron' in nm:
                score += 0.90
        else:  # explore_social
            if a == 'craft' and val >= 0.8:
                score += 0.75
            elif a == 'mine' and ('ore' in nm or 'diamond' in nm or 'coal' in nm or 'log' in nm):
                score += 0.75
            elif a in ('interact_entity', 'give'):
                score += 0.65
            elif a == 'goto' and c.get('item_drop'):
                score += 0.75

        if a in ('wait', 'stop', 'sneak', 'jump', 'drop'):
            score -= 0.25
        return score

    def state(self):
        return {'goal': self.goal, 'completed': sorted(self.completed)}


class EpisodicMemoryBank:
    """Neural Key-Value Episodic Memory (stores salient experiences, queried by cosine similarity)."""

    def __init__(self, cap=C.MEM_SLOTS, dim=C.ENC):
        self.cap = cap
        self.dim = dim
        self.keys = np.zeros((cap, dim), dtype=np.float32)
        self.vals = np.zeros((cap, dim), dtype=np.float32)
        self.imp = np.zeros(cap, dtype=np.float32)
        self.count = 0

    def query(self, l: np.ndarray) -> np.ndarray:
        if self.count == 0:
            return np.zeros(self.dim, dtype=np.float32)
        norm = float(np.linalg.norm(l))
        if norm < 1e-6:
            return np.zeros(self.dim, dtype=np.float32)
        q = l / norm
        k = self.keys[:self.count]
        sims = (k @ q) * 8.0
        sims -= np.max(sims)
        w = np.exp(sims)
        w /= np.sum(w)
        return (w @ self.vals[:self.count]).astype(np.float32)

    def store(self, key_l: np.ndarray, val_l: np.ndarray, importance: float):
        norm = float(np.linalg.norm(key_l))
        if norm < 1e-6:
            return
        k_norm = (key_l / norm).astype(np.float32)
        if self.count > 0:
            sims = self.keys[:self.count] @ k_norm
            best_i = int(np.argmax(sims))
            if float(sims[best_i]) > 0.92:
                # Merge into existing similar slot to preserve memory diversity
                if importance >= self.imp[best_i]:
                    self.vals[best_i] = (0.5 * self.vals[best_i] + 0.5 * val_l).astype(np.float32)
                    self.imp[best_i] = float(importance)
                return
        if self.count < self.cap:
            idx = self.count
            self.count += 1
        else:
            # Decay old importances slightly and replace the least important slot
            self.imp *= 0.995
            idx = int(np.argmin(self.imp))
            if importance < self.imp[idx] * 0.5:
                return
        self.keys[idx] = k_norm
        self.vals[idx] = val_l.astype(np.float32)
        self.imp[idx] = float(importance)

    def save(self, path):
        np.savez(path, keys=self.keys, vals=self.vals, imp=self.imp, count=np.array([self.count]))

    def load(self, path):
        if not os.path.exists(path):
            return
        try:
            z = np.load(path)
            if z['keys'].shape == self.keys.shape:
                self.keys = z['keys'].astype(np.float32)
                self.vals = z['vals'].astype(np.float32)
                self.imp = z['imp'].astype(np.float32)
                self.count = int(z['count'][0])
        except Exception:
            pass


class Brain:
    def __init__(self):
        self.net = M.Net(seed=1)
        self.buf = B.Replay()
        state_data = self._load_state()
        self.rw = R.Reward(state=state_data.get('reward') if state_data else None)
        self.mem = SpatialMemory(state=state_data.get('memory') if state_data else None)
        self.goals = GoalGenerator(state=state_data.get('goals') if state_data else None)
        self.ep_mem = EpisodicMemoryBank()
        self.ep_mem_path = os.path.join(C.DATA, 'memory.npz')
        self.ep_mem.load(self.ep_mem_path)

        self.hist = deque(maxlen=C.T)
        self._win_buf = np.empty((C.TOKENS, C.ENC), dtype=np.float32)
        self._hist_filled = False
        self.ema_med = None     # alpha=0.15 (~10s memory)
        self.ema_slow = None    # alpha=0.04 (~40s memory)
        self.nstep_q = deque()
        self.fail_counts = {}   # (action, x, y, z) -> recent failure count

        self.prev = None
        self.prev_x = None
        self.prev_l = None
        self.prev_cands_raw = None
        self.prev_cands = None
        self.prev_nc = 0
        self.prev_ai = 0
        self.prev_window = None

        self.last_reward = 0.0
        self.ret = 0.0
        self.ck = 0.0
        self.choices = 0
        self.rng = np.random.RandomState(7)
        self.frame_seq = 0
        self.last_type = None
        self.repeat = 0
        self.step_n = self.rw.steps
        self.model_mtime = 0.0
        self.errors = 0
        self.eps = 0.35
        self.action_counts = {}

        if os.path.exists(C.CKPT):
            try:
                self.net.load(C.CKPT)
                self.model_mtime = os.path.getmtime(C.CKPT)
                print(f'[brain] loaded checkpoint ({self.net.nparams()/1e6:.2f}M params)')
            except Exception as e:
                print('[brain] checkpoint load failed:', e)
        else:
            print(f'[brain] fresh model ({self.net.nparams()/1e6:.2f}M params)')
            os.makedirs(C.DATA, exist_ok=True)
            tmp = C.CKPT + '.tmp.npz'
            self.net.save(tmp)
            os.replace(tmp, C.CKPT)
            self.model_mtime = os.path.getmtime(C.CKPT)

    def _load_state(self):
        if os.path.exists(C.STATE):
            try:
                with open(C.STATE, 'r') as f:
                    return json.load(f)
            except Exception:
                pass
        return None

    def save_state(self):
        try:
            os.makedirs(C.DATA, exist_ok=True)
            tmp = C.STATE + '.tmp'
            with open(tmp, 'w') as f:
                json.dump({'reward': self.rw.state(), 'memory': self.mem.state(), 'goals': self.goals.state()}, f)
            os.replace(tmp, C.STATE)
            self.ep_mem.save(self.ep_mem_path)
        except Exception:
            pass

    @staticmethod
    def _cand_key(c: dict):
        return (str(c.get('a', '')), c.get('x'), c.get('y'), c.get('z'), c.get('eid'), c.get('rid'))

    def choose(self, window, cands, Mx, n, frame=None):
        if n == 0:
            return 0
        Q = self.net.infer_q(window[None, ...], Mx[None, ...], n).copy()
        self.ck = float(np.mean(Q))

        # Goal-guided policy prior: prevents random wandering early in training and decays as Q-net matures
        if frame is not None:
            prior_w = max(0.15, 1.10 * (0.99985 ** self.step_n))
            for i in range(n):
                Q[i] += prior_w * self.goals.score_candidate(cands[i], frame)

        # Penalize recently failed exact actions to prevent getting stuck
        if self.fail_counts:
            for i in range(n):
                fc = self.fail_counts.get(self._cand_key(cands[i]), 0)
                if fc > 0:
                    Q[i] -= 0.5 * fc

        self.eps = max(0.08, 0.35 * (0.9997 ** self.step_n))
        tau = 0.35
        if self.repeat > 10:
            self.eps = max(self.eps, 0.40)
            tau = 0.70

        if self.rng.rand() < self.eps:
            # Boltzmann + UCB novelty bonus over action types so exploration tries under-explored mechanics
            total_c = max(1, sum(self.action_counts.values()))
            q_exp = Q.copy()
            for i in range(n):
                at = str(cands[i].get('a', ''))
                cnt = self.action_counts.get(at, 0)
                q_exp[i] += 0.25 * math.sqrt(math.log(total_c + 1.0) / (cnt + 1.0))
            q_scaled = (q_exp - np.max(q_exp)) / tau
            probs = np.exp(np.clip(q_scaled, -20.0, 0.0))
            probs /= np.sum(probs)
            ai = int(self.rng.choice(n, p=probs))
        else:
            ai = int(np.argmax(Q))

        atype = str(cands[ai].get('a', ''))
        self.action_counts[atype] = self.action_counts.get(atype, 0) + 1
        if atype == self.last_type:
            self.repeat += 1
        else:
            self.repeat = 1
            self.last_type = atype
        return ai

    def maybe_reload(self):
        try:
            m = os.path.getmtime(C.CKPT)
        except OSError:
            return
        if m > self.model_mtime + 1e-6:
            try:
                self.net.load(C.CKPT)
                self.model_mtime = m
                print(f'[brain] model updated -> {time.strftime("%H:%M:%S")}')
            except Exception as e:
                print('[brain] reload failed:', e)

    def _flush_nstep(self, cur_window, cur_cands, ncn, force_all=False):
        while self.nstep_q and (force_all or len(self.nstep_q) >= C.N_STEP):
            k = min(len(self.nstep_q), C.N_STEP)
            R_n = 0.0
            done_n = False
            g = 1.0
            for idx in range(k):
                step_rec = self.nstep_q[idx]
                R_n += g * step_rec['rw']
                g *= C.GAMMA
                if step_rec['dn']:
                    done_n = True
                    k = idx + 1
                    break
            head = self.nstep_q.popleft()
            try:
                self.buf.append({
                    'X0': head['X0'],
                    'X1': head['X1'],
                    'Ls': head['Ls'],
                    'Ln': cur_window,
                    'Cm': head['Cm'],
                    'Cn': cur_cands,
                    'ai': head['ai'],
                    'rw': R_n,
                    'dn': done_n,
                    'nc': head['nc'],
                    'ncn': ncn,
                    'ns': k,
                })
            except Exception as e:
                print('[brain] buffer append failed:', e)
            if not force_all:
                break

    # ---------------------------------------------------------- frame path --
    def on_frame(self, fr):
        self.frame_seq = int(fr.get('seq', 0))
        ev = fr.get('ev') or []
        self.mem.update(fr, ev)
        mem_ctx = self.mem.context(fr)

        x = enc.encode(fr, mem_ctx=mem_ctx, last_rew=self.last_reward, last_ret=self.ret)
        l = self.net.infer_embed(x[None, :])[0]

        cands = fr.get('cand') or []
        cur_cands, ncn = enc.encode_candidates(cands, fr)

        # Update action-failure memory
        if self.prev_cands_raw and self.prev_ai < len(self.prev_cands_raw):
            chosen = self.prev_cands_raw[self.prev_ai]
            failed = any(isinstance(e, dict) and e.get('e') in ('act_fail', 'dig_fail', 'bad_action') for e in ev)
            ck = self._cand_key(chosen)
            if failed:
                self.fail_counts[ck] = min(self.fail_counts.get(ck, 0) + 1, 6)
            elif ck in self.fail_counts:
                self.fail_counts.pop(ck, None)
            if self.step_n % 25 == 0 and self.fail_counts:
                self.fail_counts = {k: v - 1 for k, v in self.fail_counts.items() if v > 1}

            if self.prev is not None and self.prev_x is not None and self.prev_window is not None:
                curiosity = 0.0
                if self.prev_l is not None and self.prev_nc > 0:
                    c_vec = self.prev_cands[self.prev_ai:self.prev_ai + 1]
                    l_pred = self.net.infer_aux(self.prev_l[None, :], c_vec)[0]
                    curiosity = float(np.mean((l_pred - l) ** 2))

                rew, info = self.rw.step(self.prev, fr, ev, chosen_cand=chosen, curiosity=curiosity)
                done = any((e.get('e') if isinstance(e, dict) else e) == 'death' for e in ev)

                # Store salient experiences into neural episodic memory
                if self.prev_l is not None and (abs(rew) >= 0.35 or done or info.get('craft') or info.get('kill')):
                    self.ep_mem.store(self.prev_l, l, max(abs(rew), 0.35))

                self.nstep_q.append({
                    'X0': self.prev_x,
                    'X1': x,
                    'Ls': self.prev_window,
                    'Cm': self.prev_cands,
                    'ai': self.prev_ai,
                    'rw': rew,
                    'dn': done,
                    'nc': int(self.prev_nc),
                })

                self.last_reward = rew
                self.ret = 0.99 * self.ret + rew
                self.step_n += 1

        # Advance short-term window + medium/long EMA + retrieved neural episodic memory
        if not self._hist_filled:
            self._win_buf[:C.T] = l
            self._hist_filled = True
        else:
            self._win_buf[:C.T - 1] = self._win_buf[1:C.T]
            self._win_buf[C.T - 1] = l

        if self.ema_med is None:
            self.ema_med = l.copy()
            self.ema_slow = l.copy()
        else:
            self.ema_med = (0.85 * self.ema_med + 0.15 * l).astype(np.float32)
            self.ema_slow = (0.96 * self.ema_slow + 0.04 * l).astype(np.float32)

        self._win_buf[C.T] = self.ema_med
        self._win_buf[C.T + 1] = self.ema_slow
        self._win_buf[C.T + 2] = self.ep_mem.query(l)
        window = self._win_buf.copy()

        if self.nstep_q:
            done_last = self.nstep_q[-1]['dn']
            self._flush_nstep(window, cur_cands, int(ncn), force_all=done_last)

        self.goals.update(fr, self.mem)
        ai = self.choose(window, cands, cur_cands, ncn, frame=fr)

        self.prev = fr
        self.prev_x = x
        self.prev_l = l
        self.prev_cands_raw = cands
        self.prev_cands = cur_cands
        self.prev_nc = ncn
        self.prev_ai = ai
        self.prev_window = window
        self.choices += 1

        return ai

    # ------------------------------------------------------------ metrics --
    def log(self):
        try:
            rec = {
                'ts': time.time(), 'step': self.step_n, 'frames': self.choices,
                'goal': self.goals.goal,
                'reward': round(self.last_reward, 4),
                'ret': round(self.ret, 3), 'q': round(self.ck, 4),
                'eps': round(self.eps, 4), 'buffer': self.buf.count,
                'items': len(self.rw.seen_items), 'blocks': len(self.rw.seen_blocks),
                'crafts': len(self.rw.seen_crafts), 'milestones': len(self.rw.milestones),
                'chunks': len(self.rw.chunks), 'kills': self.rw.kills,
                'deaths': self.rw.dead, 'hp': round(self.rw.hp, 1), 'food': self.rw.food,
            }
            with open(C.LOG, 'a') as f:
                f.write(json.dumps(rec) + '\n')
        except Exception:
            pass


def connect():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    for _ in range(1000):
        try:
            s.connect((HOST, PORT))
            return s
        except OSError:
            time.sleep(2)
    raise SystemExit('[brain] cannot reach bot on %s:%d' % (HOST, PORT))


def main():
    br = Brain()
    sock = connect()
    print('[brain] connected to adapter')
    buf = ''
    last_log = 0.0
    last_reload = 0.0
    last_state = 0.0

    while True:
        try:
            data = sock.recv(262144)
        except OSError as e:
            print('[brain] socket error:', e)
            time.sleep(2)
            sock = connect()
            buf = ''
            continue
        if not data:
            print('[brain] adapter closed connection')
            time.sleep(2)
            sock = connect()
            buf = ''
            continue

        buf += data.decode('utf-8', 'replace')
        if '\n' not in buf:
            continue
        lines = buf.split('\n')
        buf = lines.pop()
        for line in lines:
            if not line.strip():
                continue
            try:
                fr = json.loads(line)
            except Exception:
                continue
            if fr.get('t') != 'obs':
                continue

            try:
                ai = br.on_frame(fr)
                sock.sendall((json.dumps({'t': 'act', 'i': ai, 's': br.frame_seq}, separators=(',', ':')) + '\n').encode())
            except Exception as e:
                br.errors += 1
                print('[brain] frame error:', repr(e))
                try:
                    sock.sendall((json.dumps({'t': 'act', 'i': 0, 's': br.frame_seq}, separators=(',', ':')) + '\n').encode())
                except OSError:
                    pass

            now = time.time()
            if now - last_reload > 4.0:
                br.maybe_reload()
                last_reload = now
            if now - last_log > 5.0:
                br.log()
                last_log = now
            if now - last_state > 30.0:
                br.save_state()
                last_state = now


if __name__ == '__main__':
    main()