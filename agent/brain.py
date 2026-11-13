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
            if c.get('a') == 'use':
                nm = str(c.get('n', '')).lower()
                pos = [float(c.get('x', px)), float(c.get('y', py)), float(c.get('z', pz))]
                if 'crafting' in nm:
                    self._add_poi(self.craft_tables, pos, cap=16)
                elif 'furnace' in nm or 'smoker' in nm or 'blast' in nm:
                    self._add_poi(self.furnaces, pos, cap=16)
                elif 'chest' in nm or 'barrel' in nm:
                    self._add_poi(self.chests, pos, cap=24)
                elif 'bed' in nm:
                    self.home = pos

        now = time.time()
        for e in (fr.get('ent') or []):
            if len(e) >= 9 and int(e[8]) == 2:
                self.players[str(e[0])] = [px + float(e[2]), py + float(e[3]), pz + float(e[4]), now]

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
        active_players = [p[:3] for p in self.players.values() if now - p[3] < 60.0]
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
            'deaths': self.deaths,
        }

    def load(self, st):
        self.home = st.get('home')
        self.craft_tables = st.get('craft_tables', [])
        self.furnaces = st.get('furnaces', [])
        self.chests = st.get('chests', [])
        self.deaths = st.get('deaths', [])


class Brain:
    def __init__(self):
        self.net = M.Net(seed=1)
        self.buf = B.Replay()
        state_data = self._load_state()
        self.rw = R.Reward(state=state_data.get('reward') if state_data else None)
        self.mem = SpatialMemory(state=state_data.get('memory') if state_data else None)

        self.hist = deque(maxlen=C.T)
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
                json.dump({'reward': self.rw.state(), 'memory': self.mem.state()}, f)
            os.replace(tmp, C.STATE)
        except Exception:
            pass

    @staticmethod
    def _cand_key(c: dict):
        return (str(c.get('a', '')), c.get('x'), c.get('y'), c.get('z'), c.get('eid'), c.get('rid'))

    def choose(self, window, cands, Mx, n):
        if n == 0:
            return 0
        Q = self.net.infer_q(window[None, ...], Mx[None, ...], n).copy()
        self.ck = float(np.mean(Q))

        # Penalize recently failed exact actions to prevent getting stuck
        if self.fail_counts:
            for i in range(n):
                fc = self.fail_counts.get(self._cand_key(cands[i]), 0)
                if fc > 0:
                    Q[i] -= 0.5 * fc

        self.eps = max(0.08, 0.35 * (0.9997 ** self.step_n))
        if self.repeat > 10:
            self.eps = max(self.eps, 0.35)

        if self.rng.rand() < self.eps:
            ai = int(self.rng.randint(0, n))
        else:
            ai = int(np.argmax(Q))

        atype = str(cands[ai].get('a', ''))
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

                rew, _ = self.rw.step(self.prev, fr, ev, chosen_cand=chosen, curiosity=curiosity)
                done = any((e.get('e') if isinstance(e, dict) else e) == 'death' for e in ev)

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

        # Advance short-term window + medium/long EMA memory traces
        self.hist.append(l)
        while len(self.hist) < C.T:
            self.hist.appendleft(self.hist[0])
        if self.ema_med is None:
            self.ema_med = l.copy()
            self.ema_slow = l.copy()
        else:
            self.ema_med = (0.85 * self.ema_med + 0.15 * l).astype(np.float32)
            self.ema_slow = (0.96 * self.ema_slow + 0.04 * l).astype(np.float32)

        window = np.concatenate([np.stack(self.hist), self.ema_med[None, :], self.ema_slow[None, :]], axis=0).astype(np.float32)

        if self.nstep_q:
            done_last = self.nstep_q[-1]['dn']
            self._flush_nstep(window, cur_cands, int(ncn), force_all=done_last)

        ai = self.choose(window, cands, cur_cands, ncn)

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
        while '\n' in buf:
            line, buf = buf.split('\n', 1)
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
                sock.sendall((json.dumps({'t': 'act', 'i': ai, 's': br.frame_seq}) + '\n').encode())
            except Exception as e:
                br.errors += 1
                print('[brain] frame error:', repr(e))
                try:
                    sock.sendall((json.dumps({'t': 'act', 'i': 0, 's': br.frame_seq}) + '\n').encode())
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