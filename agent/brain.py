"""SOMA brain — the decision loop.

Connects to the bot over a local socket, turns every frame into a latent,
keeps a temporal window, learns from the consequences of the action it took,
and answers with the index of the candidate it wants executed.

Nothing here is scripted behaviour: there are no if-there-is-a-tree-then-chop
rules.  The only priors are (a) the candidate set the bot advertises and
(b) the reward definition.
"""

import json
import os
import socket
import sys
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


class Brain:
    def __init__(self):
        self.net = M.Net(seed=1)
        self.buf = B.Replay()
        self.rw = R.Reward()
        self.hist = deque(maxlen=C.T)
        self.prev = None
        self.prev_x = None
        self.prev_cands = None
        self.prev_ai = 0
        self.prev_window = None
        self.last_reward = 0.0
        self.ret = 0.0
        self.ck = 0.0
        self.choices = 0
        self.rng = np.random.RandomState(7)
        self.last_save = 0.0
        self.step_n = 0
        self.model_mtime = 0.0
        self.log_every = 50
        self.eps = 0.30
        if os.path.exists(C.CKPT):
            try:
                self.net.load(C.CKPT)
                self.model_mtime = os.path.getmtime(C.CKPT)
                print(f'[brain] loaded checkpoint ({self.net.nparams()/1e6:.2f}M params)')
            except Exception as e:
                print('[brain] checkpoint load failed:', e)
        else:
            print(f'[brain] fresh model ({self.net.nparams()/1e6:.2f}M params)')
            self.net.save(C.CKPT + '.tmp.npz')
            os.replace(C.CKPT + '.tmp.npz', C.CKPT)
            self.model_mtime = os.path.getmtime(C.CKPT)

    # ------------------------------------------------------------ helpers --
    def cand_matrix(self, cands):
        Mx = np.zeros((C.MAX_CAND, C.CAND_DIM), dtype=np.float32)
        if not cands:
            return Mx, 0
        n = min(len(cands), C.MAX_CAND)
        for i in range(n):
            Mx[i] = enc.candidate_features(cands[i], self.prev if self.prev else {})
        return Mx, n

    def choose(self, window, cands):
        Mx, n = self.cand_matrix(cands)
        if n == 0:
            return 0
        S = self.net.temporal(window[None, ...])[0]
        Q = self.net.q_all(S, Mx[None, ...])[0][0]
        self.ck = float(np.mean(Q[:n]))
        # epsilon-greedy over admissible candidates
        self.eps = max(0.05, 0.30 * (0.9995 ** self.step_n))
        if self.rng.rand() < self.eps:
            return int(self.rng.randint(0, n))
        return int(np.argmax(Q[:n]))

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

    # ---------------------------------------------------------- frame path --
    def on_frame(self, fr):
        x = enc.encode(fr)
        l = self.net.embed(x[None, :].astype(np.float32))[0][0]

        if self.prev is not None and self.prev_x is not None:
            cur_cands, ncn = self.cand_matrix(fr.get('cand') or [])
            ev = fr.get('ev') or []
            rew, info = self.rw.step(self.prev, fr, ev, None, None)
            done = any((e.get('e') if isinstance(e, dict) else e) == 'death' for e in ev)

            if self.prev_window is not None:
                try:
                    self.buf.append({
                        'X0': self.prev_x, 'X1': x, 'Ls': self.prev_window,
                        'Cm': self.prev_cands, 'Cn': cur_cands,
                        'ai': self.prev_ai, 'rw': rew, 'dn': done,
                        'nc': int(self.prev_nc), 'ncn': int(ncn),
                    })
                except Exception as e:
                    print('[brain] buffer append failed:', e)

            self.last_reward = rew
            self.ret = 0.99 * self.ret + rew
            self.step_n += 1
            if info:
                pass

        # advance the temporal window
        self.hist.append(l)
        while len(self.hist) < C.T:
            self.hist.appendleft(self.hist[0])
        window = np.stack(self.hist).astype(np.float32)

        # choose what to do now
        cands = fr.get('cand') or []
        ai = self.choose(window, cands)

        self.prev = fr
        self.prev_x = x
        self.prev_cands, self.prev_nc = self.cand_matrix(cands)
        self.prev_ai = ai
        self.prev_window = window
        self.choices += 1

        return ai

    # ------------------------------------------------------------ metrics --
    def log(self, sock):
        try:
            rec = {
                'ts': time.time(), 'step': self.step_n, 'frames': self.choices,
                'reward': round(self.last_reward, 4),
                'ret': round(self.ret, 3), 'q': round(self.ck, 4),
                'eps': round(self.eps, 4), 'buffer': self.buf.count,
                'items': len(self.rw.seen_items), 'chunks': len(self.rw.chunks),
                'deaths': self.rw.dead, 'hp': self.rw.hp, 'food': self.rw.food,
            }
            with open(C.LOG, 'a') as f:
                f.write(json.dumps(rec) + '\n')
        except Exception:
            pass


def connect():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    for attempt in range(1000):
        try:
            s.connect((HOST, PORT))
            return s
        except OSError:
            time.sleep(2)
    raise SystemExit('[brain] cannot reach bot on %s:%d' % (HOST, PORT))


def main():
    br = Brain()
    sock = connect()
    print('[brain] connected to bot')
    buf = ''
    last_log = 0.0
    last_reload = 0.0

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
            print('[brain] bot closed connection')
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

            ai = br.on_frame(fr)
            try:
                sock.sendall((json.dumps({'t': 'act', 'i': ai}) + '\n').encode())
            except OSError:
                pass

            now = time.time()
            if now - last_reload > 5.0:
                br.maybe_reload()
                last_reload = now
            if now - last_log > 5.0:
                br.log(sock)
                last_log = now


if __name__ == '__main__':
    main()