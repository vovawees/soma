"""SOMA training engine v2 — Double Dueling DQN + PER + N-step + Polyak Target.

Reads N-step transitions from the disk-backed prioritized replay buffer,
computes Double DQN targets, updates weights with Adam + gradient clipping,
and atomically publishes checkpoints for brain.py.
"""

import json
import math
import os
import time

import numpy as np

import config as C
import model as M
import buf as B


class Trainer:
    def __init__(self):
        self.net = M.Net(seed=1)
        if os.path.exists(C.CKPT):
            try:
                self.net.load(C.CKPT)
                print(f'[train] loaded checkpoint ({self.net.nparams()/1e6:.2f}M params)')
            except Exception as e:
                print('[train] load failed', e)
        self.buf = B.Replay()
        self.rng = np.random.RandomState(11)
        self.steps = 0
        self.cand_idx = np.arange(C.MAX_CAND)[None, :]

    def _lr(self):
        # Cosine annealing with floor
        progress = min(1.0, self.steps / 100_000.0)
        return C.LR_MIN + 0.5 * (C.LR - C.LR_MIN) * (1.0 + math.cos(math.pi * progress))

    def _beta(self):
        return min(1.0, C.PER_BETA_START + (1.0 - C.PER_BETA_START) * (self.steps / 100_000.0))

    # --------------------------------------------------------- batch prep --
    def make_batch(self, recs, is_w):
        B_ = recs.shape[0]
        X0 = recs['X0'].astype(np.float32)
        X1 = recs['X1'].astype(np.float32)
        Ls = recs['Ls'].reshape(B_, C.T + 2, C.ENC).astype(np.float32)
        Ln = recs['Ln'].reshape(B_, C.T + 2, C.ENC).astype(np.float32)
        Cm = recs['Cm'].reshape(B_, C.MAX_CAND, C.CAND_DIM).astype(np.float32)
        Cn = recs['Cn'].reshape(B_, C.MAX_CAND, C.CAND_DIM).astype(np.float32)
        nc = np.maximum(recs['nc'].astype(np.int64), 1)
        ncn = np.maximum(recs['ncn'].astype(np.int64), 1)

        mask = (self.cand_idx < nc[:, None]).astype(np.float32)
        mask_n = (self.cand_idx < ncn[:, None]).astype(np.float32)

        # Double DQN: select best action with online net, evaluate with target net
        S_next_online = self.net.temporal(Ln, use_target=False)[0]
        q_next_online = self.net.q_all(S_next_online, Cn, mask=mask_n, use_target=False)[0]
        q_next_online_masked = np.where(mask_n > 0, q_next_online, -1e9)
        best_a = np.argmax(q_next_online_masked, axis=1)

        S_next_target = self.net.temporal(Ln, use_target=True)[0]
        q_next_target = self.net.q_all(S_next_target, Cn, mask=mask_n, use_target=True)[0]
        q_target_sel = q_next_target[np.arange(B_), best_a]
        q_target_sel = np.where(np.isfinite(q_target_sel), q_target_sel, 0.0).astype(np.float32)

        ns = recs['ns'].astype(np.float32) if 'ns' in recs.dtype.names else float(C.N_STEP)

        return {
            'X0': X0, 'X1': X1, 'Ls': Ls, 'Cmat': Cm, 'mask': mask,
            'ai': recs['ai'].astype(np.int64),
            'rew': recs['rw'].astype(np.float32),
            'done': recs['dn'].astype(np.float32),
            'q_target_sel': q_target_sel,
            'is_w': is_w,
            'n_step': ns,
        }

    # -------------------------------------------------------------- loop ---
    def run(self):
        print(f'[train] waiting for {C.WARMUP} transitions (buffer={self.buf.count})')
        while self.buf.count < C.WARMUP:
            time.sleep(1.0)

        while True:
            t0 = time.time()
            recs, idx, is_w = self.buf.sample(self.rng, C.BATCH, alpha=C.PER_ALPHA, beta=self._beta())
            if recs is None:
                time.sleep(0.5)
                continue

            batch = self.make_batch(recs, is_w)
            grads, info = M.train_batch(self.net, batch)
            gnorm = self.net.step(grads, self._lr())
            self.net.soft_sync_target(C.TAU)
            self.buf.update_priorities(idx, info['td_vec'])

            self.steps += 1
            if self.steps % 50 == 0:
                print('[train] step=%d buf=%d lr=%.2e q_loss=%.4f aux=%.4f q=%.3f td=%.3f g=%.2f'
                      % (self.steps, self.buf.count, self._lr(), info['q_loss'],
                         info['aux_loss'], info['q'], info['td'], gnorm))
            if self.steps % 150 == 0:
                self.publish()
            if self.steps % 500 == 0:
                self.log(info)

            dt = (time.time() - t0) * 1000.0
            if dt < C.TRAIN_EVERY_MS:
                time.sleep((C.TRAIN_EVERY_MS - dt) / 1000.0)

    def publish(self):
        tmp = C.CKPT + f'.tmp.{os.getpid()}.npz'
        self.net.save(tmp)
        os.replace(tmp, C.CKPT)

    def log(self, info):
        rec = {
            'ts': time.time(), 'train_steps': self.steps, 'buffer': self.buf.count,
            'params': self.net.nparams(),
            'q_loss': round(info['q_loss'], 5),
            'aux_loss': round(info['aux_loss'], 5),
            'td': round(info['td'], 4),
        }
        try:
            with open(C.LOG, 'a') as f:
                f.write(json.dumps(rec) + '\n')
        except Exception:
            pass


if __name__ == '__main__':
    Trainer().run()