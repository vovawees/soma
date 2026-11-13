"""SOMA training engine — separate process, low priority, batch updates.

Reads transitions the brain appended to the replay file, takes small SGD steps,
and publishes the new weights atomically (tmp file + os.replace) so the brain
never sees a half-written model.

Run with:  nice -n 15 taskset -c 0-2 python3 trainer.py
"""

import json
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
            except Exception as e:
                print('[train] load failed', e)
        self.buf = B.Replay()
        self.rng = np.random.RandomState(11)
        self.steps = 0
        self.since_sync = 0
        self.last_pub = 0.0
        self.hist = []

    # --------------------------------------------------------- batch prep --
    def make_batch(self, recs):
        B_ = recs.shape[0]
        X0 = recs['X0'].astype(np.float32)
        X1 = recs['X1'].astype(np.float32)
        Ls = recs['Ls'].reshape(B_, C.T, C.ENC).astype(np.float32)
        Cm = recs['Cm'].reshape(B_, C.MAX_CAND, C.CAND_DIM).astype(np.float32)
        Cn = recs['Cn'].reshape(B_, C.MAX_CAND, C.CAND_DIM).astype(np.float32)
        nc = recs['nc'].astype(np.int64)
        ncn = recs['ncn'].astype(np.int64)

        ar = np.arange(B_)
        mask = np.zeros((B_, C.MAX_CAND), dtype=np.float32)
        for i in range(C.MAX_CAND):
            mask[:, i] = (i < np.maximum(nc, 1)).astype(np.float32)

        # next-state latents and Q (target network, no grad)
        L1 = self.net.embed(X1)[0]
        Ls_next = np.concatenate([Ls[:, 1:, :], L1[:, None, :]], axis=1)
        Xn = Ls_next.reshape(B_, C.T * C.ENC)
        hn = np.maximum(Xn @ self.net.t['W3'] + self.net.t['b3'], 0.0, dtype=np.float32)
        S_next = hn @ self.net.t['W4'] + self.net.t['b4']

        ar2 = np.arange(B_)
        Xq = np.concatenate(
            [np.repeat(S_next[:, None, :], C.MAX_CAND, axis=1), Cn], axis=2
        ).reshape(B_ * C.MAX_CAND, C.ENC + C.CAND_DIM)
        hq = np.maximum(Xq @ self.net.t['W5'] + self.net.t['b5'], 0.0, dtype=np.float32)
        q_next = (hq @ self.net.t['W6'] + self.net.t['b6']).reshape(B_, C.MAX_CAND)

        return {
            'X0': X0, 'X1': X1, 'Ls': Ls, 'Cmat': Cm, 'mask': mask,
            'ai': recs['ai'].astype(np.int64), 'rew': recs['rw'].astype(np.float32),
            'done': recs['dn'].astype(np.float32), 'q_next': q_next,
        }

    # -------------------------------------------------------------- loop ---
    def run(self):
        print(f'[train] waiting for {C.WARMUP} transitions (buffer={self.buf.count})')
        while self.buf.count < C.WARMUP:
            time.sleep(1.0)

        last_ckpt = 0.0
        while True:
            recs, _ = self.buf.sample(self.rng, C.BATCH)
            if recs is None:
                time.sleep(0.5)
                continue
            batch = self.make_batch(recs)
            grads, info = M.train_batch(self.net, batch)
            self.net.step(grads, C.LR)
            self.steps += 1
            self.since_sync += 1

            if self.since_sync >= C.TARGET_SYNC:
                self.net.sync_target()
                self.since_sync = 0

            if self.steps % 20 == 0:
                print('[train] step=%d buf=%d q_loss=%.4f v_loss=%.4f aux=%.4f q=%.3f td=%.3f'
                      % (self.steps, self.buf.count, info['q_loss'], info['v_loss'],
                         info['aux_loss'], info['q'], info['td']))
            if self.steps % 200 == 0:
                self.publish()
                last_ckpt = time.time()

            if self.steps % 500 == 0:
                self.log()

    def publish(self):
        tmp = C.CKPT + f'.tmp.{os.getpid()}'
        self.net.save(tmp)
        os.replace(tmp, C.CKPT)

    def log(self):
        rec = {'ts': time.time(), 'train_steps': self.steps, 'buffer': self.buf.count,
               'params': self.net.nparams()}
        with open(C.LOG, 'a') as f:
            f.write(json.dumps(rec) + '\n')


if __name__ == '__main__':
    Trainer().run()