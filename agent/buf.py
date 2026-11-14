"""Replay buffer v2.1: disk-backed memmap with fast O(batch) PER and multi-scale temporal traces."""

import os
import numpy as np

import config as C

HDR_N = 8
TMP_DIM = C.TOKENS * C.ENC

REC = np.dtype([
    ('X0', '<f2', (C.FRAME_DIM,)),          # frame at decision time t
    ('X1', '<f2', (C.FRAME_DIM,)),          # frame at t+1 (for aux dynamics)
    ('Ls', '<f2', (TMP_DIM,)),              # short window + medium/long EMA at t
    ('Ln', '<f2', (TMP_DIM,)),              # short window + medium/long EMA at t+n
    ('Cm', '<f2', (C.MAX_CAND * C.CAND_DIM,)),   # candidates at t
    ('Cn', '<f2', (C.MAX_CAND * C.CAND_DIM,)),   # candidates at t+n
    ('ai', '<i2'),                          # chosen candidate index at t
    ('rw', '<f4'),                          # n-step discounted return
    ('dn', 'u1'),                           # episode terminated within n steps
    ('nc', 'u1'),                           # n candidates at t
    ('ncn', 'u1'),                          # n candidates at t+n
    ('ns', 'u1'),                           # actual n_step (1..N_STEP)
])


class Replay:
    def __init__(self, path=None, n=None, create=True):
        self.path = path or C.REPLAY
        self.n = n or C.BUFFER_N
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self.hdr_path = self.path + '.hdr'
        self.prio_path = self.path + '.prio'
        need = self.n * REC.itemsize
        exists = os.path.exists(self.path) and os.path.getsize(self.path) == need
        if not exists and create:
            if os.path.exists(self.path):
                os.remove(self.path)
            with open(self.path, 'wb') as f:
                f.truncate(need)
            if os.path.exists(self.hdr_path):
                os.remove(self.hdr_path)
            if os.path.exists(self.prio_path):
                os.remove(self.prio_path)
        self.mm = np.memmap(self.path, dtype=REC, mode='r+', shape=(self.n,))

        if not os.path.exists(self.hdr_path) or os.path.getsize(self.hdr_path) != HDR_N * 8:
            with open(self.hdr_path, 'wb') as f:
                f.truncate(HDR_N * 8)
        self.hdr = np.memmap(self.hdr_path, dtype='<i8', mode='r+', shape=(HDR_N,))

        if not os.path.exists(self.prio_path) or os.path.getsize(self.prio_path) != self.n * 4:
            with open(self.prio_path, 'wb') as f:
                f.truncate(self.n * 4)
        self.prio = np.memmap(self.prio_path, dtype='<f4', mode='r+', shape=(self.n,))

        if self.hdr[0] != 3:
            self.hdr[0] = 3     # schema v2.1
            self.hdr[1] = 0     # write index
            self.hdr[2] = 0     # count
            self.hdr[3] = 1000  # max_prio * 1000
            self.hdr.flush()

        # 2-level block-sum index for O(B + N/512) PER sampling (BLK = 512)
        self._blk = 512
        self._n_blk = (self.n + self._blk - 1) // self._blk
        self._pa_buf = None
        self._pa_2d = None
        self._blk_sums = None
        self._pa_cnt = 0
        self._pa_alpha = -1.0
        self._writes_since_flush = 0

    # ------------------------------------------------------------- writing --
    def append(self, rec: dict):
        i = int(self.hdr[1])
        self.mm[i] = (
            rec['X0'].astype(np.float16),
            rec['X1'].astype(np.float16),
            rec['Ls'].reshape(-1).astype(np.float16),
            rec['Ln'].reshape(-1).astype(np.float16),
            rec['Cm'].reshape(-1).astype(np.float16),
            rec['Cn'].reshape(-1).astype(np.float16),
            np.int16(rec['ai']),
            np.float32(rec['rw']),
            np.uint8(1 if rec['dn'] else 0),
            np.uint8(rec['nc']),
            np.uint8(rec['ncn']),
            np.uint8(rec.get('ns', 1)),
        )
        max_p = max(float(self.hdr[3]) / 1000.0, 1.0)
        self.prio[i] = np.float32(max_p)
        self.hdr[1] = (i + 1) % self.n
        self.hdr[2] = min(self.n, int(self.hdr[2]) + 1)
        self._writes_since_flush += 1
        if self._writes_since_flush >= 64:
            self.hdr.flush()
            self._writes_since_flush = 0

    @property
    def count(self):
        return int(self.hdr[2])

    @property
    def write_idx(self):
        return int(self.hdr[1])

    def _ensure_pa(self, cnt: int, alpha: float):
        if self._pa_buf is None or self._pa_alpha != alpha:
            self._pa_buf = np.zeros(self._n_blk * self._blk, dtype=np.float64)
            self._pa_2d = self._pa_buf.reshape(self._n_blk, self._blk)
            self._blk_sums = np.zeros(self._n_blk, dtype=np.float64)
            p = np.maximum(self.prio[:cnt].astype(np.float64), 1e-4)
            self._pa_buf[:cnt] = p ** alpha
            n_blk_active = (cnt + self._blk - 1) // self._blk
            self._blk_sums[:n_blk_active] = self._pa_2d[:n_blk_active].sum(axis=1)
            self._pa_cnt = cnt
            self._pa_alpha = alpha
        elif cnt > self._pa_cnt:
            p = np.maximum(self.prio[self._pa_cnt:cnt].astype(np.float64), 1e-4)
            self._pa_buf[self._pa_cnt:cnt] = p ** alpha
            b0 = self._pa_cnt // self._blk
            b1 = (cnt - 1) // self._blk
            self._blk_sums[b0:b1 + 1] = self._pa_2d[b0:b1 + 1].sum(axis=1)
            self._pa_cnt = cnt

    # ------------------------------------------------------------- reading --
    def sample(self, rng: np.random.RandomState, bs: int, alpha: float = C.PER_ALPHA, beta: float = 0.4):
        cnt = self.count
        if cnt == 0:
            return None, None, None
        k = min(bs, cnt)
        if alpha <= 0.0 or cnt < 32:
            idx = rng.randint(0, cnt, size=k)
            is_w = np.ones(k, dtype=np.float32)
            return self.mm[idx], idx, is_w

        self._ensure_pa(cnt, alpha)
        n_blk = (cnt + self._blk - 1) // self._blk
        blk_cdf = np.cumsum(self._blk_sums[:n_blk])
        total = blk_cdf[-1]
        if not np.isfinite(total) or total <= 0.0:
            idx = rng.randint(0, cnt, size=k)
            is_w = np.ones(k, dtype=np.float32)
            return self.mm[idx], idx, is_w

        # Stratified sampling across k segments via 2-level block-sum tree
        u = (np.arange(k, dtype=np.float64) + rng.rand(k)) * (total / k)
        blk_idx = np.clip(np.searchsorted(blk_cdf, u, side='right'), 0, n_blk - 1)
        blk_start = blk_cdf[blk_idx] - self._blk_sums[blk_idx]
        rem = u - blk_start
        local_cdf = np.cumsum(self._pa_2d[blk_idx], axis=1)
        local_idx = np.clip((local_cdf <= rem[:, None]).sum(axis=1), 0, self._blk - 1)
        idx = np.minimum(blk_idx * self._blk + local_idx, cnt - 1)

        probs = self._pa_buf[idx] / total
        weights = (cnt * np.maximum(probs, 1e-12)) ** (-beta)
        weights /= np.maximum(weights.max(), 1e-8)
        return self.mm[idx], idx, weights.astype(np.float32)

    def update_priorities(self, idx, td_errors):
        if idx is None or len(idx) == 0:
            return
        new_p = np.clip(np.abs(td_errors) + 1e-3, 1e-3, 50.0).astype(np.float32)
        self.prio[idx] = new_p
        if self._pa_buf is not None and self._pa_alpha > 0.0:
            valid = idx < self._pa_cnt
            if np.any(valid):
                vi = idx[valid]
                self._pa_buf[vi] = new_p[valid].astype(np.float64) ** self._pa_alpha
                dirty_blks = np.unique(vi // self._blk)
                self._blk_sums[dirty_blks] = self._pa_2d[dirty_blks].sum(axis=1)
        cur_max = float(self.hdr[3]) / 1000.0
        m = float(new_p.max())
        if m > cur_max:
            self.hdr[3] = int(min(m, 50.0) * 1000)
            self.hdr.flush()