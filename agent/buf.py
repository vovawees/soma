"""Replay buffer: one fixed-size record per transition, kept on disk.

Observations are stored as raw float16 frames (not just latents) so the encoder
can still be trained later.  Everything lives in a single memmap file; RAM usage
stays at whatever the OS page cache decides, not at buffer size.
"""

import os
import numpy as np

import config as C

HDR_N = 8

REC = np.dtype([
    ('X0', '<f2', (C.FRAME_DIM,)),          # frame at decision time
    ('X1', '<f2', (C.FRAME_DIM,)),          # resulting frame
    ('Ls', '<f2', (C.T * C.ENC,)),          # temporal window at decision time
    ('Cm', '<f2', (C.MAX_CAND * C.CAND_DIM,)),   # candidates at s
    ('Cn', '<f2', (C.MAX_CAND * C.CAND_DIM,)),   # candidates at s'
    ('ai', '<i2'),                          # chosen candidate index
    ('rw', '<f4'),                          # reward
    ('dn', 'u1'),                           # episode done
    ('nc', 'u1'),                           # n candidates at s
    ('ncn', 'u1'),                          # n candidates at s'
])


class Replay:
    def __init__(self, path=None, n=None, create=True):
        self.path = path or C.REPLAY
        self.n = n or C.BUFFER_N
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self.hdr_path = self.path + '.hdr'
        need = self.n * REC.itemsize
        exists = os.path.exists(self.path) and os.path.getsize(self.path) == need
        if not exists and create:
            if os.path.exists(self.path):
                os.remove(self.path)
            with open(self.path, 'wb') as f:
                f.truncate(need)
        self.mm = np.memmap(self.path, dtype=REC, mode='r+', shape=(self.n,))
        if not os.path.exists(self.hdr_path) or os.path.getsize(self.hdr_path) != HDR_N * 8:
            with open(self.hdr_path, 'wb') as f:
                f.truncate(HDR_N * 8)
        self.hdr = np.memmap(self.hdr_path, dtype='<i8', mode='r+', shape=(HDR_N,))
        if self.hdr[0] == 0:
            self.hdr[0] = 1
            self.hdr[1] = 0   # write index
            self.hdr[2] = 0   # count
            self.hdr.flush()

    # ------------------------------------------------------------- writing --
    def append(self, rec: dict):
        i = int(self.hdr[1])
        self.mm[i] = (
            rec['X0'].astype(np.float16), rec['X1'].astype(np.float16),
            rec['Ls'].reshape(-1).astype(np.float16),
            rec['Cm'].reshape(-1).astype(np.float16),
            rec['Cn'].reshape(-1).astype(np.float16),
            np.int16(rec['ai']), np.float32(rec['rw']),
            np.uint8(1 if rec['dn'] else 0),
            np.uint8(rec['nc']), np.uint8(rec['ncn']),
        )
        self.hdr[1] = (i + 1) % self.n
        self.hdr[2] = min(self.n, int(self.hdr[2]) + 1)
        self.hdr.flush()

    @property
    def count(self):
        return int(self.hdr[2])

    @property
    def write_idx(self):
        return int(self.hdr[1])

    # ------------------------------------------------------------- reading --
    def sample(self, rng: np.random.RandomState, bs: int):
        cnt = self.count
        if cnt == 0:
            return None
        idx = rng.randint(0, cnt, size=min(bs, cnt))
        return self.mm[idx], idx