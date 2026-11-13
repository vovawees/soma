"""Offline smoke test: no Minecraft, no server.

Checks the pieces that can be validated without a live world:
  frame encoding -> latent -> Q head -> backward -> loss decreases
  replay buffer record round-trip
  brain frame handling (with a stub socket)
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C
import enc
import model as M
import buf as B

FAILS = []


def check(name, cond, extra=''):
    print(('  ok   ' if cond else '  FAIL ') + name + ('  ' + str(extra) if extra else ''))
    if not cond:
        FAILS.append(name)


def fake_frame(seq=0, px=100.0, py=64.0, pz=-200.0, yaw=0.7, hp=20.0, food=20.0):
    ids = [0] * C.GRID_CELLS
    for i in range(0, C.GRID_CELLS, 7):
        ids[i] = [1, 3, 8, 12, 17, 24, 31][i % 7]
    return {
        't': 'obs', 'seq': seq, 'tick': seq * 3,
        'self': {
            'x': px, 'y': py, 'z': pz, 'vx': 0.02, 'vy': 0.0, 'vz': -0.01,
            'yaw': yaw, 'pitch': 0.1, 'hp': hp, 'food': food, 'air': 20, 'sat': 7.0,
            'light': 9.0, 'dim': 'minecraft:overworld', 'on_ground': True,
            'in_water': False, 'in_lava': False, 'sprinting': False, 'sneaking': False,
            'flying': False, 'fall': 0.0, 'collided': False, 'time': 6000,
            'raining': False, 'thunder': False,
        },
        'grid': {'ox': int(px) - 7, 'oy': int(py) - 7, 'oz': int(pz) - 7, 'step': 2, 'n': 8, 'ids': ids},
        'ent': [[7, 1.2, 0.4, -0.8, 1.6, 20.0, 1], [8, -2.0, 0.0, 3.0, 3.6, 10.0, 0]],
        'inv': [[0, 225, 12, 0], [1, 961, 1, 1], [36, 1000, 1, 0]],
        'cand': [
            {'a': 'mine', 'x': 102, 'y': 63, 'z': -201, 'n': 'stone', 'd': 2.2},
            {'a': 'goto', 'x': 104, 'y': 64, 'z': -202, 'd': 4},
            {'a': 'goto', 'x': 96, 'y': 64, 'z': -198, 'd': 4},
            {'a': 'attack', 'eid': 7, 'n': 'zombie', 'h': 1, 'd': 1.6},
            {'a': 'wait', 'd': 0},
        ],
        'ev': [{'e': 'dig', 'm': 1, 'x': 102, 'y': 63, 'z': -201, 'n': 'stone'}],
        'prev': {'a': 'goto', 'ticks': 2, 'ok': True},
        'res': None,
    }


def main():
    print('== 1. encoding ==')
    fr = fake_frame()
    x = enc.encode(fr)
    check('frame dim', x.shape == (C.FRAME_DIM,), (x.shape, C.FRAME_DIM))
    check('finite', bool(np.isfinite(x).all()))
    check('not all zero', float(np.abs(x).sum()) > 1.0, float(np.abs(x).sum()))
    x2 = enc.encode(fake_frame(px=1000.0, pz=-3000.0, yaw=2.0))
    check('translation invariance of grid', abs(float(x[32:32 + 60].sum()) - float(x2[32:32 + 60].sum())) < 1e-4)

    print('== 2. model ==')
    net = M.Net(seed=3)
    n = net.nparams()
    check('param budget ~3-5M', 2_000_000 < n < 6_000_000, n)
    L = net.embed(x[None, :])[0]
    check('latent shape', L.shape == (1, C.ENC), L.shape)

    print('== 3. training step ==')
    rng = np.random.RandomState(0)
    B_ = 16
    X0 = np.stack([enc.encode(fake_frame(seq=i, px=100.0 + i)) for i in range(B_)]).astype(np.float32)
    X1 = np.stack([enc.encode(fake_frame(seq=i + 1, px=100.0 + i)) for i in range(B_)]).astype(np.float32)
    Ls = np.stack([net.embed(X0[i:i + 1])[0][0] for i in range(B_)]).astype(np.float32)
    Ls = np.repeat(Ls[:, None, :], C.T, axis=1)
    Cm = np.zeros((B_, C.MAX_CAND, C.CAND_DIM), dtype=np.float32)
    for i in range(B_):
        for j in range(5):
            Cm[i, j] = enc.candidate_features(fr['cand'][j], fr)
    mask = np.zeros((B_, C.MAX_CAND), dtype=np.float32)
    mask[:, :5] = 1.0
    batch = {
        'X0': X0, 'X1': X1, 'Ls': Ls, 'Cmat': Cm, 'mask': mask,
        'ai': rng.randint(0, 5, B_), 'rew': rng.randn(B_).astype(np.float32) * 0.3,
        'done': np.zeros(B_, np.float32), 'q_next': np.zeros((B_, C.MAX_CAND), np.float32),
    }
    g, info = M.train_batch(net, batch)
    check('grads finite', all(np.isfinite(v).all() for v in g.values()))
    check('aux loss finite', np.isfinite(info['aux_loss']), info)
    l0 = info['aux_loss']
    first, last = [], []
    for i in range(150):
        g, info = M.train_batch(net, batch)
        net.step(g, 2e-3)
        (first if i < 10 else (last if i >= 140 else [])).append(info['aux_loss'])
    a0, a1 = float(np.mean(first)), float(np.mean(last))
    check('aux loss decreases', a1 < a0 * 0.9, f"{a0:.4f} -> {a1:.4f}")

    print('== 4. replay round-trip ==')
    p = '/tmp/opencode/smoke_replay.f16'
    for f in (p, p + '.hdr'):
        if os.path.exists(f):
            os.remove(f)
    rp = B.Replay(path=p, n=64)
    rec = {'X0': x, 'X1': x2, 'Ls': Ls[0], 'Cm': Cm[0], 'Cn': Cm[0],
           'ai': 2, 'rw': 0.5, 'dn': False, 'nc': 5, 'ncn': 5}
    rp.append(rec)
    check('count', rp.count == 1, rp.count)
    out, _ = rp.sample(rng, 1)
    check('roundtrip X0', np.allclose(out['X0'][0].astype(np.float32), x.astype(np.float16).astype(np.float32)))
    check('roundtrip ai', int(out['ai'][0]) == 2)
    print('   record bytes:', B.REC.itemsize, '-> 100k =', round(B.REC.itemsize * 100000 / 1e9, 2), 'GB')

    print()
    if FAILS:
        print('FAILED:', FAILS)
        sys.exit(1)
    print('all checks passed')


if __name__ == '__main__':
    main()