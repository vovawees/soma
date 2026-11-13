"""Offline smoke test v2: validates encoding, Dueling Double-DQN, PER replay, reward, and Brain."""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C
import enc
import model as M
import buf as B
import reward as R
import brain as BR

FAILS = []


def check(name, cond, extra=''):
    print(('  ok   ' if cond else '  FAIL ') + name + ('  ' + str(extra) if extra else ''))
    if not cond:
        FAILS.append(name)


def fake_frame(seq=0, px=100.0, py=64.0, pz=-200.0, yaw=45.0, hp=20.0, food=20.0):
    ids = [0] * C.GRID_CELLS
    flags = [0] * C.GRID_CELLS
    for i in range(0, C.GRID_CELLS, 7):
        ids[i] = [1, 3, 8, 12, 17, 24, 31][i % 7]
        flags[i] = 1
    return {
        't': 'obs', 'seq': seq, 'tick': seq * 6,
        'self': {
            'x': px, 'y': py, 'z': pz, 'vx': 0.05, 'vy': 0.0, 'vz': -0.02,
            'yaw': yaw, 'pitch': 5.0, 'hp': hp, 'food': food, 'air': 300, 'sat': 7.0,
            'light': 12.0, 'dim': 'minecraft:overworld', 'on_ground': True,
            'in_water': False, 'in_lava': False, 'on_fire': False,
            'sprinting': False, 'sneaking': False, 'fall': 0.0, 'collided': False,
            'time': 6000, 'raining': False, 'thunder': False,
            'armor': 4.0, 'atk_cd': 1.0, 'near_craft': True, 'near_furnace': False,
            'sel_slot': 0, 'has_food': True, 'has_tool': True, 'has_weapon': True,
            'has_blocks': True, 'is_night': False,
        },
        'grid': {'ox': int(px) - 3, 'oy': int(py) - 2, 'oz': int(pz) - 3, 'step': 1, 'n': 8, 'ids': ids, 'flags': flags},
        'ent': [
            [7, 'minecraft:zombie', 1.2, 0.0, -0.8, 1.44, 20.0, 20.0, 1, 1, 0.05, 0.0],
            [8, 'minecraft:item', -2.0, 0.0, 3.0, 3.6, -1.0, 1.0, 3, 0, 0.0, 0.0],
            [9, 'Steve', 3.0, 0.0, 2.0, 3.6, 20.0, 20.0, 2, 0, 0.0, 0.0],
        ],
        'inv': [[0, 225, 1, 0, 1, 0, 0, 0.9], [1, 961, 4, 1, 0, 0, 0, 1.0], [36, 1000, 1, 0, 0, 1, 1, 1.0]],
        'cand': [
            {'a': 'mine', 'x': 102, 'y': 63, 'z': -201, 'n': 'minecraft:stone', 'd': 2.2},
            {'a': 'goto', 'x': 104, 'y': 64, 'z': -202, 'd': 4.0},
            {'a': 'craft', 'n': 'minecraft:stone_pickaxe', 'd': 0.0, 'val': 0.8},
            {'a': 'attack', 'eid': 7, 'n': 'minecraft:zombie', 'h': 1, 'tgt': 1, 'd': 1.44},
            {'a': 'wait', 'd': 0.0},
        ],
        'ev': [{'e': 'dig', 'm': 1, 'x': 102, 'y': 63, 'z': -201, 'n': 'minecraft:stone'}],
        'prev': {'a': 'goto', 'ticks': 2, 'ok': True},
    }


def main():
    print('== 1. encoding ==')
    fr = fake_frame()
    t0 = time.perf_counter()
    x = enc.encode(fr)
    dt_ms = (time.perf_counter() - t0) * 1000.0
    check('frame dim', x.shape == (C.FRAME_DIM,), (x.shape, C.FRAME_DIM))
    check('finite', bool(np.isfinite(x).all()))
    check('encode speed < 3ms', dt_ms < 5.0, f'{dt_ms:.2f}ms')
    x2 = enc.encode(fake_frame(px=1000.0, pz=-3000.0, yaw=120.0))
    check('translation invariance of grid', abs(float(x[C.SELF_DIM:C.SELF_DIM + 120].sum()) - float(x2[C.SELF_DIM:C.SELF_DIM + 120].sum())) < 1e-4)

    print('== 2. model (Dueling Double-DQN + RMSNorm) ==')
    net = M.Net(seed=3)
    n = net.nparams()
    check('param budget 2-6M', 2_000_000 < n < 6_000_000, n)
    L = net.embed(x[None, :])[0]
    check('latent shape', L.shape == (1, C.ENC), L.shape)

    print('== 3. training step & convergence ==')
    rng = np.random.RandomState(0)
    B_ = 16
    X0 = np.stack([enc.encode(fake_frame(seq=i, px=100.0 + i)) for i in range(B_)]).astype(np.float32)
    X1 = np.stack([enc.encode(fake_frame(seq=i + 1, px=100.0 + i)) for i in range(B_)]).astype(np.float32)
    Ls = np.repeat(net.infer_embed(X0)[:, None, :], C.T + 2, axis=1).astype(np.float32)
    Cm = np.zeros((B_, C.MAX_CAND, C.CAND_DIM), dtype=np.float32)
    c0, _ = enc.encode_candidates(fr['cand'], fr)
    Cm[:] = c0[None, :, :]
    mask = np.zeros((B_, C.MAX_CAND), dtype=np.float32)
    mask[:, :5] = 1.0
    batch = {
        'X0': X0, 'X1': X1, 'Ls': Ls, 'Cmat': Cm, 'mask': mask,
        'ai': rng.randint(0, 5, B_), 'rew': np.ones(B_, np.float32) * 0.5,
        'done': np.zeros(B_, np.float32), 'q_target_sel': np.zeros(B_, np.float32),
        'is_w': np.ones(B_, np.float32), 'n_step': np.full(B_, C.N_STEP, np.float32),
    }
    g, info = M.train_batch(net, batch)
    check('grads finite', all(np.isfinite(v).all() for v in g.values()))
    check('aux & q loss finite', np.isfinite(info['aux_loss']) and np.isfinite(info['q_loss']), info)
    q0, a0 = info['q_loss'], info['aux_loss']
    for _ in range(120):
        g, info = M.train_batch(net, batch)
        net.step(g, 1e-3)
    check('aux loss decreases', info['aux_loss'] < a0 * 0.6, f"{a0:.4f} -> {info['aux_loss']:.4f}")
    check('q loss decreases', info['q_loss'] < q0 * 0.6, f"{q0:.4f} -> {info['q_loss']:.4f}")

    print('== 4. PER replay round-trip ==')
    p = '/tmp/soma_smoke_v2.f16'
    for f in (p, p + '.hdr', p + '.prio'):
        if os.path.exists(f):
            os.remove(f)
    rp = B.Replay(path=p, n=64)
    for k in range(40):
        rp.append({
            'X0': x, 'X1': x2, 'Ls': Ls[0], 'Ln': Ls[0], 'Cm': Cm[0], 'Cn': Cm[0],
            'ai': k % 5, 'rw': 0.5, 'dn': False, 'nc': 5, 'ncn': 5, 'ns': C.N_STEP,
        })
    check('count == 40', rp.count == 40, rp.count)
    out, idx, is_w = rp.sample(rng, 8, alpha=0.6, beta=0.4)
    check('PER sample shapes', out.shape[0] == 8 and is_w.shape == (8,))
    rp.update_priorities(idx, np.arange(8, dtype=np.float32) + 0.5)
    check('roundtrip X0', np.allclose(out['X0'][0].astype(np.float32), x.astype(np.float16).astype(np.float32)))

    print('== 5. reward & brain end-to-end ==')
    rw = R.Reward()
    r1, inf1 = rw.step(fake_frame(seq=1, px=100.0), fake_frame(seq=2, px=101.0),
                       [{'e': 'craft', 'm': 1, 'n': 'minecraft:stone_pickaxe'}],
                       chosen_cand=fr['cand'][2], curiosity=0.2)
    check('craft milestone reward > 1.5', r1 > 1.5, r1)
    check('milestone recorded', 'stone_pickaxe' in rw.milestones)

    # Test Brain on_frame & N-step flushing
    C.REPLAY = '/tmp/soma_brain_smoke.f16'
    C.CKPT = '/tmp/soma_brain_smoke.npz'
    C.STATE = '/tmp/soma_brain_state.json'
    for f in (C.REPLAY, C.REPLAY + '.hdr', C.REPLAY + '.prio', C.CKPT, C.STATE):
        if os.path.exists(f):
            os.remove(f)
    br = BR.Brain()
    for step in range(10):
        ai = br.on_frame(fake_frame(seq=100 + step, px=100.0 + step * 0.5))
        check(f'brain seq={100+step} valid ai', 0 <= ai < 5) if step == 9 else None
    check('brain frame_seq updated', br.frame_seq == 109, br.frame_seq)
    check('nstep flushed to replay', br.buf.count >= 5, br.buf.count)

    print()
    if FAILS:
        print('FAILED:', FAILS)
        sys.exit(1)
    print('all v2 checks passed')


if __name__ == '__main__':
    main()