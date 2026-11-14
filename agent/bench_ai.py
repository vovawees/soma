"""Comprehensive AI benchmark & verification suite for SOMA v3.

Tests:
  1. Numerical gradient exactness across all 32 parameter tensors
  2. Real-time inference latency (encode + brain.on_frame) & training throughput
  3. EpisodicMemoryBank cosine deduplication & retrieval
  4. Multi-scenario Minecraft survival learning (combat, eat, flee, craft, place torch)
"""

import os
import sys
import time
import tempfile
import shutil

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config as C
import enc
import model as M
import brain as BR


def test_gradients():
    print('== 1. numerical gradient check ==')
    rng = np.random.RandomState(42)
    net = M.Net(seed=0)
    B = 4
    batch = {
        'X0': rng.normal(0, 0.5, (B, C.FRAME_DIM)).astype(np.float32),
        'X1': rng.normal(0, 0.5, (B, C.FRAME_DIM)).astype(np.float32),
        'Ls': rng.normal(0, 0.5, (B, C.TOKENS, C.ENC)).astype(np.float32),
        'Cmat': rng.normal(0, 0.5, (B, C.MAX_CAND, C.CAND_DIM)).astype(np.float32),
        'mask': np.zeros((B, C.MAX_CAND), dtype=np.float32),
        'ai': np.array([0, 2, 5, 1], dtype=np.int64),
        'rew': rng.normal(0, 1.0, (B,)).astype(np.float32),
        'done': np.array([0, 0, 1, 0], dtype=np.float32),
        'q_target_sel': rng.normal(0, 1.0, (B,)).astype(np.float32),
        'is_w': rng.uniform(0.5, 1.5, (B,)).astype(np.float32),
    }
    for i in range(B):
        batch['mask'][i, :10] = 1.0

    old_munch = C.MUNCH_ALPHA
    C.MUNCH_ALPHA = 0.0  # test exact analytical gradient of Q + V + Aux loss
    try:
        L1_frozen = net.infer_embed(batch['X1'], use_target=True).copy()

        def loss_fn():
            L0, _ = net.embed(batch['X0'])
            S, _ = net.temporal(batch['Ls'])
            q, q_cache = net.q_all(S, batch['Cmat'], mask=batch['mask'])
            v = q_cache[0]
            ar = np.arange(B)
            gamma_n = C.GAMMA ** C.N_STEP
            target = (batch['rew'] + (1.0 - batch['done']) * gamma_n * batch['q_target_sel']).astype(np.float32)
            td = q[ar, batch['ai']] - target
            huber_q = np.where(np.abs(td) <= 1.0, 0.5 * td * td, np.abs(td) - 0.5)
            vd = v - target
            huber_v = np.where(np.abs(vd) <= 1.0, 0.5 * vd * vd, np.abs(vd) - 0.5)
            l_pred, r_pred, *_ = net.aux(L0, batch['Cmat'][ar, batch['ai']])
            r_target = np.clip(batch['rew'] / 3.0, -2.0, 2.0)
            return float(
                np.mean(huber_q * batch['is_w'])
                + C.V_W * np.mean(huber_v * batch['is_w'])
                + C.AUX_W * np.mean((l_pred - L1_frozen) ** 2)
                + C.REW_AUX_W * np.mean((r_pred - r_target) ** 2)
            )

        grads, _ = M.train_batch(net, batch, do_encoder=True)
        eps = 2e-3
        max_abs_all = 0.0
        for k in sorted(grads.keys()):
            p = net.p[k]
            g = grads[k]
            for _ in range(4):
                idx = tuple(rng.randint(0, s) for s in p.shape)
                old = p[idx]
                p[idx] = old + eps
                lp = loss_fn()
                p[idx] = old - eps
                lm = loss_fn()
                p[idx] = old
                num = (lp - lm) / (2 * eps)
                ana = float(g[idx])
                max_abs_all = max(max_abs_all, abs(num - ana))
        assert max_abs_all < 5e-4, f'gradient mismatch: max_abs={max_abs_all}'
        print(f'  ok   all 32 param tensors exact (max_abs={max_abs_all:.2e})')
    finally:
        C.MUNCH_ALPHA = old_munch


def test_episodic_memory():
    print('== 2. episodic memory deduplication & retrieval ==')
    mem = BR.EpisodicMemoryBank(cap=16, dim=C.ENC)
    rng = np.random.RandomState(3)
    base = rng.normal(0, 1, C.ENC).astype(np.float32)
    # Feed 40 near-identical frames -> should merge into 1 slot
    for i in range(40):
        k = base + rng.normal(0, 0.02, C.ENC).astype(np.float32)
        mem.store(k, k * 2.0, importance=1.0 + i * 0.01)
    assert mem.count <= 2, f'expected deduplication, got count={mem.count}'

    # Feed 8 orthogonal experiences -> should all be stored
    distinct = []
    for i in range(8):
        k = rng.normal(0, 1, C.ENC).astype(np.float32)
        v = rng.normal(0, 1, C.ENC).astype(np.float32)
        mem.store(k, v, importance=2.0)
        distinct.append((k, v))
    assert mem.count >= 8, f'expected distinct slots stored, got {mem.count}'

    q = distinct[3][0]
    ret = mem.query(q)
    cos = float(np.dot(ret, distinct[3][1]) / (np.linalg.norm(ret) * np.linalg.norm(distinct[3][1]) + 1e-8))
    assert cos > 0.9, f'expected high retrieval similarity, got {cos:.3f}'
    print(f'  ok   dedup count={mem.count}/16, retrieval cos={cos:.3f}')


def _make_scenario_frame(rng, scenario, seq):
    px, py, pz = 100.0 + rng.uniform(-5, 5), 64.0, 200.0 + rng.uniform(-5, 5)
    yaw = float(rng.uniform(-180, 180))
    hp, food, light, lava = 20.0, 20.0, 14, 0
    inv = [[0, 'minecraft:stick', 2]]
    ents = []
    cands = [
        {'a': 'move', 'x': px + 2, 'y': py, 'z': pz, 'dx': 2, 'dz': 0, 'd': 2.0, 'h': 0},
        {'a': 'look', 'yaw': yaw + 30, 'pitch': 0.0},
        {'a': 'dig', 'n': 'minecraft:stone', 'x': px + 1, 'y': py, 'z': pz, 'd': 1.5, 'h': 1.5},
    ]

    if scenario == 0:  # Combat: hostile nearby -> attack
        ents.append([10, 'minecraft:zombie', 1.5, 0.0, 1.5, 2.1, 20.0, 20.0, 1, 1, 0.0, 0.0])
        cands.append({'a': 'attack', 'eid': 10, 'n': 'minecraft:zombie', 'dx': 1.5, 'dy': 0.0, 'dz': 1.5, 'd': 2.1, 'h': 1.0})
        opt_action = 'attack'
    elif scenario == 1:  # Hunger: low hp & low food -> eat
        hp, food = 7.0, 5.0
        inv.append([1, 'minecraft:cooked_beef', 4])
        cands.append({'a': 'eat', 'slot': 1, 'n': 'minecraft:cooked_beef'})
        opt_action = 'eat'
    elif scenario == 2:  # Danger: critical hp + hostiles + lava -> flee
        hp, lava = 3.5, 1
        ents.append([11, 'minecraft:skeleton', 2.0, 0.0, 1.0, 2.2, 20.0, 20.0, 1, 1, 0.0, 0.0])
        ents.append([12, 'minecraft:creeper', -1.5, 0.0, 2.0, 2.5, 20.0, 20.0, 1, 1, 0.0, 0.0])
        cands.append({'a': 'attack', 'eid': 11, 'n': 'minecraft:skeleton', 'dx': 2.0, 'dy': 0.0, 'dz': 1.0, 'd': 2.2, 'h': 1.0})
        cands.append({'a': 'flee', 'x': px - 6, 'y': py, 'z': pz - 6, 'dx': -6, 'dz': -6, 'd': 8.5})
        opt_action = 'flee'
    elif scenario == 3:  # Crafting: logs in inventory -> craft
        inv.append([2, 'minecraft:oak_log', 4])
        cands.append({'a': 'craft', 'rid': 'minecraft:oak_planks', 'n': 'minecraft:oak_planks'})
        opt_action = 'craft'
    else:  # Dark cave with torch -> place torch
        light = 2
        inv.append([3, 'minecraft:torch', 8])
        cands.append({'a': 'place', 'n': 'minecraft:torch', 'x': px + 1, 'y': py, 'z': pz, 'd': 1.4})
        opt_action = 'place'

    fr = {
        'seq': seq,
        'self': {
            'x': px, 'y': py, 'z': pz, 'yaw': yaw, 'pitch': 0.0,
            'hp': hp, 'food': food, 'light': light, 'lava_near': lava,
            'ground': 1, 'alive': 1, 'hurt': 1 if scenario == 2 else 0,
            'day': 0.7 if scenario == 4 else 0.2,
        },
        'grid': {},
        'ent': ents,
        'inv': inv,
        'ev': [],
        'cand': cands,
    }
    return fr, opt_action


def test_speed_and_learning():
    print('== 3. inference latency & survival learning benchmark ==')
    rng = np.random.RandomState(123)
    net = M.Net(seed=7)

    # Pre-build a pool of encoded scenario transitions
    X0_list, X1_list, Cm_list, mask_list, nc_list, ai_list, rw_list = [], [], [], [], [], [], []
    test_samples = []
    for i in range(512):
        sc = i % 5
        fr, opt = _make_scenario_frame(rng, sc, i)
        x = enc.encode(fr)
        cm, nc = enc.encode_candidates(fr['cand'], fr)
        opt_idx = next(idx for idx, c in enumerate(fr['cand']) if c['a'] == opt)
        if i < 400:
            # Mix optimal and suboptimal actions in replay
            if i % 2 == 0:
                ai = opt_idx
                rw = 1.5
            else:
                bad = [idx for idx in range(nc) if idx != opt_idx]
                ai = int(rng.choice(bad))
                rw = -0.5
            m = np.zeros(C.MAX_CAND, dtype=np.float32)
            m[:nc] = 1.0
            X0_list.append(x)
            X1_list.append(x)  # terminal single-step transition for scenario discrimination
            Cm_list.append(cm)
            mask_list.append(m)
            nc_list.append(nc)
            ai_list.append(ai)
            rw_list.append(rw)
        else:
            test_samples.append((x, cm, nc, opt_idx))

    X0_arr = np.stack(X0_list)
    X1_arr = np.stack(X1_list)
    Cm_arr = np.stack(Cm_list)
    mask_arr = np.stack(mask_list)
    ai_arr = np.array(ai_list, dtype=np.int64)
    rw_arr = np.array(rw_list, dtype=np.float32)

    # Measure inference latency (encode + embed + infer_q)
    fr0, _ = _make_scenario_frame(rng, 0, 1)
    t0 = time.perf_counter()
    for _ in range(100):
        x0 = enc.encode(fr0)
        cm0, nc0 = enc.encode_candidates(fr0['cand'], fr0)
        l0 = net.infer_embed(x0[None, :])[0]
        win = np.repeat(l0[None, :], C.TOKENS, axis=0)
        net.infer_q(win[None, ...], cm0[None, ...], nc0)
    infer_ms = (time.perf_counter() - t0) / 100 * 1000
    assert infer_ms < 5.0, f'inference too slow: {infer_ms:.2f}ms'
    print(f'  ok   full frame inference < 5ms ({infer_ms:.2f}ms)')

    # Train for 250 SGD steps
    B = C.BATCH
    N_pool = len(X0_arr)
    t_train_start = time.perf_counter()
    first_loss = None
    last_loss = None
    for step in range(250):
        idx = rng.randint(0, N_pool, size=B)
        L0_b = net.infer_embed(X0_arr[idx])
        Ls_b = np.repeat(L0_b[:, None, :], C.TOKENS, axis=1)
        batch = {
            'X0': X0_arr[idx],
            'X1': X1_arr[idx],
            'Ls': Ls_b,
            'Cmat': Cm_arr[idx],
            'mask': mask_arr[idx],
            'ai': ai_arr[idx],
            'rew': rw_arr[idx],
            'done': np.ones(B, dtype=np.float32),
            'q_target_sel': np.zeros(B, dtype=np.float32),
            'is_w': np.ones(B, dtype=np.float32),
        }
        grads, info = M.train_batch(net, batch)
        net.step(grads, C.LR)
        net.soft_sync_target(C.TAU)
        if first_loss is None:
            first_loss = info['q_loss']
        last_loss = info['q_loss']

    step_ms = (time.perf_counter() - t_train_start) / 250 * 1000

    # Evaluate on held-out test scenarios
    correct = 0
    q_gaps = []
    for x, cm, nc, opt_idx in test_samples:
        l = net.infer_embed(x[None, :])[0]
        win = np.repeat(l[None, :], C.TOKENS, axis=0)
        q = net.infer_q(win[None, ...], cm[None, ...], nc)
        pred = int(np.argmax(q))
        if pred == opt_idx:
            correct += 1
        other = np.delete(q, opt_idx)
        q_gaps.append(float(q[opt_idx] - np.max(other)))

    acc = correct / len(test_samples)
    mean_gap = float(np.mean(q_gaps))
    assert acc >= 0.90, f'held-out survival accuracy too low: {acc:.2%}'
    assert mean_gap > 0.30, f'Q-gap too small: {mean_gap:.3f}'
    print(f'  ok   250 SGD steps ({step_ms:.1f}ms/step): q_loss {first_loss:.4f} -> {last_loss:.4f}')
    print(f'  ok   held-out survival accuracy={acc:.1%} ({correct}/{len(test_samples)}), mean Q-gap={mean_gap:.3f}')


def test_goal_policy():
    print('== 4. goal generator & non-random survival policy ==')
    mem = BR.SpatialMemory()
    gg = BR.GoalGenerator()

    scenarios = [
        # (self_dict, inv, ent, cands, expected_goal, expected_actions)
        ({'hp': 20, 'food': 8, 'has_food': 1}, [[0, 'minecraft:bread', 3]], [],
         [{'a': 'wait'}, {'a': 'goto', 'x': 5, 'y': 64, 'z': 0, 'd': 5}, {'a': 'eat', 'val': 1.0}],
         'eat_food', {'eat'}),
        ({'hp': 20, 'food': 20, 'has_bow': 1}, [[0, 'minecraft:bow', 1]],
         [[9, 'minecraft:zombie', 8, 0, 0, 8.0, 20, 20, 1, 1, 0, 0]],
         [{'a': 'wait'}, {'a': 'goto', 'x': 3, 'y': 64, 'z': 0, 'd': 3}, {'a': 'shoot', 'eid': 9, 'h': 1, 'val': 0.9}],
         'fight_hostile', {'shoot'}),
        ({'hp': 20, 'food': 20}, [[0, 'minecraft:oak_log', 4]], [],
         [{'a': 'wait'}, {'a': 'goto', 'x': 4, 'y': 64, 'z': 0, 'd': 4},
          {'a': 'craft', 'rid': 'minecraft:oak_planks', 'n': 'minecraft:oak_planks', 'shift': 1, 'cnt': 4, 'val': 0.93}],
         'craft_basics', {'craft'}),
        ({'hp': 20, 'food': 20, 'near_craft': 0}, [[0, 'minecraft:crafting_table', 1]], [],
         [{'a': 'wait'}, {'a': 'place', 'x': 1, 'y': 64, 'z': 0, 'n': 'minecraft:crafting_table', 'd': 1.5, 'val': 0.95}],
         'place_table', {'place'}),
        ({'hp': 5, 'food': 20}, [[0, 'minecraft:cobblestone', 8]],
         [[11, 'minecraft:zombie', 2, 0, 0, 2.0, 20, 20, 1, 1, 0, 0]],
         [{'a': 'wait'}, {'a': 'pillar', 'n': 'minecraft:cobblestone', 'd': 0.0, 'val': 0.82}],
         'survive_danger', {'pillar'}),
    ]

    for idx, (s, inv, ent, cands, exp_goal, exp_acts) in enumerate(scenarios):
        fr = {'seq': idx + 1, 'self': {'x': 0, 'y': 64, 'z': 0, 'yaw': 0, 'pitch': 0, **s}, 'grid': {}, 'ent': ent, 'inv': inv, 'ev': [], 'cand': cands}
        g = gg.update(fr, mem)
        assert g == exp_goal, f'scenario {idx}: expected goal {exp_goal}, got {g}'
        scores = [gg.score_candidate(c, fr) for c in cands]
        best_act = cands[int(np.argmax(scores))]['a']
        assert best_act in exp_acts, f'scenario {idx}: expected {exp_acts}, got {best_act}'
    print(f'  ok   all {len(scenarios)} survival/crafting/combat/building goal scenarios passed')


if __name__ == '__main__':
    test_gradients()
    test_episodic_memory()
    test_speed_and_learning()
    test_goal_policy()
    print('\nall AI benchmarks passed!')
