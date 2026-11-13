"""SOMA decision model v2.1 — Split-Stream Encoder + Multi-Scale Temporal + Dueling Double-DQN.

Upgrades over v2.0:
  * Split-stream encoder (3072 voxel -> 256, 708 core -> 256, fused -> 192 + RMSNorm):
    3x fewer FLOPs and prevents 3072 voxel cells from drowning out vitals/entities/inv.
  * Multi-scale temporal memory: short-term window (T=8) + medium/long EMA traces (2x192).
  * Deeper 2-layer Advantage stream for stronger state-candidate matching.
  * Dedicated zero-overhead inference methods for brain.py.
"""

import numpy as np

import config as C

LEAKY = 0.05
NORM_EPS = 1e-5
G_START = C.SELF_DIM
G_END = C.SELF_DIM + C.GRID_DIM
TMP_IN = (C.T + 2) * C.ENC


def _he(rng, fan_in, shape):
    return rng.normal(0.0, np.sqrt(2.0 / max(1, fan_in)), shape).astype(np.float32)


def _lrelu(z):
    return np.where(z > 0.0, z, LEAKY * z).astype(np.float32)


def _lrelu_grad(z):
    return np.where(z > 0.0, 1.0, LEAKY).astype(np.float32)


def _rmsnorm(x, g):
    ms = np.mean(x * x, axis=-1, keepdims=True)
    inv = 1.0 / np.sqrt(ms + NORM_EPS)
    xh = x * inv
    return (xh * g).astype(np.float32), xh, inv


def _rmsnorm_fast(x, g):
    inv = 1.0 / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + NORM_EPS)
    return (x * inv * g).astype(np.float32)


def _rmsnorm_grad(dy, xh, inv, g):
    D = xh.shape[-1]
    dg = np.sum(dy * xh, axis=0)
    dxh = dy * g
    dot = np.sum(dxh * xh, axis=-1, keepdims=True) / D
    dx = inv * (dxh - xh * dot)
    return dx.astype(np.float32), dg.astype(np.float32)


def _split_x(X):
    Xg = X[:, G_START:G_END]
    Xc = np.concatenate([X[:, :G_START], X[:, G_END:]], axis=1)
    return Xg, Xc


class Net:
    def __init__(self, seed=0):
        rng = np.random.RandomState(seed)
        p = {}
        # Split-stream Encoder
        p['Wg'] = _he(rng, C.GRID_DIM, (C.GRID_DIM, C.H_GRID)); p['bg'] = np.zeros(C.H_GRID, np.float32)
        p['Wc'] = _he(rng, C.CORE_DIM, (C.CORE_DIM, C.H_CORE)); p['bc'] = np.zeros(C.H_CORE, np.float32)
        p['W2'] = _he(rng, C.H_GRID + C.H_CORE, (C.H_GRID + C.H_CORE, C.ENC)); p['b2'] = np.zeros(C.ENC, np.float32)
        p['gn_e'] = np.ones(C.ENC, np.float32)

        # Multi-scale Temporal
        p['W3'] = _he(rng, TMP_IN, (TMP_IN, C.H2)); p['b3'] = np.zeros(C.H2, np.float32)
        p['W4'] = _he(rng, C.H2, (C.H2, C.ENC));  p['b4'] = np.zeros(C.ENC, np.float32)
        p['gn_t'] = np.ones(C.ENC, np.float32)

        # 2-layer Dueling Advantage stream A(S, c)
        p['W5'] = _he(rng, C.ENC + C.CAND_DIM, (C.ENC + C.CAND_DIM, C.HID)); p['b5'] = np.zeros(C.HID, np.float32)
        p['W5b'] = _he(rng, C.HID, (C.HID, 128)); p['b5b'] = np.zeros(128, np.float32)
        p['W6'] = _he(rng, 128, (128, 1));        p['b6'] = np.zeros(1, np.float32)

        # Dueling Value stream V(S)
        p['W7'] = _he(rng, C.ENC, (C.ENC, C.HID)); p['b7'] = np.zeros(C.HID, np.float32)
        p['W8'] = _he(rng, C.HID, (C.HID, 1));     p['b8'] = np.zeros(1, np.float32)

        # Action-conditioned Auxiliary Dynamics (L0, C_chosen) -> L1
        p['W10'] = _he(rng, C.ENC + C.CAND_DIM, (C.ENC + C.CAND_DIM, C.AUX_HID)); p['b10'] = np.zeros(C.AUX_HID, np.float32)
        p['W11'] = _he(rng, C.AUX_HID, (C.AUX_HID, C.ENC));                       p['b11'] = np.zeros(C.ENC, np.float32)

        self.p = p
        self.t = {k: v.copy() for k, v in p.items()}
        self.m = {k: np.zeros_like(v) for k, v in p.items()}
        self.v = {k: np.zeros_like(v) for k, v in p.items()}
        self.tstep = 0

    # -------------------------------------------------------- fast inference --
    def infer_embed(self, X):
        w = self.p
        Xg, Xc = _split_x(X)
        hg = _lrelu(Xg @ w['Wg'] + w['bg'])
        hc = _lrelu(Xc @ w['Wc'] + w['bc'])
        l_raw = np.concatenate([hg, hc], axis=1) @ w['W2'] + w['b2']
        return _rmsnorm_fast(l_raw, w['gn_e'])

    def infer_q(self, L_seq, Cmat, n):
        """L_seq: (1, T+2, ENC), Cmat: (1, MAX_CAND, CAND_DIM), n: valid count."""
        w = self.p
        X = L_seq.reshape(1, TMP_IN)
        ht = _lrelu(X @ w['W3'] + w['b3'])
        S = _rmsnorm_fast(ht @ w['W4'] + w['b4'], w['gn_t'])

        C_valid = Cmat[0, :n]  # (n, CAND_DIM)
        S_rep = np.repeat(S, n, axis=0)
        Xa = np.concatenate([S_rep, C_valid], axis=1)
        ha1 = _lrelu(Xa @ w['W5'] + w['b5'])
        ha2 = _lrelu(ha1 @ w['W5b'] + w['b5b'])
        a = (ha2 @ w['W6'] + w['b6'])[:, 0]

        hv = _lrelu(S @ w['W7'] + w['b7'])
        v = float((hv @ w['W8'] + w['b8'])[0, 0])
        return v + (a - np.mean(a))

    def infer_aux(self, L0, C_act):
        w = self.p
        h = _lrelu(np.concatenate([L0, C_act], axis=1) @ w['W10'] + w['b10'])
        return h @ w['W11'] + w['b11']

    # ---------------------------------------------------- training forward --
    def embed(self, X, use_target=False):
        w = self.t if use_target else self.p
        Xg, Xc = _split_x(X)
        zg = Xg @ w['Wg'] + w['bg']
        hg = _lrelu(zg)
        zc = Xc @ w['Wc'] + w['bc']
        hc = _lrelu(zc)
        h_cat = np.concatenate([hg, hc], axis=1)
        l_raw = h_cat @ w['W2'] + w['b2']
        ln, xh, inv = _rmsnorm(l_raw, w['gn_e'])
        return ln, (Xg, Xc, zg, hg, zc, hc, h_cat, xh, inv)

    def temporal(self, L_seq, use_target=False):
        w = self.t if use_target else self.p
        B = L_seq.shape[0]
        X = L_seq.reshape(B, TMP_IN)
        z = X @ w['W3'] + w['b3']
        h = _lrelu(z)
        s_raw = h @ w['W4'] + w['b4']
        sn, xh, inv = _rmsnorm(s_raw, w['gn_t'])
        return sn, z, h, X, xh, inv

    def advantage(self, S, Cmat, use_target=False):
        w = self.t if use_target else self.p
        B, M, _ = Cmat.shape
        X = np.concatenate([np.repeat(S[:, None, :], M, axis=1), Cmat], axis=2).reshape(B * M, C.ENC + C.CAND_DIM)
        z1 = X @ w['W5'] + w['b5']
        h1 = _lrelu(z1)
        z2 = h1 @ w['W5b'] + w['b5b']
        h2 = _lrelu(z2)
        a = (h2 @ w['W6'] + w['b6']).reshape(B, M)
        return a, z1, h1, z2, h2, X

    def value(self, S, use_target=False):
        w = self.t if use_target else self.p
        z = S @ w['W7'] + w['b7']
        h = _lrelu(z)
        v = (h @ w['W8'] + w['b8'])[:, 0]
        return v, z, h

    def q_all(self, S, Cmat, mask=None, use_target=False):
        a, za1, ha1, za2, ha2, Xa = self.advantage(S, Cmat, use_target=use_target)
        v, zv, hv = self.value(S, use_target=use_target)
        if mask is not None:
            cnt = np.maximum(mask.sum(axis=1, keepdims=True), 1.0)
            a_mean = (a * mask).sum(axis=1, keepdims=True) / cnt
        else:
            a_mean = a.mean(axis=1, keepdims=True)
        q = v[:, None] + (a - a_mean)
        return q, (v, zv, hv, a, za1, ha1, za2, ha2, Xa, a_mean)

    def aux(self, L0, C_act):
        X = np.concatenate([L0, C_act], axis=1)
        z = X @ self.p['W10'] + self.p['b10']
        h = _lrelu(z)
        return h @ self.p['W11'] + self.p['b11'], z, h, X

    # --------------------------------------------------------------- adam ---
    def step(self, grads, lr):
        sq_norm = 0.0
        for g in grads.values():
            if g is not None:
                sq_norm += float(np.sum(g * g))
        gnorm = float(np.sqrt(sq_norm))
        scale = min(1.0, C.GRAD_CLIP / max(gnorm, 1e-8))

        self.tstep += 1
        b1, b2, eps = C.ADAM['b1'], C.ADAM['b2'], C.ADAM['eps']
        c1 = 1.0 - b1 ** self.tstep
        c2 = 1.0 - b2 ** self.tstep
        lr_eff = lr / c1
        inv_c2 = 1.0 / c2
        for k, g in grads.items():
            if g is None:
                continue
            gs = g * scale
            self.m[k] = b1 * self.m[k] + (1.0 - b1) * gs
            self.v[k] = b2 * self.v[k] + (1.0 - b2) * (gs * gs)
            self.p[k] -= (lr_eff * self.m[k] / (np.sqrt(self.v[k] * inv_c2) + eps)).astype(np.float32)
        return gnorm

    def sync_target(self):
        for k in self.t:
            self.t[k] = self.p[k].copy()

    def soft_sync_target(self, tau=C.TAU):
        omt = 1.0 - tau
        for k in self.t:
            self.t[k] = (omt * self.t[k] + tau * self.p[k]).astype(np.float32)

    # ----------------------------------------------------------- (de)serial --
    def save(self, path):
        np.savez(path, **self.p)

    def load(self, path):
        z = np.load(path)
        for k in self.p:
            if k in z and z[k].shape == self.p[k].shape:
                self.p[k] = z[k].astype(np.float32)
                self.t[k] = self.p[k].copy()

    def nparams(self):
        return int(sum(v.size for v in self.p.values()))


def train_batch(net: Net, batch: dict, do_encoder=True):
    X0, X1 = batch['X0'], batch['X1']
    Ls, Cmat, mask = batch['Ls'], batch['Cmat'], batch['mask']
    ai, rew, done = batch['ai'], batch['rew'], batch['done']
    is_w = batch.get('is_w')
    B, M = Cmat.shape[0], Cmat.shape[1]
    ar = np.arange(B)
    if is_w is None:
        is_w = np.ones(B, dtype=np.float32)

    # 1. Encoder forward
    L0, enc_cache = net.embed(X0)
    L1 = net.infer_embed(X1)

    # 2. Temporal + Dueling Q forward
    S, zt, ht, Xt, xhs, invs = net.temporal(Ls)
    q, (v, zv, hv, a, za1, ha1, za2, ha2, Xa, _) = net.q_all(S, Cmat, mask=mask)

    q_next_sel = batch['q_target_sel']
    gamma_n = C.GAMMA ** batch.get('n_step', C.N_STEP)
    target = (rew + (1.0 - done) * gamma_n * q_next_sel).astype(np.float32)

    qsel = q[ar, ai]
    td = qsel - target
    huber_g = np.clip(td, -1.0, 1.0) * is_w * (1.0 / B)
    dq = np.zeros_like(q)
    dq[ar, ai] = huber_g * mask[ar, ai]

    # Dueling backward
    dv_from_q = dq.sum(axis=1)
    cnt = np.maximum(mask.sum(axis=1, keepdims=True), 1.0)
    da = dq - mask * (dq.sum(axis=1, keepdims=True) / cnt)

    # 2-layer Advantage backward
    dW6 = ha2.T @ da.reshape(-1, 1)
    db6 = np.array([da.sum()], dtype=np.float32)
    dha2 = da.reshape(-1, 1) @ net.p['W6'].T
    dza2 = dha2 * _lrelu_grad(za2)
    dW5b = ha1.T @ dza2
    db5b = dza2.sum(axis=0)
    dha1 = dza2 @ net.p['W5b'].T
    dza1 = dha1 * _lrelu_grad(za1)
    dW5 = Xa.T @ dza1
    db5 = dza1.sum(axis=0)
    dXa = dza1 @ net.p['W5'].T
    dS_from_a = dXa.reshape(B, M, C.ENC + C.CAND_DIM)[:, :, :C.ENC].sum(axis=1)

    # Value backward
    v_loss_vec = (v - target)
    dv = dv_from_q + (C.V_W / B) * is_w * np.clip(v_loss_vec, -1.0, 1.0)
    dW8 = hv.T @ dv.reshape(-1, 1)
    db8 = np.array([dv.sum()], dtype=np.float32)
    dhv = dv.reshape(-1, 1) @ net.p['W8'].T
    dzv = dhv * _lrelu_grad(zv)
    dW7 = S.T @ dzv
    db7 = dzv.sum(axis=0)
    dS_from_v = dzv @ net.p['W7'].T

    # Temporal backward
    dS = dS_from_a + dS_from_v
    dS_raw, dgn_t = _rmsnorm_grad(dS, xhs, invs, net.p['gn_t'])
    dW4 = ht.T @ dS_raw
    db4 = dS_raw.sum(axis=0)
    dht = dS_raw @ net.p['W4'].T
    dzt = dht * _lrelu_grad(zt)
    dW3 = Xt.T @ dzt
    db3 = dzt.sum(axis=0)

    # 3. Auxiliary Dynamics forward & backward
    C_chosen = Cmat[ar, ai]
    y, zax, hax, Xax = net.aux(L0, C_chosen)
    dy = ((2.0 * C.AUX_W) / (B * C.ENC)) * (y - L1)
    dW11 = hax.T @ dy
    db11 = dy.sum(axis=0)
    dhax = dy @ net.p['W11'].T
    dzax = dhax * _lrelu_grad(zax)
    dW10 = Xax.T @ dzax
    db10 = dzax.sum(axis=0)
    dXax = dzax @ net.p['W10'].T
    dL0 = dXax[:, :C.ENC]

    grads = {
        'W3': dW3, 'b3': db3, 'W4': dW4, 'b4': db4, 'gn_t': dgn_t,
        'W5': dW5, 'b5': db5, 'W5b': dW5b, 'b5b': db5b, 'W6': dW6, 'b6': db6,
        'W7': dW7, 'b7': db7, 'W8': dW8, 'b8': db8,
        'W10': dW10, 'b10': db10, 'W11': dW11, 'b11': db11,
    }

    if do_encoder:
        Xg, Xc, zg, hg, zc, hc, h_cat, xh0, inv0 = enc_cache
        dL0_raw, dgn_e = _rmsnorm_grad(dL0, xh0, inv0, net.p['gn_e'])
        dW2 = h_cat.T @ dL0_raw
        db2 = dL0_raw.sum(axis=0)
        dh_cat = dL0_raw @ net.p['W2'].T
        dhg = dh_cat[:, :C.H_GRID]
        dhc = dh_cat[:, C.H_GRID:]
        dzg = dhg * _lrelu_grad(zg)
        dzc = dhc * _lrelu_grad(zc)
        dWg = Xg.T @ dzg
        dbg = dzg.sum(axis=0)
        dWc = Xc.T @ dzc
        dbc = dzc.sum(axis=0)
        grads.update({
            'Wg': dWg, 'bg': dbg, 'Wc': dWc, 'bc': dbc,
            'W2': dW2, 'b2': db2, 'gn_e': dgn_e,
        })

    huber_loss = np.where(np.abs(td) <= 1.0, 0.5 * td * td, np.abs(td) - 0.5)
    info = {
        'q_loss': float(np.mean(huber_loss * is_w)),
        'v_loss': float(np.mean(v_loss_vec * v_loss_vec)),
        'aux_loss': float(np.mean((y - L1) ** 2)),
        'td': float(np.mean(np.abs(td))),
        'td_vec': np.abs(td),
        'q': float(np.mean(qsel)),
    }
    return grads, info