"""SOMA decision model v3 — Transformer + SwiGLU + Episodic Memory + World Model.

Architecture:
  1. Split-stream Encoder: (3072 voxel -> 256) + (708 core -> 256) -> 192 + RMSNorm
  2. Context Transformer (11 tokens x 192):
       Tokens = [L_{t-7}..L_t (8), EMA_med, EMA_slow, Mem_retrieved]
       + pos_emb -> Pre-RMSNorm Multi-Head Attention (4 heads) + Residual
       -> Pre-RMSNorm SwiGLU FFN + Residual -> State S (192)
  3. Dueling Double-DQN Heads (SiLU): V(S) + (A(S, c) - mean_valid(A))
  4. Dual World-Model Aux Head: (L0, C_chosen) -> (L1_pred, r_pred)
"""

import numpy as np

import config as C

NORM_EPS = 1e-5
G_START = C.SELF_DIM
G_END = C.SELF_DIM + C.GRID_DIM
N_TOK = C.TOKENS
D_MODEL = C.ENC
N_H = C.N_HEADS
D_H = D_MODEL // N_H
INV_SQRT_DH = np.float32(1.0 / np.sqrt(D_H))


def _he(rng, fan_in, shape):
    return rng.normal(0.0, np.sqrt(2.0 / max(1, fan_in)), shape).astype(np.float32)


_ONE = np.float32(1.0)
_CLIP_LO = np.float32(-20.0)
_CLIP_HI = np.float32(20.0)
_EPS32 = np.float32(NORM_EPS)


def _silu(x):
    xc = np.clip(x, _CLIP_LO, _CLIP_HI)
    sig = _ONE / (_ONE + np.exp(-xc))
    return x * sig, sig


def _silu_fast(x):
    xc = np.clip(x, _CLIP_LO, _CLIP_HI)
    return x / (_ONE + np.exp(-xc))


def _silu_grad(x, sig):
    return sig * (_ONE + x * (_ONE - sig))


def _rmsnorm(x, g):
    ms = np.mean(x * x, axis=-1, keepdims=True)
    inv = _ONE / np.sqrt(ms + _EPS32)
    xh = x * inv
    return xh * g, xh, inv


def _rmsnorm_fast(x, g):
    inv = _ONE / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + _EPS32)
    return x * inv * g


def _rmsnorm_grad(dy, xh, inv, g):
    D = np.float32(xh.shape[-1])
    dg = np.sum(dy * xh, axis=tuple(range(dy.ndim - 1)))
    dxh = dy * g
    dot = np.sum(dxh * xh, axis=-1, keepdims=True) / D
    dx = inv * (dxh - xh * dot)
    return dx, dg


def _softmax(x):
    xm = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(xm)
    return e / np.sum(e, axis=-1, keepdims=True)


def _split_x(X):
    return X[:, G_START:G_END], np.concatenate([X[:, :G_START], X[:, G_END:]], axis=1)


class Net:
    def __init__(self, seed=0):
        rng = np.random.RandomState(seed)
        p = {}
        # 1. Split-stream Encoder
        p['Wg'] = _he(rng, C.GRID_DIM, (C.GRID_DIM, C.H_GRID)); p['bg'] = np.zeros(C.H_GRID, np.float32)
        p['Wc'] = _he(rng, C.CORE_DIM, (C.CORE_DIM, C.H_CORE)); p['bc'] = np.zeros(C.H_CORE, np.float32)
        p['W2'] = _he(rng, C.H_GRID + C.H_CORE, (C.H_GRID + C.H_CORE, D_MODEL)); p['b2'] = np.zeros(D_MODEL, np.float32)
        p['gn_e'] = np.ones(D_MODEL, np.float32)

        # 2. Transformer Block over N_TOK=11 tokens
        p['pos'] = (rng.normal(0.0, 0.02, (1, N_TOK, D_MODEL))).astype(np.float32)
        p['gn_a'] = np.ones(D_MODEL, np.float32)
        p['Wqkv'] = _he(rng, D_MODEL, (D_MODEL, 3 * D_MODEL))
        p['Wo'] = _he(rng, D_MODEL, (D_MODEL, D_MODEL))
        p['gn_f'] = np.ones(D_MODEL, np.float32)
        p['Wfg'] = _he(rng, D_MODEL, (D_MODEL, C.FFN_HID))
        p['Wfu'] = _he(rng, D_MODEL, (D_MODEL, C.FFN_HID))
        p['Wfd'] = _he(rng, C.FFN_HID, (C.FFN_HID, D_MODEL))
        p['gn_t'] = np.ones(D_MODEL, np.float32)

        # 3. Dueling Advantage A(S, c) & Value V(S)
        p['W5'] = _he(rng, D_MODEL + C.CAND_DIM, (D_MODEL + C.CAND_DIM, C.HID)); p['b5'] = np.zeros(C.HID, np.float32)
        p['W5b'] = _he(rng, C.HID, (C.HID, 128)); p['b5b'] = np.zeros(128, np.float32)
        p['W6'] = _he(rng, 128, (128, 1));        p['b6'] = np.zeros(1, np.float32)

        p['W7'] = _he(rng, D_MODEL, (D_MODEL, C.HID)); p['b7'] = np.zeros(C.HID, np.float32)
        p['W8'] = _he(rng, C.HID, (C.HID, 1));         p['b8'] = np.zeros(1, np.float32)

        # 4. World-Model Aux Head: (L0, C_chosen) -> (L1_pred, r_pred)
        p['W10'] = _he(rng, D_MODEL + C.CAND_DIM, (D_MODEL + C.CAND_DIM, C.AUX_HID)); p['b10'] = np.zeros(C.AUX_HID, np.float32)
        p['W11'] = _he(rng, C.AUX_HID, (C.AUX_HID, D_MODEL)); p['b11'] = np.zeros(D_MODEL, np.float32)
        p['W12'] = _he(rng, C.AUX_HID, (C.AUX_HID, 1));       p['b12'] = np.zeros(1, np.float32)

        self.p = p
        self.t = {k: v.copy() for k, v in p.items()}
        self.m = {k: np.zeros_like(v) for k, v in p.items()}
        self.v = {k: np.zeros_like(v) for k, v in p.items()}
        self.tstep = 0

    # -------------------------------------------------------- fast inference --
    def infer_embed(self, X, use_target=False):
        w = self.t if use_target else self.p
        Xg, Xc = _split_x(X)
        hg = _silu_fast(Xg @ w['Wg'] + w['bg'])
        hc = _silu_fast(Xc @ w['Wc'] + w['bc'])
        l_raw = np.concatenate([hg, hc], axis=1) @ w['W2'] + w['b2']
        return _rmsnorm_fast(l_raw, w['gn_e'])

    def _transformer_fast(self, L_seq, w):
        B = L_seq.shape[0]
        BN = B * N_TOK
        x = L_seq + w['pos']
        xn = _rmsnorm_fast(x, w['gn_a']).reshape(BN, D_MODEL)
        qkv = (xn @ w['Wqkv']).reshape(B, N_TOK, 3, N_H, D_H).transpose(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        att = _softmax((q @ k.transpose(0, 1, 3, 2)) * INV_SQRT_DH)
        out_a = (att @ v).transpose(0, 2, 1, 3).reshape(BN, D_MODEL)
        x = x + (out_a @ w['Wo']).reshape(B, N_TOK, D_MODEL)
        xf = _rmsnorm_fast(x, w['gn_f']).reshape(BN, D_MODEL)
        ff = ((_silu_fast(xf @ w['Wfg']) * (xf @ w['Wfu'])) @ w['Wfd']).reshape(B, N_TOK, D_MODEL)
        x = x + ff
        s_raw = np.float32(0.5) * x[:, C.T - 1, :] + np.float32(0.5) * np.mean(x, axis=1)
        return _rmsnorm_fast(s_raw, w['gn_t'])

    def infer_q(self, L_seq, Cmat, n):
        w = self.p
        S = self._transformer_fast(L_seq, w)
        C_valid = Cmat[0, :n]
        z_s = S @ w['W5'][:D_MODEL]
        z_c = C_valid @ w['W5'][D_MODEL:] + w['b5']
        ha1 = _silu_fast(z_s + z_c)
        ha2 = _silu_fast(ha1 @ w['W5b'] + w['b5b'])
        a = (ha2 @ w['W6'] + w['b6'])[:, 0]
        hv = _silu_fast(S @ w['W7'] + w['b7'])
        v = float((hv @ w['W8'] + w['b8'])[0, 0])
        return v + (a - np.mean(a))

    def infer_aux(self, L0, C_act):
        w = self.p
        h = _silu_fast(np.concatenate([L0, C_act], axis=1) @ w['W10'] + w['b10'])
        return h @ w['W11'] + w['b11']

    # ---------------------------------------------------- training forward --
    def embed(self, X):
        w = self.p
        Xg, Xc = _split_x(X)
        zg = Xg @ w['Wg'] + w['bg']
        hg, sg = _silu(zg)
        zc = Xc @ w['Wc'] + w['bc']
        hc, sc = _silu(zc)
        h_cat = np.concatenate([hg, hc], axis=1)
        l_raw = h_cat @ w['W2'] + w['b2']
        ln, xh, inv = _rmsnorm(l_raw, w['gn_e'])
        return ln, (Xg, Xc, zg, sg, zc, sc, h_cat, xh, inv)

    def temporal(self, L_seq, use_target=False):
        w = self.t if use_target else self.p
        if use_target:
            return self._transformer_fast(L_seq, w), None

        B = L_seq.shape[0]
        BN = B * N_TOK
        x0 = L_seq + w['pos']
        xn_3d, xha, inva = _rmsnorm(x0, w['gn_a'])
        xn = xn_3d.reshape(BN, D_MODEL)
        qkv = xn @ w['Wqkv']
        qkv_r = qkv.reshape(B, N_TOK, 3, N_H, D_H).transpose(2, 0, 3, 1, 4)
        q, k, v = qkv_r[0], qkv_r[1], qkv_r[2]
        scores = (q @ k.transpose(0, 1, 3, 2)) * INV_SQRT_DH
        att = _softmax(scores)
        ctx = (att @ v).transpose(0, 2, 1, 3).reshape(BN, D_MODEL)
        x1 = x0 + (ctx @ w['Wo']).reshape(B, N_TOK, D_MODEL)

        xf_3d, xhf, invf = _rmsnorm(x1, w['gn_f'])
        xf = xf_3d.reshape(BN, D_MODEL)
        zg = xf @ w['Wfg']
        g_act, g_sig = _silu(zg)
        u_lin = xf @ w['Wfu']
        h_glu = g_act * u_lin
        x2 = x1 + (h_glu @ w['Wfd']).reshape(B, N_TOK, D_MODEL)

        s_raw = np.float32(0.5) * x2[:, C.T - 1, :] + np.float32(0.5) * np.mean(x2, axis=1)
        sn, xht, invt = _rmsnorm(s_raw, w['gn_t'])
        cache = (xn, xha, inva, q, k, v, att, ctx, xf, xhf, invf, zg, g_act, g_sig, u_lin, h_glu, xht, invt)
        return sn, cache

    @staticmethod
    def _effective_m(mask, M):
        if mask is None:
            return M
        active = np.any(mask > 0, axis=0)
        return int(np.max(np.where(active)[0])) + 1 if np.any(active) else 1

    def q_all_fast(self, S, Cmat, mask=None, use_target=False, need_v=True):
        w = self.t if use_target else self.p
        B, M, _ = Cmat.shape
        M_eff = self._effective_m(mask, M)
        C_eff = Cmat[:, :M_eff, :]
        z_s = (S @ w['W5'][:D_MODEL])[:, None, :]
        z_c = (C_eff.reshape(B * M_eff, C.CAND_DIM) @ w['W5'][D_MODEL:] + w['b5']).reshape(B, M_eff, C.HID)
        ha1 = _silu_fast(z_s + z_c).reshape(B * M_eff, C.HID)
        ha2 = _silu_fast(ha1 @ w['W5b'] + w['b5b'])
        a = (ha2 @ w['W6'] + w['b6']).reshape(B, M_eff)
        if not need_v:
            if M_eff < M:
                out = np.full((B, M), -1e9, dtype=np.float32)
                out[:, :M_eff] = a
                return out
            return a
        hv = _silu_fast(S @ w['W7'] + w['b7'])
        v = (hv @ w['W8'] + w['b8'])[:, 0]
        if mask is not None:
            m_eff = mask[:, :M_eff]
            cnt = np.maximum(m_eff.sum(axis=1, keepdims=True), 1.0)
            a_mean = (a * m_eff).sum(axis=1, keepdims=True) / cnt
        else:
            a_mean = a.mean(axis=1, keepdims=True)
        q_eff = v[:, None] + (a - a_mean)
        if M_eff < M:
            out = np.zeros((B, M), dtype=np.float32)
            out[:, :M_eff] = q_eff
            return out
        return q_eff

    def q_all(self, S, Cmat, mask=None, use_target=False):
        w = self.t if use_target else self.p
        B, M, _ = Cmat.shape
        M_eff = self._effective_m(mask, M)
        C_eff = Cmat[:, :M_eff, :]
        z_s = (S @ w['W5'][:D_MODEL])[:, None, :]
        z_c = (C_eff.reshape(B * M_eff, C.CAND_DIM) @ w['W5'][D_MODEL:] + w['b5']).reshape(B, M_eff, C.HID)
        za1 = (z_s + z_c).reshape(B * M_eff, C.HID)
        ha1, sa1 = _silu(za1)
        za2 = ha1 @ w['W5b'] + w['b5b']
        ha2, sa2 = _silu(za2)
        a = (ha2 @ w['W6'] + w['b6']).reshape(B, M_eff)

        zv = S @ w['W7'] + w['b7']
        hv, sv = _silu(zv)
        v = (hv @ w['W8'] + w['b8'])[:, 0]

        if mask is not None:
            m_eff = mask[:, :M_eff]
            cnt = np.maximum(m_eff.sum(axis=1, keepdims=True), 1.0)
            a_mean = (a * m_eff).sum(axis=1, keepdims=True) / cnt
        else:
            a_mean = a.mean(axis=1, keepdims=True)
        q = v[:, None] + (a - a_mean)
        return q, (v, zv, hv, sv, a, za1, ha1, sa1, za2, ha2, sa2, C_eff, M_eff)

    def aux(self, L0, C_act):
        X = np.concatenate([L0, C_act], axis=1)
        z = X @ self.p['W10'] + self.p['b10']
        h, s = _silu(z)
        l_pred = h @ self.p['W11'] + self.p['b11']
        r_pred = (h @ self.p['W12'] + self.p['b12'])[:, 0]
        return l_pred, r_pred, z, h, s, X

    # --------------------------------------------------------------- adamw --
    def step(self, grads, lr):
        sq_norm = 0.0
        for g in grads.values():
            if g is not None:
                sq_norm += float(np.sum(g * g))
        gnorm = float(np.sqrt(sq_norm))
        scale = np.float32(min(1.0, C.GRAD_CLIP / max(gnorm, 1e-8)))

        self.tstep += 1
        b1, b2, eps = np.float32(C.ADAM['b1']), np.float32(C.ADAM['b2']), np.float32(C.ADAM['eps'])
        om_b1 = np.float32(1.0 - b1)
        om_b2 = np.float32(1.0 - b2)
        lr_eff = np.float32(lr / (1.0 - float(b1) ** self.tstep))
        inv_sqrt_c2 = np.float32(1.0 / np.sqrt(1.0 - float(b2) ** self.tstep))
        wd = np.float32(1.0 - lr * getattr(C, 'WEIGHT_DECAY', 0.0))

        for k, g in grads.items():
            if g is None:
                continue
            gs = g * scale if scale < 1.0 else g
            m = self.m[k]
            v = self.v[k]
            m *= b1
            m += om_b1 * gs
            v *= b2
            v += om_b2 * (gs * gs)
            p = self.p[k]
            if p.ndim >= 2 and k != 'pos' and wd < 1.0:
                p *= wd
            p -= lr_eff * m / (np.sqrt(v) * inv_sqrt_c2 + eps)
        return gnorm

    def sync_target(self):
        for k in self.t:
            self.t[k] = self.p[k].copy()

    def soft_sync_target(self, tau=C.TAU):
        omt = np.float32(1.0 - tau)
        t_f = np.float32(tau)
        for k in self.t:
            tk = self.t[k]
            tk *= omt
            tk += t_f * self.p[k]

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

    # 1. Encoder forward (EMA target encoder for collapse-free world model target)
    L0, enc_cache = net.embed(X0)
    L1 = net.infer_embed(X1, use_target=True)

    # 2. Transformer + Dueling Q forward
    S, tmp_cache = net.temporal(Ls)
    q, q_cache = net.q_all(S, Cmat, mask=mask)
    v, zv, hv, sv, a, za1, ha1, sa1, za2, ha2, sa2, C_eff, M_eff = q_cache
    mask_eff = mask[:, :M_eff]

    q_next_sel = batch['q_target_sel']
    gamma_n = C.GAMMA ** batch.get('n_step', C.N_STEP)
    munch_alpha = getattr(C, 'MUNCH_ALPHA', 0.0)
    if munch_alpha > 0.0:
        tau = getattr(C, 'MUNCH_TAU', 0.03)
        q_masked = np.where(mask_eff > 0, q, -1e9)
        q_max = np.max(q_masked, axis=1, keepdims=True)
        lse = q_max + tau * np.log(
            np.sum(np.exp(np.clip((q_masked - q_max) / tau, -30.0, 0.0)) * mask_eff, axis=1, keepdims=True) + 1e-12
        )
        munch = np.clip(q[ar, ai] - lse[:, 0], -1.0, 0.0)
        rew_eff = rew + munch_alpha * munch
    else:
        rew_eff = rew
    target = (rew_eff + (1.0 - done) * gamma_n * q_next_sel).astype(np.float32)

    qsel = q[ar, ai]
    td = qsel - target
    huber_g = np.clip(td, -1.0, 1.0) * is_w * (1.0 / B)
    dq = np.zeros_like(q)
    dq[ar, ai] = huber_g * mask_eff[ar, ai]

    # Dueling backward (factorized W5 over M_eff candidates)
    dv_from_q = dq.sum(axis=1)
    cnt = np.maximum(mask_eff.sum(axis=1, keepdims=True), 1.0)
    da = dq - mask_eff * (dq.sum(axis=1, keepdims=True) / cnt)

    dW6 = ha2.T @ da.reshape(-1, 1)
    db6 = np.array([da.sum()], dtype=np.float32)
    dha2 = da.reshape(-1, 1) @ net.p['W6'].T
    dza2 = dha2 * _silu_grad(za2, sa2)
    dW5b = ha1.T @ dza2
    db5b = dza2.sum(axis=0)
    dha1 = dza2 @ net.p['W5b'].T
    dza1 = dha1 * _silu_grad(za1, sa1)
    dza1_3d = dza1.reshape(B, M_eff, C.HID)
    dza1_sum = dza1_3d.sum(axis=1)
    dW5_s = S.T @ dza1_sum
    dW5_c = C_eff.reshape(B * M_eff, C.CAND_DIM).T @ dza1
    dW5 = np.concatenate([dW5_s, dW5_c], axis=0)
    db5 = dza1_sum.sum(axis=0)
    dS_from_a = dza1_sum @ net.p['W5'][:D_MODEL].T

    v_loss_vec = (v - target)
    dv = dv_from_q + (C.V_W / B) * is_w * np.clip(v_loss_vec, -1.0, 1.0)
    dW8 = hv.T @ dv.reshape(-1, 1)
    db8 = np.array([dv.sum()], dtype=np.float32)
    dhv = dv.reshape(-1, 1) @ net.p['W8'].T
    dzv = dhv * _silu_grad(zv, sv)
    dW7 = S.T @ dzv
    db7 = dzv.sum(axis=0)
    dS_from_v = dzv @ net.p['W7'].T

    # Transformer backward
    (xn, xha, inva, q_t, k_t, v_t, att, ctx, xf, xhf, invf, zg, g_act, g_sig, u_lin, h_glu, xht, invt) = tmp_cache
    dS = dS_from_a + dS_from_v
    ds_raw, dgn_t = _rmsnorm_grad(dS, xht, invt, net.p['gn_t'])
    dx2 = np.broadcast_to(ds_raw[:, None, :] * (0.5 / N_TOK), (B, N_TOK, D_MODEL)).copy()
    dx2[:, C.T - 1, :] += 0.5 * ds_raw

    # SwiGLU FFN backward
    BN = B * N_TOK
    dx2_flat = dx2.reshape(BN, D_MODEL)
    dWfd = h_glu.T @ dx2_flat
    dh_glu = dx2_flat @ net.p['Wfd'].T
    dg_act = dh_glu * u_lin
    du_lin = dh_glu * g_act
    dzg = dg_act * _silu_grad(zg, g_sig)
    dWfg = xf.T @ dzg
    dWfu = xf.T @ du_lin
    dxf = (dzg @ net.p['Wfg'].T + du_lin @ net.p['Wfu'].T).reshape(B, N_TOK, D_MODEL)
    dx1_norm, dgn_f = _rmsnorm_grad(dxf, xhf, invf, net.p['gn_f'])
    dx1 = dx2 + dx1_norm

    # Multi-Head Attention backward
    dx1_flat = dx1.reshape(BN, D_MODEL)
    dWo = ctx.T @ dx1_flat
    dctx = (dx1_flat @ net.p['Wo'].T).reshape(B, N_TOK, N_H, D_H).transpose(0, 2, 1, 3)
    datt = dctx @ v_t.transpose(0, 1, 3, 2)
    dv_t = att.transpose(0, 1, 3, 2) @ dctx
    dscores = att * (datt - np.sum(datt * att, axis=-1, keepdims=True)) * INV_SQRT_DH
    dq_t = dscores @ k_t
    dk_t = dscores.transpose(0, 1, 3, 2) @ q_t
    dqkv = np.stack([dq_t, dk_t, dv_t], axis=0).transpose(1, 3, 0, 2, 4).reshape(BN, 3 * D_MODEL)
    dWqkv = xn.T @ dqkv
    dxn = (dqkv @ net.p['Wqkv'].T).reshape(B, N_TOK, D_MODEL)
    dx0_norm, dgn_a = _rmsnorm_grad(dxn, xha, inva, net.p['gn_a'])
    dx0 = dx1 + dx0_norm
    dpos = dx0.sum(axis=0, keepdims=True)

    # 3. World-Model Aux Head backward (Next Latent + Reward Prediction)
    C_chosen = Cmat[ar, ai]
    l_pred, r_pred, zax, hax, sax, Xax = net.aux(L0, C_chosen)
    dy = ((2.0 * C.AUX_W) / (B * D_MODEL)) * (l_pred - L1)
    r_target = np.clip(rew / 3.0, -2.0, 2.0)
    dr = ((2.0 * C.REW_AUX_W) / B) * (r_pred - r_target)

    dW11 = hax.T @ dy
    db11 = dy.sum(axis=0)
    dW12 = hax.T @ dr.reshape(-1, 1)
    db12 = np.array([dr.sum()], dtype=np.float32)

    dhax = dy @ net.p['W11'].T + dr.reshape(-1, 1) @ net.p['W12'].T
    dzax = dhax * _silu_grad(zax, sax)
    dW10 = Xax.T @ dzax
    db10 = dzax.sum(axis=0)
    dXax = dzax @ net.p['W10'].T
    dL0 = dXax[:, :D_MODEL]

    grads = {
        'pos': dpos, 'gn_a': dgn_a, 'Wqkv': dWqkv, 'Wo': dWo,
        'gn_f': dgn_f, 'Wfg': dWfg, 'Wfu': dWfu, 'Wfd': dWfd, 'gn_t': dgn_t,
        'W5': dW5, 'b5': db5, 'W5b': dW5b, 'b5b': db5b, 'W6': dW6, 'b6': db6,
        'W7': dW7, 'b7': db7, 'W8': dW8, 'b8': db8,
        'W10': dW10, 'b10': db10, 'W11': dW11, 'b11': db11, 'W12': dW12, 'b12': db12,
    }

    if do_encoder:
        Xg, Xc, zg, sg, zc, sc, h_cat, xh0, inv0 = enc_cache
        dL0_raw, dgn_e = _rmsnorm_grad(dL0, xh0, inv0, net.p['gn_e'])
        dW2 = h_cat.T @ dL0_raw
        db2 = dL0_raw.sum(axis=0)
        dh_cat = dL0_raw @ net.p['W2'].T
        dzg = dh_cat[:, :C.H_GRID] * _silu_grad(zg, sg)
        dzc = dh_cat[:, C.H_GRID:] * _silu_grad(zc, sc)
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
        'aux_loss': float(np.mean((l_pred - L1) ** 2)),
        'rew_loss': float(np.mean((r_pred - r_target) ** 2)),
        'td': float(np.mean(np.abs(td))),
        'td_vec': np.abs(td),
        'q': float(np.mean(qsel)),
    }
    return grads, info