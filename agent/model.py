"""SOMA decision model — compact vector-native MLP with factorised heads.

Layout (idea.txt section 7):

    frame(3450) --encoder--> latent(192) --\
                                          > temporal over 6 frames --> state(192)
    frame_t+1 --encoder (shared) ---------/            |
                                                       +--> Q head (per candidate)
                                                       +--> V head
    (latent_t, latent_t+1) -------------------------------> aux/dynamics head

The encoder is trained by the auxiliary (self-supervised) loss only; the RL loss
uses detached latents.  Policy learning stays cheap while perception keeps
improving from every single transition.
"""

import numpy as np

import config as C


def _he(rng, fan_in, shape):
    return (rng.normal(0.0, np.sqrt(2.0 / max(1, fan_in)), shape)).astype(np.float32)


class Net:
    def __init__(self, seed=0):
        rng = np.random.RandomState(seed)
        p = {}
        p['W1'] = _he(rng, C.FRAME_DIM, (C.FRAME_DIM, C.H1)); p['b1'] = np.zeros(C.H1, np.float32)
        p['W2'] = _he(rng, C.H1, (C.H1, C.ENC));            p['b2'] = np.zeros(C.ENC, np.float32)
        p['W3'] = _he(rng, C.T * C.ENC, (C.T * C.ENC, C.H2)); p['b3'] = np.zeros(C.H2, np.float32)
        p['W4'] = _he(rng, C.H2, (C.H2, C.ENC));            p['b4'] = np.zeros(C.ENC, np.float32)
        p['W5'] = _he(rng, C.ENC + C.CAND_DIM, (C.ENC + C.CAND_DIM, C.HID)); p['b5'] = np.zeros(C.HID, np.float32)
        p['W6'] = _he(rng, C.HID, (C.HID, 1));              p['b6'] = np.zeros(1, np.float32)
        p['W7'] = _he(rng, C.ENC, (C.ENC, 64));             p['b7'] = np.zeros(64, np.float32)
        p['W8'] = _he(rng, 64, (64, 1));                    p['b8'] = np.zeros(64, np.float32)
        p['W10'] = _he(rng, 2 * C.ENC, (2 * C.ENC, C.AUX_HID)); p['b10'] = np.zeros(C.AUX_HID, np.float32)
        p['W11'] = _he(rng, C.AUX_HID, (C.AUX_HID, C.ENC)); p['b11'] = np.zeros(C.ENC, np.float32)
        self.p = p
        self.t = {k: v.copy() for k, v in p.items()}
        self.m = {k: np.zeros_like(v) for k, v in p.items()}
        self.v = {k: np.zeros_like(v) for k, v in p.items()}
        self.tstep = 0

    # ------------------------------------------------------------- forward --
    def embed(self, X):
        z1 = X @ self.p['W1'] + self.p['b1']
        a1 = np.maximum(z1, 0.0, dtype=np.float32)
        return a1 @ self.p['W2'] + self.p['b2'], z1, a1

    def temporal(self, L):
        B = L.shape[0]
        X = L.reshape(B, C.T * C.ENC)
        z = X @ self.p['W3'] + self.p['b3']
        h = np.maximum(z, 0.0, dtype=np.float32)
        return h @ self.p['W4'] + self.p['b4'], z, h, X

    def q_all(self, S, Cmat):
        B, M, _ = Cmat.shape
        X = np.concatenate([np.repeat(S[:, None, :], M, axis=1), Cmat], axis=2).reshape(B * M, C.ENC + C.CAND_DIM)
        z = X @ self.p['W5'] + self.p['b5']
        h = np.maximum(z, 0.0, dtype=np.float32)
        q = (h @ self.p['W6'] + self.p['b6']).reshape(B, M)
        return q, z, h, X

    def q_all_target(self, S, Cmat):
        B, M, _ = Cmat.shape
        X = np.concatenate([np.repeat(S[:, None, :], M, axis=1), Cmat], axis=2).reshape(B * M, C.ENC + C.CAND_DIM)
        h = np.maximum(X @ self.t['W5'] + self.t['b5'], 0.0, dtype=np.float32)
        return (h @ self.t['W6'] + self.t['b6']).reshape(B, M)

    def value(self, S):
        z = S @ self.p['W7'] + self.p['b7']
        h = np.maximum(z, 0.0, dtype=np.float32)
        return (h @ self.p['W8'] + self.p['b8'])[:, 0], z, h

    def value_target(self, S):
        h = np.maximum(S @ self.t['W7'] + self.t['b7'], 0.0, dtype=np.float32)
        return (h @ self.t['W8'] + self.t['b8'])[:, 0]

    def aux(self, L0, L1):
        X = np.concatenate([L0, L1], axis=1)
        z = X @ self.p['W10'] + self.p['b10']
        h = np.maximum(z, 0.0, dtype=np.float32)
        return h @ self.p['W11'] + self.p['b11'], z, h, X

    # --------------------------------------------------------------- adam ---
    def step(self, grads, lr):
        self.tstep += 1
        b1, b2, eps = C.ADAM['b1'], C.ADAM['b2'], C.ADAM['eps']
        c1 = 1.0 - b1 ** self.tstep
        c2 = 1.0 - b2 ** self.tstep
        for k, g in grads.items():
            if g is None:
                continue
            self.m[k] = b1 * self.m[k] + (1 - b1) * g
            self.v[k] = b2 * self.v[k] + (1 - b2) * (g * g)
            self.p[k] -= (lr * (self.m[k] / c1) / (np.sqrt(self.v[k] / c2) + eps)).astype(np.float32)

    def sync_target(self):
        for k in self.t:
            self.t[k] = self.p[k].copy()

    # ----------------------------------------------------------- (de)serial --
    def save(self, path):
        np.savez(path, **self.p)

    def load(self, path):
        z = np.load(path)
        for k in self.p:
            if k in z:
                self.p[k] = z[k].astype(np.float32)
                self.t[k] = self.p[k].copy()

    def nparams(self):
        return int(sum(v.size for v in self.p.values()))


def embed_grad(net, X, a1, z1, dL):
    """Backward from dL (B,ENC) into the encoder weights for one batch of frames."""
    da1 = dL @ net.p['W2'].T
    dz1 = da1 * (z1 > 0)
    dW1 = X.T @ dz1
    db1 = dz1.sum(axis=0)
    dW2 = a1.T @ dL
    db2 = dL.sum(axis=0)
    return dW1, db1, dW2, db2


def train_batch(net: Net, batch: dict, do_encoder=True):
    """One forward/backward pass over a batch.  Returns (grads, info).

    batch keys:
      X0 (B,F), X1 (B,F)                 consecutive frames
      Ls (B,T,ENC)                       temporal window latents (detached)
      Cmat (B,M,CAND_DIM), mask (B,M)
      ai (B,), rew (B,), done (B,), q_next (B,)   q_next from the target net
    """
    X0, X1 = batch['X0'], batch['X1']
    Ls, Cmat, mask = batch['Ls'], batch['Cmat'], batch['mask']
    rew, done = batch['rew'], batch['done']
    B, M = Cmat.shape[0], Cmat.shape[1]
    ar = np.arange(B)

    L0, z10, a10 = net.embed(X0)
    L1, z11, a11 = net.embed(X1)

    S, _, _, _ = net.temporal(Ls)                      # detached policy path

    q, zq, hq, Xq = net.q_all(S, Cmat)
    qmax_next = np.where(mask > 0, batch['q_next'], -1e6).max(axis=1)
    qmax_next = np.where(np.isfinite(qmax_next), qmax_next, 0.0)
    target = (rew + (1.0 - done) * (C.GAMMA ** C.N_STEP) * qmax_next).astype(np.float32)

    qsel = q[ar, batch['ai']]
    td = qsel - target
    dq = np.zeros_like(q)
    dq[ar, batch['ai']] = (2.0 / B) * td * mask[ar, batch['ai']]

    dW6 = hq.T @ dq.reshape(-1, 1)
    db6 = dq.sum()
    dh = dq.reshape(-1, 1) @ net.p['W6'].T
    dz = dh * (zq > 0)
    dW5 = Xq.T @ dz
    db5 = dz.sum(axis=0)

    v, zv, hv = net.value(S)
    v_target = np.where(done > 0, rew, target)
    dv = ((2.0 * C.V_W) / B) * (v - v_target)
    dW8 = hv.T @ dv.reshape(-1, 1)
    db8 = dv.sum()
    dhv = dv.reshape(-1, 1) @ net.p['W8'].T
    dzv = dhv * (zv > 0)
    dW7 = S.T @ dzv
    db7 = dzv.sum(axis=0)

    y, za, ha, Xa = net.aux(L0, L1)
    dy = ((2.0 * C.AUX_W) / (B * C.ENC)) * (y - L1)
    dW11 = ha.T @ dy
    db11 = dy.sum(axis=0)
    dha = dy @ net.p['W11'].T
    dza = dha * (za > 0)
    dW10 = Xa.T @ dza
    db10 = dza.sum(axis=0)
    dXa = dza @ net.p['W10'].T
    dL0 = dXa[:, :C.ENC]
    dL1 = dXa[:, C.ENC:]

    grads = {
        'W6': dW6, 'b6': db6, 'W5': dW5, 'b5': db5,
        'W8': dW8, 'b8': db8, 'W7': dW7, 'b7': db7,
        'W11': dW11, 'b11': db11, 'W10': dW10, 'b10': db10,
    }

    if do_encoder:
        gW1 = np.zeros_like(net.p['W1']); gb1 = np.zeros_like(net.p['b1'])
        gW2 = np.zeros_like(net.p['W2']); gb2 = np.zeros_like(net.p['b2'])
        for X, a1, z1, dL in ((X0, a10, z10, dL0), (X1, a11, z11, dL1)):
            a, b, c, d = embed_grad(net, X, a1, z1, dL)
            gW1 += a; gb1 += b; gW2 += c; gb2 += d
        grads.update({'W1': gW1, 'b1': gb1, 'W2': gW2, 'b2': gb2})

    info = {
        'q_loss': float(np.mean(td * td)),
        'v_loss': float(np.mean((v - v_target) ** 2)),
        'aux_loss': float(np.mean((y - L1) ** 2)),
        'td': float(np.mean(np.abs(td))),
        'q': float(np.mean(q)),
    }
    return grads, info