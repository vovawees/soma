"""SOMA agent — configuration and constants (v2.1 polished).

Split-stream encoder (voxel stream + core stream), multi-scale EMA memory traces,
Dueling Double-DQN + RMSNorm + PER + N-step returns.
"""

import os

# ---------------------------------------------------------------- geometry --
GRID_N = 8          # 8x8x8 dense voxel window around the player
GRID_STEP = 1       # step=1 -> gap-free [-3..+4]x[-2..+5]x[-3..+4]
GRID_CELLS = GRID_N * GRID_N * GRID_N

SELF_DIM = 48
GRID_DIM = GRID_CELLS * 6          # 3072

ENT_MAX = 16
ENT_DIM = 16
ENT_TOTAL = ENT_MAX * ENT_DIM      # 256

INV_SLOTS = 41                     # 0-8 hotbar, 9-35 main, 36-39 armour, 40 offhand
INV_DIM = 8
INV_TOTAL = INV_SLOTS * INV_DIM    # 328

EV_MAX = 10
EV_DIM = 6
EV_TOTAL = EV_MAX * EV_DIM         # 60

META_DIM = 16

CORE_DIM = SELF_DIM + ENT_TOTAL + INV_TOTAL + EV_TOTAL + META_DIM  # 708
FRAME_DIM = CORE_DIM + GRID_DIM                                     # 3780

# ------------------------------------------------------------------- model --
T = 8               # short-term temporal window (frames)
ENC = 192           # per-frame latent
H_GRID = 256        # voxel stream hidden
H_CORE = 256        # core (self+ent+inv+ev+meta) stream hidden
H2 = 384            # temporal hidden (over T*ENC + 2*ENC multi-scale EMA traces)
HID = 192           # dueling head hidden
MAX_CAND = 48       # candidates per frame
CAND_DIM = 24       # per-candidate features
AUX_HID = 192

# ---------------------------------------------------------------- training --
GAMMA = 0.98
N_STEP = 4
BATCH = 64
LR = 3e-4
LR_MIN = 5e-5
AUX_W = 0.5
V_W = 0.25
GRAD_CLIP = 5.0
TAU = 0.005                 # Polyak soft target update rate
BUFFER_N = 200_000
TRAIN_EVERY_MS = 40         # trainer cadence
WARMUP = 1_000              # transitions before training starts
PER_ALPHA = 0.6             # prioritized replay exponent
PER_BETA_START = 0.4        # importance-sampling exponent

# ------------------------------------------------------------------- paths --
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, 'data')
CKPT = os.path.join(DATA, 'model.npz')
REPLAY = os.path.join(DATA, 'replay.f16')
STATE = os.path.join(DATA, 'state.json')
LOG = os.path.join(DATA, 'metrics.jsonl')

# -------------------------------------------------------------------- misc --
ADAM = dict(b1=0.9, b2=0.999, eps=1e-8)