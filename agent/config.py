"""SOMA agent v3 — Transformer + SwiGLU + Episodic Key-Value Memory + World Model."""

import os

# ---------------------------------------------------------------- geometry --
GRID_N = 8          # 8x8x8 dense voxel window around the player
GRID_STEP = 1       # step=1 -> gap-free [-3..+4]x[-2..+5]x[-3..+4]
GRID_CELLS = GRID_N * GRID_N * GRID_N

SELF_DIM = 64
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

CORE_DIM = SELF_DIM + ENT_TOTAL + INV_TOTAL + EV_TOTAL + META_DIM  # 724
FRAME_DIM = CORE_DIM + GRID_DIM                                     # 3796

# ------------------------------------------------------------------- model --
T = 8               # short-term temporal window (frames)
TOKENS = T + 3      # 8 short-term + 2 EMA (med/slow) + 1 retrieved episodic memory = 11
ENC = 192           # latent / transformer d_model
N_HEADS = 4         # multi-head self-attention (head_dim = 48)
H_GRID = 256        # voxel stream hidden
H_CORE = 256        # core stream hidden
FFN_HID = 384       # SwiGLU hidden dim
HID = 192           # dueling head hidden
MAX_CAND = 48       # candidates per frame
CAND_DIM = 24       # per-candidate features
AUX_HID = 192
MEM_SLOTS = 64      # neural episodic key-value memory capacity

# ---------------------------------------------------------------- training --
GAMMA = 0.98
N_STEP = 4
BATCH = 64
LR = 3e-4
LR_MIN = 5e-5
WEIGHT_DECAY = 1e-4
AUX_W = 0.5
REW_AUX_W = 0.25    # auxiliary reward prediction weight
V_W = 0.25
MUNCH_ALPHA = 0.9   # Munchausen-DQN log-policy reward scaling
MUNCH_TAU = 0.03    # Munchausen-DQN softmax temperature
GRAD_CLIP = 5.0
TAU = 0.005                 # Polyak soft target update rate
BUFFER_N = 200_000
TRAIN_EVERY_MS = 40
WARMUP = 1_000
PER_ALPHA = 0.6
PER_BETA_START = 0.4

# ------------------------------------------------------------------- paths --
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, 'data')
CKPT = os.path.join(DATA, 'model.npz')
REPLAY = os.path.join(DATA, 'replay.f16')
STATE = os.path.join(DATA, 'state.json')
LOG = os.path.join(DATA, 'metrics.jsonl')

# -------------------------------------------------------------------- misc --
ADAM = dict(b1=0.9, b2=0.999, eps=1e-8)