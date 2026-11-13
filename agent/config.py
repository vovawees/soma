"""SOMA agent — configuration and constants.

Every number here is a *starting point* to be measured, not a promise.
Dimensions are frozen so that the wire protocol (bot <-> agent) and the
replay buffer stay compatible; changing them requires a schema bump.
"""

# ---------------------------------------------------------------- geometry --
GRID_N = 8          # 8x8x8 voxel window around the player's feet
GRID_STEP = 2       # sampled every 2 blocks  -> +-7 blocks reach
GRID_CELLS = GRID_N * GRID_N * GRID_N

SELF_DIM = 32
GRID_DIM = GRID_CELLS * 6          # 6 features per cell

ENT_MAX = 12
ENT_DIM = 10
ENT_TOTAL = ENT_MAX * ENT_DIM

INV_SLOTS = 41                     # 0-8 hotbar, 9-35 main, 36-39 armour, 40 offhand
INV_DIM = 4
INV_TOTAL = INV_SLOTS * INV_DIM

EV_MAX = 10
EV_DIM = 5
EV_TOTAL = EV_MAX * EV_DIM

META_DIM = 12

FRAME_DIM = SELF_DIM + GRID_DIM + ENT_TOTAL + INV_TOTAL + EV_TOTAL + META_DIM

# ------------------------------------------------------------------- model --
T = 6               # temporal window (frames)
ENC = 192           # per-frame latent
H1 = 768            # encoder hidden
H2 = 384            # temporal hidden
HID = 128           # head hidden
MAX_CAND = 48       # candidates per frame (matches bot)
CAND_DIM = 16       # per-candidate features
AUX_HID = 128

# ---------------------------------------------------------------- training --
GAMMA = 0.97
N_STEP = 3
BATCH = 64
LR = 3e-4
LR_AUX = 3e-4
AUX_W = 0.5
V_W = 0.25
TARGET_SYNC = 2000          # gradient steps between target-network syncs
BUFFER_N = 200_000
TRAIN_EVERY_MS = 100        # trainer cadence
WARMUP = 2_000              # transitions before training starts

# ------------------------------------------------------------------- paths --
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, 'data')
CKPT = os.path.join(DATA, 'model.npz')
REPLAY = os.path.join(DATA, 'replay.f16')
STATE = os.path.join(DATA, 'state.json')
LOG = os.path.join(DATA, 'metrics.jsonl')

# -------------------------------------------------------------------- misc --
ADAM = dict(b1=0.9, b2=0.999, eps=1e-8)