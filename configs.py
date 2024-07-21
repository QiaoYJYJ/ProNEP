import yacs
from yacs.config import CfgNode as CN

_C = CN()

# Protein feature extractor
_C.PROTEIN = CN()
_C.PROTEIN.NUM_FILTERS = [128, 128, 128, 128]
_C.PROTEIN.KERNEL_SIZE = [3, 5, 7, 4]
_C.PROTEIN.EMBEDDING_DIM = 128
_C.PROTEIN.PADDING = True

# BCN setting
_C.BCN = CN()
_C.BCN.HEADS = 2

# MLP decoder
_C.DECODER = CN()
_C.DECODER.NAME = "MLP"
_C.DECODER.IN_DIM = 256
_C.DECODER.HIDDEN_DIM = 512
_C.DECODER.OUT_DIM = 128
_C.DECODER.BINARY = 1

# SOLVER
_C.SOLVER = CN()
_C.SOLVER.MAX_EPOCH = 100
_C.SOLVER.BATCH_SIZE = 16
_C.SOLVER.NUM_WORKERS = 0
_C.SOLVER.LR = 5e-5
_C.SOLVER.DA_LR = 1e-3
_C.SOLVER.SEED = 2048

# RESULT
_C.RESULT = CN()
_C.RESULT.OUTPUT_DIR = "./result"
_C.RESULT.SAVE_MODEL = True

# Domain adaptation
_C.DA = CN()
_C.DA.TASK = False
_C.DA.METHOD = "CDAN"
_C.DA.USE = False
_C.DA.INIT_EPOCH = 10
_C.DA.LAMB_DA = 1
_C.DA.RANDOM_LAYER = False
_C.DA.ORIGINAL_RANDOM = False
_C.DA.RANDOM_DIM = None
_C.DA.USE_ENTROPY = True

# Comet config, ignore it If not installed.
_C.COMET = CN()
# Please change to your own workspace name on comet.
_C.COMET.WORKSPACE = "QiaoYJYJ"
_C.COMET.PROJECT_NAME = "ProNEP"
_C.COMET.USE = False
_C.COMET.TAG = None


def get_cfg_defaults():
    return _C.clone()
