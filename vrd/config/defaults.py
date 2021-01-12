import os

from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()
_C.TASK = "ObjCls"

_C.MODEL = CN()
_C.MODEL.WEIGHT = ""
_C.MODEL.DEVICE = "cuda"

_C.MODEL.NUM_OBJ_CLASSES = 100
_C.MODEL.NUM_PRED_CLASSES = 46
_C.MODEL.NUM_DETECTIONS = 15
_C.MODEL.NEG2POS  = 3
_C.MODEL.NUM_PRED_SAMPLES = 30
_C.MODEL.HIDDEN_CHANNELS = 768

_C.DATASETS = CN()
_C.DATASETS.TRAIN =()
_C.DATASETS.TEST = ()

_C.INPUT = CN()
_C.INPUT.SIZE = (224, 224)

_C.OUTPUT_DIR = "./outputs"

_C.SOLVER = CN()
_C.SOLVER.BATCH_SIZE = 64
_C.SOLVER.MILESTONES = (8, 9)
_C.SOLVER.MAX_EPOCH = 10
_C.SOLVER.LR = 0.01
_C.SOLVER.GAMMA = 0.1
_C.SOLVER.TEST_PERIOD = 1
_C.SOLVER.CHECKPOINT_PERIOD = 1

_C.DATALOADER = CN()
_C.DATALOADER.NUM_WORKERS = 1

_C.TEST = CN()
_C.TEST.BATCH_SIZE = 64
_C.TEST.SAVE = True
_C.TEST.VISUALIZE = False

_C.PATHS_CATALOG = os.path.join(os.path.dirname(__file__), "paths_catalog.py")
