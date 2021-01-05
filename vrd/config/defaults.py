import os

from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()
_C.TASK = "ObjCls"

_C.MODEL = CN()
_C.MODEL.WEIGHTS = ( )
_C.MODEL.DEVICE = "cuda"

_C.MODEL.NUM_OBJ_CLASSES = 100
_C.MODEL.NUM_PRED_CLASSES = 46
_C.MODEL.IOU_FLOOR = 0.5
_C.MODEL.N2P = 3
_C.MODEL.HIDDEN_CHANNEL = 768

_C.MODEL.BACKBONE = CN()
_C.MODEL.BACKBONE.ARCHITECTURE = "vgg16"

_C.MODEL.DETECTOR = CN()
_C.MODEL.DETECTOR.NUM_DETECTIONS = 15
_C.MODEL.DETECTOR.SCORE_THRESH = 0.01
_C.MODEL.DETECTOR.NMS_THRESH = 0.5
_C.MODEL.DETECTOR.FG_IOU_THRESH = 0.5
_C.MODEL.DETECTOR.BG_IOU_THRESH = 0.5

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
