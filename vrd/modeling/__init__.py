from .tasks import build_objcls, build_predcls, build_predet, build_objdet, build_reldet, build_catreldet

def build_model(cfg):
    return eval('build_' + cfg.TASK.lower())(cfg)