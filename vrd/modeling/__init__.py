from .model import Model

def build_model(cfg, is_train):
    return Model(cfg, is_train) 
