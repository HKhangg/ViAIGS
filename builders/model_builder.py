from builders.registry import MODEL_REGISTRY
import importlib, os

# Auto-import tất cả file trong models/
def _auto_import_models():
    models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    for fname in os.listdir(models_dir):
        if fname.endswith('_model.py'):
            module_name = f"models.{fname[:-3]}"
            importlib.import_module(module_name)

def build_model(cfg):
    _auto_import_models()
    name = cfg['model']['name']
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Model '{name}' chưa được đăng ký. Có: {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](cfg)