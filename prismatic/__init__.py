# Lazy import: only pull in the full model/training stack when explicitly requested.
# This prevents the eval/inference path from loading training-only dependencies
# (dlimp, RLDS datasets, etc.) just because prismatic.extern.hf is imported.
def __getattr__(name):
    if name in ("available_model_names", "available_models", "get_model_description", "load"):
        from .models import available_model_names, available_models, get_model_description, load
        return locals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
