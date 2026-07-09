"""
OpenSciDraw Models Module

This module provides model registration for custom models.
For most use cases, the model_factory in utils/ handles loading
models from diffusers and transformers libraries dynamically.

To add a custom model:
1. Define your model class in this directory
2. Register it using the MODELS registry from sciforma.registry
3. Add the model specification to MODEL_REGISTRY in utils/model_factory.py

Example:
    from sciforma.registry import MODELS
    
    @MODELS.register_module()
    class MyCustomTransformer(nn.Module):
        ...

Then in model_factory.py, add:
    MODEL_REGISTRY["MyCustomModel"] = ModelSpec(
        transformer_module="OpenSciDraw.models",
        transformer_class="MyCustomTransformer",
        ...
    )
"""

from sciforma.registry import MODELS

# Import any custom models here to register them
# from .my_custom_model import MyCustomTransformer

__all__ = [
    'MODELS',
]
