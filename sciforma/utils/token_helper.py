# ============================================================
# Token Helper Module
# ============================================================
# This module provides a unified way to get sensitive tokens
# with proper fallback hierarchy:
#   1. Environment variables (highest priority, for AMLT/cloud)
#   2. _local_secrets.py file (for local development)
#   3. None / default value (fallback)
#
# Usage in your code:
#     from sciforma.utils.token_helper import get_hf_token, get_wandb_config
#
# For AMLT:
#     Just set environment variables in your yaml, no need to hardcode tokens
#
# NOTE: We use _local_secrets.py instead of secrets.py to avoid
#       conflicting with Python's standard library 'secrets' module
# ============================================================

import os
from typing import Optional

# Try to import from _local_secrets.py (for local development)
# NOTE: Cannot use 'secrets.py' as it conflicts with Python stdlib
try:
    from _local_secrets import HF_TOKEN as _SECRETS_HF_TOKEN
    from _local_secrets import WANDB_API_KEY as _SECRETS_WANDB_API_KEY
    from _local_secrets import WANDB_ENTITY as _SECRETS_WANDB_ENTITY
    from _local_secrets import WANDB_PROJECT as _SECRETS_WANDB_PROJECT
    from _local_secrets import WANDB_BASE_URL as _SECRETS_WANDB_BASE_URL
    _SECRETS_AVAILABLE = True
except ImportError:
    _SECRETS_HF_TOKEN = None
    _SECRETS_WANDB_API_KEY = None
    _SECRETS_WANDB_ENTITY = None
    _SECRETS_WANDB_PROJECT = None
    _SECRETS_WANDB_BASE_URL = None
    _SECRETS_AVAILABLE = False

def get_hf_token(config_token: Optional[str] = None) -> Optional[str]:
    """
    Get HuggingFace token with proper fallback hierarchy.
    
    Priority:
        1. HF_TOKEN environment variable
        2. config_token parameter (from config file)
        3. secrets.py file
        4. None
    
    Args:
        config_token: Token from config file (optional)
    
    Returns:
        HuggingFace token string or None
    """
    # 1. Environment variable (highest priority for cloud/AMLT)
    env_token = os.environ.get("HF_TOKEN")
    if env_token:
        return env_token
    
    # 2. Config file token
    if config_token and config_token.startswith("hf_"):
        return config_token
    
    # 3. Secrets file
    if _SECRETS_AVAILABLE and _SECRETS_HF_TOKEN:
        return _SECRETS_HF_TOKEN
    
    # 4. Return None (will require huggingface-cli login)
    return None


def get_wandb_api_key(config_key: Optional[str] = None) -> Optional[str]:
    """Get WandB API key with proper fallback."""
    env_key = os.environ.get("WANDB_API_KEY")
    if env_key:
        return env_key
    if config_key:
        return config_key
    if _SECRETS_AVAILABLE and _SECRETS_WANDB_API_KEY:
        return _SECRETS_WANDB_API_KEY
    return None


def get_wandb_config() -> dict:
    """Get WandB configuration as a dictionary."""
    return {
        "api_key": get_wandb_api_key(),
        "entity": os.environ.get("WANDB_ENTITY", _SECRETS_WANDB_ENTITY if _SECRETS_AVAILABLE else None),
        "project": os.environ.get("WANDB_PROJECT", _SECRETS_WANDB_PROJECT if _SECRETS_AVAILABLE else None),
        "base_url": os.environ.get("WANDB_BASE_URL", _SECRETS_WANDB_BASE_URL if _SECRETS_AVAILABLE else None),
    }


def setup_wandb_env():
    """Set WandB environment variables from secrets if not already set."""
    config = get_wandb_config()
    
    if config["api_key"] and "WANDB_API_KEY" not in os.environ:
        os.environ["WANDB_API_KEY"] = config["api_key"]
    if config["entity"] and "WANDB_ENTITY" not in os.environ:
        os.environ["WANDB_ENTITY"] = config["entity"]
    if config["project"] and "WANDB_PROJECT" not in os.environ:
        os.environ["WANDB_PROJECT"] = config["project"]
    if config["base_url"] and "WANDB_BASE_URL" not in os.environ:
        os.environ["WANDB_BASE_URL"] = config["base_url"]


def setup_hf_env():
    """Set HF_TOKEN environment variable from secrets if not already set."""
    token = get_hf_token()
    if token and "HF_TOKEN" not in os.environ:
        os.environ["HF_TOKEN"] = token


def setup_all_env():
    """Setup all environment variables from secrets."""
    setup_hf_env()
    setup_wandb_env()


# Debug helper
def print_token_status():
    """Print whether tokens are configured (values are never logged)."""
    import sys
    hf_set = bool(os.environ.get('HF_TOKEN') or _SECRETS_HF_TOKEN)
    wandb_set = bool(os.environ.get('WANDB_API_KEY'))
    sys.stderr.write(f"[token_helper] HF_TOKEN={'SET' if hf_set else 'NOT SET'}, "
                     f"WANDB_API_KEY={'SET' if wandb_set else 'NOT SET'}\n")


if __name__ == "__main__":
    print_token_status()
