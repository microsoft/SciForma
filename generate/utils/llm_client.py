"""
Lightweight async LLM client supporting OpenAI and Azure OpenAI.

Credential priority:
  1. SCIFORMA_AZURE_CONFIG_PATH                 → Azure CLI + endpoint pool config
  2. AZURE_OPENAI_API_KEY + AZURE_OPENAI_ENDPOINT → Azure with key
  3. AZURE_OPENAI_ENDPOINT (no key)             → Azure managed identity
  4. OPENAI_API_KEY                             → standard OpenAI
"""
from __future__ import annotations

import ast
import asyncio
import os
from pathlib import Path

from openai import AsyncAzureOpenAI, AsyncOpenAI


def _endpoint_from_config(path: str) -> tuple[str, str]:
    """Read one endpoint and token scope from a Python config without executing it."""
    tree = ast.parse(Path(path).expanduser().read_text())
    endpoint_map = None
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if any(
            isinstance(target, ast.Name) and target.id == "endpoint_token_provider_dict"
            for target in node.targets
        ):
            endpoint_map = ast.literal_eval(node.value)
            break
    if not endpoint_map:
        raise ValueError("No active endpoint_token_provider_dict found in Azure config")
    index = int(os.environ.get("SCIFORMA_AZURE_ENDPOINT_INDEX", "0"))
    endpoints = list(endpoint_map.items())
    if not 0 <= index < len(endpoints):
        raise ValueError(
            "SCIFORMA_AZURE_ENDPOINT_INDEX must be between 0 and "
            f"{len(endpoints) - 1}"
        )
    return endpoints[index]


def _make_client():
    endpoint_config = os.environ.get("SCIFORMA_AZURE_CONFIG_PATH")
    azure_ep  = os.environ.get("AZURE_OPENAI_ENDPOINT")
    azure_key = os.environ.get("AZURE_OPENAI_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")
    api_ver = os.environ.get("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")

    if endpoint_config:
        from azure.identity import AzureCliCredential, get_bearer_token_provider

        endpoint, token_scope = _endpoint_from_config(endpoint_config)
        sync_token_provider = get_bearer_token_provider(AzureCliCredential(), token_scope)

        async def token_provider() -> str:
            return await asyncio.to_thread(sync_token_provider)

        return AsyncOpenAI(base_url=endpoint, api_key=token_provider)
    if azure_ep:
        # Strip trailing path components — AsyncAzureOpenAI needs base URL only
        # e.g. https://xxx.openai.azure.com/openai/v1 → https://xxx.openai.azure.com
        import re
        base_ep = re.sub(r"/openai.*$", "", azure_ep.rstrip("/"))

        if azure_key:
            return AsyncAzureOpenAI(
                api_key=azure_key,
                azure_endpoint=base_ep,
                api_version=api_ver,
            )
        else:
            from azure.identity import DefaultAzureCredential, get_bearer_token_provider
            token_provider = get_bearer_token_provider(
                DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
            )
            return AsyncAzureOpenAI(
                azure_ad_token_provider=token_provider,
                azure_endpoint=base_ep,
                api_version=api_ver,
            )
    elif openai_key:
        return AsyncOpenAI(api_key=openai_key)
    else:
        raise OSError(
            "Set SCIFORMA_AZURE_CONFIG_PATH, AZURE_OPENAI_ENDPOINT, or OPENAI_API_KEY."
        )


def _completion_parameters(model: str, max_tokens: int, temperature: float) -> dict:
    """Choose compatible Chat Completions parameters with optional env overrides."""
    normalized = model.rsplit("/", 1)[-1].lower()
    reasoning_family = normalized.startswith(("o1", "o3", "o4", "gpt-5"))

    token_mode = os.environ.get("SCIFORMA_MAX_TOKENS_PARAM", "auto").strip().lower()
    if token_mode not in {"auto", "max_tokens", "max_completion_tokens"}:
        raise ValueError(
            "SCIFORMA_MAX_TOKENS_PARAM must be auto, max_tokens, or "
            "max_completion_tokens"
        )
    use_completion_tokens = token_mode == "max_completion_tokens" or (
        token_mode == "auto" and reasoning_family
    )
    token_parameter = "max_completion_tokens" if use_completion_tokens else "max_tokens"
    parameters = {token_parameter: max_tokens}

    temperature_mode = (
        os.environ.get("SCIFORMA_SEND_TEMPERATURE", "auto").strip().lower()
    )
    if temperature_mode not in {"auto", "always", "never"}:
        raise ValueError("SCIFORMA_SEND_TEMPERATURE must be auto, always, or never")
    if temperature_mode == "always" or (
        temperature_mode == "auto" and not reasoning_family
    ):
        parameters["temperature"] = temperature
    return parameters


async def chat(
    messages: list[dict],
    model: str = "gpt-5.4",
    system: str | None = None,
    max_tokens: int = 8192,
    temperature: float = 1.0,
    retries: int = 3,
) -> str:
    client = _make_client()
    full_messages = []
    if system:
        full_messages.append({"role": "system", "content": system})
    full_messages.extend(messages)

    try:
        request_args = _completion_parameters(model, max_tokens, temperature)
        for attempt in range(retries):
            try:
                resp = await client.chat.completions.create(
                    model=model,
                    messages=full_messages,
                    **request_args,
                )
                return resp.choices[0].message.content.strip()
            except Exception:
                if attempt == retries - 1:
                    raise
                await asyncio.sleep(2 ** attempt)
        return ""
    finally:
        await client.close()
