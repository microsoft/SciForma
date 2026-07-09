"""
Lightweight async LLM client supporting OpenAI and Azure OpenAI.

Credential priority:
  1. OPENAI_API_KEY                             → standard OpenAI
  2. AZURE_OPENAI_API_KEY + AZURE_OPENAI_ENDPOINT → Azure with key
  3. AZURE_OPENAI_ENDPOINT (no key)             → Azure managed identity (az login)
"""
from __future__ import annotations
import asyncio, os
from openai import AsyncOpenAI, AsyncAzureOpenAI


def _make_client():
    azure_ep  = os.environ.get("AZURE_OPENAI_ENDPOINT")
    azure_key = os.environ.get("AZURE_OPENAI_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")
    api_ver = os.environ.get("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")

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
        raise EnvironmentError(
            "Set OPENAI_API_KEY, or AZURE_OPENAI_ENDPOINT (+ optional AZURE_OPENAI_API_KEY)"
        )


async def chat(
    messages: list[dict],
    model: str = "gpt-4o",
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

    for attempt in range(retries):
        try:
            # o-series models use max_completion_tokens instead of max_tokens
            is_o_series = any(model.startswith(p) for p in ("o1", "o3", "o4", "gpt-5"))
            token_param = "max_completion_tokens" if is_o_series else "max_tokens"
            resp = await client.chat.completions.create(
                model=model,
                messages=full_messages,
                **{token_param: max_tokens},
                temperature=temperature,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            if attempt == retries - 1:
                raise
            await asyncio.sleep(2 ** attempt)
    return ""
