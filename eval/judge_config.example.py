"""
SciFormaBench-2K — Judge Configuration
=======================================
Copy this file to judge_config.py and fill in your credentials.
judge_config.py is gitignored and should never be committed.

Three supported backends (set BACKEND to one of the options below):

  "openai"           — Standard OpenAI API
  "azure_apikey"     — Azure OpenAI with API key (bring-your-own Azure deployment)
  "azure_cli"        — Azure OpenAI via AzureCliCredential (az login, for teams
                       with managed-identity or Entra ID access)

Paper evaluation used: azure_cli with gpt-5.4 deployment.
Use gpt-5.4 to reproduce the reported benchmark scores.
"""

# ─── Select backend ────────────────────────────────────────────────────────────
BACKEND = "openai"   # "openai" | "azure_apikey" | "azure_cli"


# ─── Backend A: Standard OpenAI ────────────────────────────────────────────────
# Paper benchmark judge.
# Get your key at https://platform.openai.com/api-keys
OPENAI_API_KEY  = "sk-YOUR-KEY-HERE"
OPENAI_BASE_URL = "https://api.openai.com/v1"   # or a compatible proxy
OPENAI_MODEL    = "gpt-5.4"


# ─── Backend B: Azure OpenAI with API key ──────────────────────────────────────
# If you have your own Azure OpenAI resource with a vision-capable deployment.
# Format: { endpoint_url: api_key }
AZURE_APIKEY_ENDPOINTS = {
    # "https://YOUR-RESOURCE.openai.azure.com/openai/v1": "YOUR-AZURE-API-KEY",
}
AZURE_APIKEY_DEPLOYMENT = "gpt-5.4"   # your deployment name


# ─── Backend C: Azure with AzureCliCredential (az login) ───────────────────────
# For users with Entra ID / managed-identity access to Azure OpenAI.
# Run `az login` first, then set your endpoint(s) below.
# Format: { endpoint_url: token_scope }
AZURE_CLI_ENDPOINTS = {
    # "https://YOUR-RESOURCE.openai.azure.com/openai/v1": "https://cognitiveservices.azure.com/.default",
}
AZURE_CLI_DEPLOYMENT = "gpt-5.4"
