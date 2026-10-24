"""
Check HuggingFace token identity and probe available Prompt Guard repo names.
Token read from HF_TOKEN env var only — never hardcoded.
"""
import os, sys
from huggingface_hub import HfApi, model_info

TOKEN = os.environ.get("HF_TOKEN", "")
if not TOKEN:
    print("[FAIL] HF_TOKEN not set")
    sys.exit(1)

api = HfApi(token=TOKEN)

# ── Who is the token owner? ──────────────────────────────────────────────────
print("── Checking token identity ──")
try:
    info = api.whoami()
    print(f"  Logged in as : {info.get('name', 'unknown')}")
    auth = info.get("auth", {})
    print(f"  Token type   : {auth.get('type', 'unknown')}")
    print(f"  Access level : {auth.get('accessToken', {}).get('role', 'unknown')}")
except Exception as e:
    print(f"  [ERROR] whoami failed: {e}")

# ── Probe alternate Prompt Guard repo names ──────────────────────────────────
print("\n-- Probing repo names --")
repos = [
    "meta-llama/Prompt-Guard-2-22M",
    "meta-llama/Prompt-Guard-86M",
    "meta-llama/PromptGuard-2-22M",
    "meta-llama/LlamaGuard-2-22M",
]
for repo in repos:
    try:
        m = model_info(repo, token=TOKEN)
        gated = getattr(m, "gated", "unknown")
        print(f"  [FOUND]   {repo}  gated={gated}")
    except Exception as e:
        short_err = str(e).split("\n")[0][:100]
        print(f"  [MISSING] {repo}  -> {short_err}")
