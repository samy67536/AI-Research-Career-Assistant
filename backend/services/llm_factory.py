# backend/services/llm_factory.py
"""
Multi-LLM Factory — Dynamic API Key Support
API key can be passed at runtime (from UI) OR read from .env
Priority: runtime key > .env key
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


class LLMResponse:
    def __init__(self, text: str, tokens_used: int = 0):
        self.text = text
        self.tokens_used = tokens_used


_runtime_keys = {
    "claude":   "",
    "openai":   "",
    "gemini":   "",
    "deepseek": "",
}
_runtime_provider = ""
_runtime_model    = ""


def set_runtime_config(provider: str, api_key: str, model: str = ""):
    global _runtime_provider, _runtime_model
    _runtime_keys[provider.lower()] = api_key.strip()
    _runtime_provider = provider.lower()
    if model:
        _runtime_model = model
    print(f"[LLM] Runtime config set → provider={provider}, key={'*'*8 + api_key[-4:] if len(api_key) > 8 else '???'}")


def get_api_key(provider: str) -> str:
    runtime = _runtime_keys.get(provider.lower(), "").strip()
    if runtime and not runtime.startswith("YOUR_") and len(runtime) > 10:
        return runtime

    from config import settings
    env_map = {
        "claude":   settings.anthropic_api_key,
        "openai":   settings.openai_api_key,
        "gemini":   settings.gemini_api_key,
        "deepseek": settings.deepseek_api_key,
    }
    env_key = env_map.get(provider.lower(), "").strip()
    if env_key and not env_key.startswith("YOUR_") and not env_key.startswith("sk-ant-api03-YOUR"):
        return env_key

    raise ValueError(
        f"❌ No valid API key found for '{provider}'.\n"
        f"👉 In the Streamlit sidebar: enter your {provider.upper()} API key and click 'Apply'.\n"
        f"💡 Tip: Google Gemini has a FREE tier — no credit card needed!"
    )


def _friendly_error(provider: str, raw_error) -> str:
    """Convert raw API errors into friendly, actionable messages."""
    err = str(raw_error).lower()

    if any(k in err for k in ["insufficient_quota", "insufficient balance",
                                "exceeded your current quota", "billing_hard_limit",
                                "out of credits", "you have run out", "credit balance"]):
        tips = {
            "openai":   "👉 Add credits at: https://platform.openai.com/billing\n"
                        "💡 Or switch to Gemini (FREE) or DeepSeek (very cheap).",
            "claude":   "👉 Add credits at: https://console.anthropic.com\n"
                        "💡 Or switch to Gemini (FREE) or DeepSeek (very cheap).",
            "gemini":   "👉 You may have hit the free-tier rate limit. Wait a minute and retry.",
            "deepseek": "👉 Add credits at: https://platform.deepseek.com",
        }
        tip = tips.get(provider.lower(), "💡 Switch to Gemini (FREE tier available).")
        return (
            f"💳 {provider.upper()} account has insufficient credits / quota exceeded.\n"
            f"{tip}"
        )

    if any(k in err for k in ["invalid api key", "incorrect api key", "invalid_api_key",
                                "authentication", "unauthorized", "401", "permission"]):
        key_urls = {
            "openai":   "https://platform.openai.com/api-keys",
            "claude":   "https://console.anthropic.com",
            "gemini":   "https://aistudio.google.com/apikey",
            "deepseek": "https://platform.deepseek.com",
        }
        url = key_urls.get(provider.lower(), "the provider website")
        return (
            f"🔑 Invalid {provider.upper()} API key.\n"
            f"👉 Double-check the key — make sure you copied it fully with no spaces.\n"
            f"   Get/verify your key at: {url}"
        )

    if any(k in err for k in ["model not found", "does not exist", "invalid model",
                                "no such model", "model_not_found"]):
        return (
            f"🤖 Model not found for {provider.upper()}.\n"
            f"👉 Select a different model from the sidebar dropdown."
        )

    if any(k in err for k in ["connection", "timeout", "network", "unreachable",
                                "connectionerror", "connecttimeout"]):
        return (
            f"🌐 Network error connecting to {provider.upper()} API.\n"
            f"👉 Check your internet connection and try again."
        )

    return f"❌ {provider.upper()} API Error: {raw_error}"


class ClaudeProvider:
    """Anthropic Claude API"""
    DEFAULT_MODEL = "claude-3-5-sonnet-20241022"

    def __init__(self, api_key: str = "", model: str = ""):
        import anthropic
        key = api_key or get_api_key("claude")
        self.client = anthropic.Anthropic(api_key=key)
        self.model  = model or _runtime_model or self.DEFAULT_MODEL

    def generate(self, system_prompt: str, user_prompt: str,
                 max_tokens: int = 2048, temperature: float = 0.3) -> LLMResponse:
        try:
            msg = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}]
            )
            text   = msg.content[0].text
            tokens = msg.usage.input_tokens + msg.usage.output_tokens
            return LLMResponse(text=text, tokens_used=tokens)
        except Exception as e:
            raise RuntimeError(_friendly_error("claude", e))


class OpenAIProvider:
    """OpenAI GPT API"""
    DEFAULT_MODEL = "gpt-4o-mini"

    def __init__(self, api_key: str = "", model: str = ""):
        from openai import OpenAI
        key = api_key or get_api_key("openai")
        self.client = OpenAI(api_key=key)
        self.model  = model or _runtime_model or self.DEFAULT_MODEL

    def generate(self, system_prompt: str, user_prompt: str,
                 max_tokens: int = 2048, temperature: float = 0.3) -> LLMResponse:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt}
                ]
            )
            return LLMResponse(
                text=response.choices[0].message.content,
                tokens_used=response.usage.total_tokens
            )
        except Exception as e:
            raise RuntimeError(_friendly_error("openai", e))


class GeminiProvider:
    """Google Gemini API — has a FREE tier"""
    DEFAULT_MODEL = "gemini-1.5-flash"

    def __init__(self, api_key: str = "", model: str = ""):
        import google.generativeai as genai
        key = api_key or get_api_key("gemini")
        genai.configure(api_key=key)
        model_name = model or _runtime_model or self.DEFAULT_MODEL
        self.model = genai.GenerativeModel(
            model_name=model_name,
            system_instruction="You are an expert AI assistant."
        )

    def generate(self, system_prompt: str, user_prompt: str,
                 max_tokens: int = 2048, temperature: float = 0.3) -> LLMResponse:
        try:
            import google.generativeai as genai
            config = genai.GenerationConfig(max_output_tokens=max_tokens, temperature=temperature)
            full_prompt = f"{system_prompt}\n\n{user_prompt}"
            response = self.model.generate_content(full_prompt, generation_config=config)
            return LLMResponse(text=response.text, tokens_used=0)
        except Exception as e:
            raise RuntimeError(_friendly_error("gemini", e))


class DeepSeekProvider:
    """DeepSeek API (OpenAI-compatible endpoint)"""
    DEFAULT_MODEL = "deepseek-chat"

    def __init__(self, api_key: str = "", model: str = ""):
        from openai import OpenAI
        key = api_key or get_api_key("deepseek")
        self.client = OpenAI(api_key=key, base_url="https://api.deepseek.com")
        self.model  = model or _runtime_model or self.DEFAULT_MODEL

    def generate(self, system_prompt: str, user_prompt: str,
                 max_tokens: int = 2048, temperature: float = 0.3) -> LLMResponse:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt}
                ]
            )
            return LLMResponse(
                text=response.choices[0].message.content,
                tokens_used=response.usage.total_tokens
            )
        except Exception as e:
            raise RuntimeError(_friendly_error("deepseek", e))


def get_llm_provider(provider: str = None, api_key: str = "", model: str = ""):
    from config import settings
    p = (provider or _runtime_provider or settings.llm_provider or "gemini").lower()

    providers = {
        "claude":   ClaudeProvider,
        "openai":   OpenAIProvider,
        "gemini":   GeminiProvider,
        "deepseek": DeepSeekProvider,
    }
    if p not in providers:
        raise ValueError(f"Unknown provider '{p}'. Choose: {list(providers.keys())}")

    return providers[p](api_key=api_key, model=model)


def validate_api_key(provider: str, api_key: str) -> dict:
    try:
        set_runtime_config(provider, api_key)
        llm = get_llm_provider(provider, api_key=api_key)
        resp = llm.generate(
            system_prompt="You are a helpful assistant.",
            user_prompt="Reply with exactly: OK",
            max_tokens=10,
            temperature=0
        )
        ok = "ok" in resp.text.lower() or len(resp.text) > 0
        return {"valid": ok, "message": f"✅ {provider.upper()} key is working!", "response": resp.text}
    except Exception as e:
        return {"valid": False, "message": str(e)}
