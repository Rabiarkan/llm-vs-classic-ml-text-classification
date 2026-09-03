import os
from abc import ABC, abstractmethod


class LLMProvider(ABC):
    name: str
    model: str
    client: object

    @abstractmethod
    def complete(self, system: str, user: str, max_tokens: int = 256) -> tuple[str, int, int]:
        ...


class AnthropicProvider(LLMProvider):
    name = "anthropic"

    def __init__(self, model: str):
        import anthropic
        self.model = model
        self.client = anthropic.Anthropic() 

    def complete(self, system, user, max_tokens=256):
        r = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            extra_body={"temperature": 0},
            system=[{"type": "text", "text": system,
                     "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": user}],
        )
        text = "".join(b.text for b in r.content if b.type == "text")
        return text, r.usage.input_tokens, r.usage.output_tokens



class GroqProvider(LLMProvider):
    name = "groq"

    def __init__(self, model: str):
        from openai import OpenAI
        self.model = model
        self.client = OpenAI(
            api_key=os.environ["GROQ_API_KEY"],
            base_url="https://api.groq.com/openai/v1",
        )

    def complete(self, system, user, max_tokens=256):
        r = self.client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=0,
            reasoning_effort="low",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        text = r.choices[0].message.content or ""
        return text, r.usage.prompt_tokens, r.usage.completion_tokens


def get_provider(name: str, model: str) -> LLMProvider:
    if name == "groq":
        return GroqProvider(model)
    if name == "anthropic":
        return AnthropicProvider(model)
    raise ValueError(f"unknown provider: {name!r} (groq | anthropic)")

def provider_for(model: str) -> str:
    return "anthropic" if model.startswith("claude") else "groq"

def assert_model_available(provider: LLMProvider) -> None:
    models = getattr(provider.client, "models", None)
    if models is None or not hasattr(models, "list"):
        return
    try:
        ids = {m.id for m in models.list().data}
    except Exception:
        return 
    if provider.model not in ids:
        raise ValueError(
            f"'{provider.model}' not available from this provider.\nAvailable: {sorted(ids)}"
        )