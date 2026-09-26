from langchain_anthropic import ChatAnthropic
from langchain_core.rate_limiters import InMemoryRateLimiter

# Shared across every chain so the whole pipeline stays under Anthropic's request rate
_rate_limiter = InMemoryRateLimiter(requests_per_second=1)


def get_llm(model: str = "claude-haiku-4-5"):
    # API key is read from the ANTHROPIC_API_KEY env var by ChatAnthropic itself.
    # The anthropic SDK already retries 429/5xx/connection errors with backoff
    # (max_retries, default 2) -- bump it here since free/low-tier limits are why
    # this project moved off Mistral in the first place.
    # No temperature override: current-gen models (Sonnet 5, Opus 5) reject
    # sampling params outright (400) -- determinism is tuned via adaptive
    # thinking/effort now, not temperature.
    return ChatAnthropic(
        model=model,
        rate_limiter=_rate_limiter,
        max_retries=5,
    )
