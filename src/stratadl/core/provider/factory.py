from typing import Literal, Optional
from stratadl.core.provider.base import BaseProvider
from stratadl.core.provider.ollama import OllamaProvider



def get_provider(
    provider_name: Literal["ollama"] = "ollama",
    config: Optional[dict] = None,
    **kwargs,
) -> BaseProvider:
    """
    Factory pour créer un provider.

    Examples:
        provider = get_provider("ollama", { "api_url" : "http://127.0.0.1:11434", "model" : "llama3.1" })
    """
    if provider_name == "ollama":
        return OllamaProvider(config, **kwargs)

    else:
        raise ValueError(
            f"Provider '{provider_name}' inconnu. "
            f"Providers disponibles: ollama"
        )
