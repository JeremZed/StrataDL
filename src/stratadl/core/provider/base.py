from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List


class BaseProvider(ABC):
    """
    Classe de base pour tous les providers LLM.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.provider_name = self.__class__.__name__.replace('Provider', '').lower()

    @abstractmethod
    def generate(self, prompt: str, stream:bool, **kwargs) -> str:
        """
        Génère une réponse à partir d'un prompt.
        """
        pass

    @abstractmethod
    def chat(self, prompt: str, stream:bool, **kwargs) -> str:
        """
        Génère une réponse à partir d'un prompt avec la possibilité d'injecter un historique de messages et une liste d'outils
        """
        pass

    @abstractmethod
    def embed(self, text: str, **kwargs) -> List[float]:
        """
        Retourne un vecteur d'embedding pour un texte.
        """
        pass
