from stratadl.core.chunk.strategy.base import ChunkStrategy
from stratadl.core.chunk.strategy.type import ChunkTypeStrategy
from typing import Dict, Optional

class TokenLLMChunk(ChunkStrategy):
    """
        Permet de chunker le contenu en token type LLM
    """
    def __init__(self, chunk_size:int = 512, overlap:int = 50):
        super().__init__(chunk_size=chunk_size, overlap=overlap)

    def chunk(self, content:str,
              pretrained_model_name_or_path="bert-base-uncased",
              metadata: Optional[Dict] = None):
        """
            Permet de lancer le découpage en token au style de BERT
        """
        metadata = metadata or {}
        self.chunks = self.splitter_by_token(
            content,
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            metadata={**metadata, "type": ChunkTypeStrategy.TOKEN.value}
        )

        return self.chunks