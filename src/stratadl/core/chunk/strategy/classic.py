from stratadl.core.chunk.strategy.base import ChunkStrategy, ChunkItem
from stratadl.core.chunk.strategy.type import ChunkTypeStrategy
from typing import List, Dict, Optional

class ClassicChunk(ChunkStrategy):
    """
        Permet de chunker de manière classique en fonction du nombre de caractères + overlap
    """

    def __init__(self, chunk_size : int = 500, overlap : int = 50):
        # chunk_size représente le nombre de caractères souhaité pour un chunk
        super().__init__(chunk_size=chunk_size, overlap=overlap)

    def chunk(self, content:str, metadata: Optional[Dict] = None) -> List[ChunkItem]:
        """
            Permet de lancer le chunk du contenu passé en paramètre
        """
        metadata = metadata or {}
        self.chunks = self.splitter(content, { **metadata, "type" : ChunkTypeStrategy.CLASSIC.value })

        return self.chunks