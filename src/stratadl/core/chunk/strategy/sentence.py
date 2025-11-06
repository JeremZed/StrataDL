from stratadl.core.chunk.strategy.base import ChunkStrategy
from stratadl.core.chunk.strategy.type import ChunkTypeStrategy
from typing import List, Dict, Optional
from nltk.tokenize import sent_tokenize

class SentenceChunk(ChunkStrategy):
    """
        Permet de chunker le contenu par phrase
    """
    def __init__(self, chunk_size:int = 512, overlap:int = 50):
        # chunk_size représente le nombre de caractères souhaité pour un chunk, si une phrase dépasse
        # cette longeur alors on la découpe également
        super().__init__(chunk_size=chunk_size, overlap=overlap)

    def chunk(self, content:str, metadata: Optional[Dict] = None):
        """
            Permet de lancer le découpage
        """
        metadata = metadata or {}
        sentences = sent_tokenize(content)

        # Découpe chaque phrase (et si elle dépasse chunk_size, splitter() la segmente automatiquement)
        for line, sentence in enumerate(sentences):
            self.chunks.extend( self.splitter(sentence, { **metadata, "type" : ChunkTypeStrategy.SENTENCE.value, "line" : line }) )

        return self.chunks