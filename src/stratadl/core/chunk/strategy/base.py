from abc import ABC, abstractmethod
from typing import List, Dict, Optional
from dataclasses import dataclass
from transformers import AutoTokenizer
import uuid

@dataclass
class ChunkItem:
    """ Représente le chunk et ses metas à injecter dans le vector store """
    content: str
    metadata: Dict[str, any]

class ChunkStrategy(ABC):
    def __init__(self, chunk_size : int = 500, overlap : int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.chunks = []

    @abstractmethod
    def chunk(self, content:str, metadata: Optional[Dict] = None) -> List[ChunkItem]:
        pass

    def splitter(self, content, metadata: Optional[Dict] = None):
        """
            Fonction standard de découpage d'un texte en fonction de la taille du chunk
        """
        metadata = metadata or {}
        chunks = []
        content_size = len(content)

        if content_size < self.chunk_size:

            c = ChunkItem(
                content=content,
                metadata={**metadata, "start_pos": 0, "end_pos": content_size}
            )

            chunks.append(c)
        else:
            start = 0
            end = 0
            running = True
            while(running):

                if end == 0:
                    start = 0
                    end  = start + self.chunk_size
                else:
                    start = (end - self.overlap)
                    end = start + (self.chunk_size - self.overlap)

                if end > content_size:
                    chunk_content = content[start:]
                    running = False
                else:
                    chunk_content = content[start:end]

                c = ChunkItem(
                    content=chunk_content,
                    metadata={**metadata, "start_pos": start, "end_pos": end}
                )

                chunks.append(c)

        return chunks

    def splitter_by_token(self, content: str,
                      pretrained_model_name_or_path: str = "bert-base-uncased",
                      metadata: Optional[Dict] = None) -> List[ChunkItem]:
        """
            Fonction de découpage d'un texte en fonction du nombre de tokens LLM
            avec overlap
        """
        metadata = metadata or {}

        # Initialisation du tokenizer
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
        tokens = tokenizer.tokenize(content)

        # Utilisation de splitter() sur la liste de tokens
        token_chunks = self.splitter(
            tokens,
            metadata={**metadata, "model": pretrained_model_name_or_path}
        )

        # Conversion des tokens en texte pour chaque chunk
        chunks = []
        for chunk_item in token_chunks:
            chunk_text = tokenizer.convert_tokens_to_string(chunk_item.content)

            # Mise à jour du chunk avec le texte converti et métadonnées enrichies
            c = ChunkItem(
                content=chunk_text,
                metadata={**chunk_item.metadata}
            )
            chunks.append(c)

        return chunks


    def _generate_chunk_id(self):
        return str(uuid.uuid4())