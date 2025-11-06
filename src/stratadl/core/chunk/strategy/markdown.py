from stratadl.core.chunk.strategy.base import ChunkItem
from stratadl.core.chunk.strategy.hierarchy import HierarchyChunk
from typing import List, Dict, Optional

class MarkdownChunk(HierarchyChunk):
    """
    Chunking spécialisé pour Markdown, respectant la structure du document
    @TODO: Ajouter la gestion des tableaux et autres éléments spécifiques au markdown
    """
    def __init__(self, chunk_size: int = 512, overlap: int = 50):
        markdown_separators = [
            "\n---\n",
            "\n## ",
            "\n### ",
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ]
        super().__init__(chunk_size=chunk_size, overlap=overlap, separators=markdown_separators)
        self.separator_names = {
            "\n---\n": "horizontal_rule",
            "\n## ": "heading2",
            "\n### ": "heading3",
            "\n\n": "paragraph",
            "\n": "line",
            ". ": "sentence",
            " ": "word",
            "": "char"
        }

    def chunk(self, content: str, metadata: Optional[Dict] = None) -> List[ChunkItem]:
        metadata = metadata or {}
        metadata["format"] = "markdown"
        return super().chunk(content, metadata=metadata)