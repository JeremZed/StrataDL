from stratadl.core.chunk.strategy.base import ChunkStrategy, ChunkItem
from stratadl.core.chunk.strategy.type import ChunkTypeStrategy
from typing import List, Dict, Optional

class HierarchyChunk(ChunkStrategy):
    """
        Permet de chunker un contenu en fonction d'une suite de séparateurs
        de la plus grosse unité à la plus petite
        Par exemple pour la liste suivante : separators = ["\n\n", "\n", ".", " ", ""] cela donne le découpage suivant
        \n\n → paragraphes
        \n → lignes
        . → phrases
        " " → mots
        "" → caractère

        On commence par le séparateur de plus haute priorité (le plus gros, ex: paragraphes)
        On découpe le texte selon ce séparateur
        Pour chaque morceau :
            - Si le chunk est plus petit que chunk_size, on le garde
            - Sinon :
                On descend au séparateur suivant dans la hiérarchie
                On réapplique la même logique sur ce morceau
        On répète jusqu'à obtenir des chunks de taille acceptable ou jusqu'au niveau le plus fin.
    """
    def __init__(self, chunk_size:int = 512, overlap:int = 50, separators:Optional[list] = None):
        super().__init__(chunk_size=chunk_size, overlap=overlap)
        self.separators = separators or ["\n\n", "\n", ".", " ", ""]
        self.separator_names = {
            "\n\n": "paragraph",
            "\n": "line",
            ".": "sentence",
            " ": "word",
            "": "char"
        }

    def _split_by_separator(self, text: str, separator: str) -> List[str]:
        """
        Divise le texte par un séparateur en conservant le séparateur dans les morceaux
        """
        if separator == "":
            return list(text)

        parts = text.split(separator)
        result = []
        for i, part in enumerate(parts):
            if i < len(parts) - 1:
                result.append(part + separator)
            elif part:
                result.append(part)
        return result

    def _recursive_split(self, text: str, separators: List[str],
                     hierarchy_path: List[str] = None,
                     parent_id: str = None) -> List[tuple]:
        """
        Découpe récursivement le texte en utilisant la hiérarchie de séparateurs
        Retourne une liste de tuples (texte, chemin_hiérarchique, chunk_id, parent_id)
        """
        if hierarchy_path is None:
            hierarchy_path = []

        if not text or not separators:
            chunk_id = self._generate_chunk_id()
            return [(text, hierarchy_path, chunk_id, parent_id)] if text else []

        if len(text) <= self.chunk_size:
            chunk_id = self._generate_chunk_id()
            return [(text, hierarchy_path, chunk_id, parent_id)]

        current_separator = separators[0]
        remaining_separators = separators[1:]
        level_name = self.separator_names.get(current_separator, f"sep_{len(hierarchy_path)}")
        new_hierarchy = hierarchy_path + [level_name]

        splits = self._split_by_separator(text, current_separator)
        result = []
        current_chunk = ""
        current_chunk_hierarchy = new_hierarchy

        for split in splits:
            if len(split) > self.chunk_size:
                # Sauvegarde le chunk en cours
                if current_chunk:
                    chunk_id = self._generate_chunk_id()
                    result.append((current_chunk, current_chunk_hierarchy, chunk_id, parent_id))
                    # Le chunk actuel devient le parent des sous-chunks
                    current_parent = chunk_id
                    current_chunk = ""
                else:
                    current_parent = parent_id

                # Découpe récursive avec le nouveau parent
                if remaining_separators:
                    result.extend(self._recursive_split(split, remaining_separators,
                                                    new_hierarchy, current_parent))
                else:
                    forced_hierarchy = new_hierarchy + ["forced_split"]
                    forced_chunks = self.splitter_by_token(split)
                    for chunk in forced_chunks:
                        chunk_id = self._generate_chunk_id()
                        result.append((chunk.content, forced_hierarchy, chunk_id, current_parent))

            elif len(current_chunk) + len(split) <= self.chunk_size:
                current_chunk += split
                current_chunk_hierarchy = new_hierarchy
            else:
                if current_chunk:
                    chunk_id = self._generate_chunk_id()
                    result.append((current_chunk, current_chunk_hierarchy, chunk_id, parent_id))
                current_chunk = split
                current_chunk_hierarchy = new_hierarchy

        if current_chunk:
            chunk_id = self._generate_chunk_id()
            result.append((current_chunk, current_chunk_hierarchy, chunk_id, parent_id))

        return result

    def _apply_overlap(self, chunks_with_hierarchy: List[tuple]) -> List[tuple]:
        """
        Applique l'overlap entre les chunks
        Retourne (texte_avec_overlap, hiérarchie, chunk_id, parent_id)
        """
        if not chunks_with_hierarchy or self.overlap == 0:
            return chunks_with_hierarchy

        overlapped_chunks = []

        for i, (chunk, hierarchy, chunk_id, parent_id) in enumerate(chunks_with_hierarchy):
            if i == 0:
                overlapped_chunks.append((chunk, hierarchy, chunk_id, parent_id))
            else:
                previous_chunk_text = chunks_with_hierarchy[i - 1][0]
                overlap_text = previous_chunk_text[-self.overlap:] if len(previous_chunk_text) >= self.overlap else previous_chunk_text
                # On garde le chunk_id et parent_id d'origine
                overlapped_chunks.append((overlap_text + chunk, hierarchy, chunk_id, parent_id))

        return overlapped_chunks

    def chunk(self, content: str, separators: List[str] = None,
            metadata: Optional[Dict] = None) -> List[ChunkItem]:
        metadata = metadata or {}
        separators = separators or self.separators

        # Découpage hiérarchique
        raw_chunks_with_hierarchy = self._recursive_split(content, separators)

        # Application de l'overlap
        overlapped_chunks = self._apply_overlap(raw_chunks_with_hierarchy)

        # Création des ChunkItem (les IDs sont déjà générés)
        for i, (chunk_content, hierarchy_path, chunk_id, parent_id) in enumerate(overlapped_chunks):
            chunk_item = ChunkItem(
                content=chunk_content,
                metadata={
                    **metadata,
                    "chunk_id": chunk_id,
                    "chunk_parent": parent_id,
                    "type": ChunkTypeStrategy.HIERARCHY.value,
                    "chunk_index": i,
                    "hierarchy": ",".join(hierarchy_path) if hierarchy_path else "root"
                }
            )
            self.chunks.append(chunk_item)

        return self.chunks