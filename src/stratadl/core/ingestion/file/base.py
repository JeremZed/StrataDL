from abc import ABC, abstractmethod
from typing import Optional
import os


class BaseFileIngestor(ABC):
    """
    Classe de base pour tous les ingesteurs de contenu de fichier.
    """

    def __init__(self, path: str, output_dir: str = "output"):
        self.file_path = path
        self.output_dir = output_dir

    @abstractmethod
    def convert(self) -> str:
        """
        Converti le contenu du fichier en markdown.
        """
        pass

    def save(self, filename: Optional[str] = None) -> str:
        """
        Enregistre le contenu Markdown retourné par `convert()` dans un fichier .md.

        Args:
            filename (str, optional): Nom du fichier de sortie (par défaut, nom du fichier d'entrée avec extension .md).

        Returns:
            str: Chemin absolu du fichier sauvegardé.
        """

        # Création du répertoire de sortie s'il n'existe pas
        os.makedirs(self.output_dir, exist_ok=True)

        # Détermination du nom de fichier
        if filename is None:
            base_name = os.path.splitext(os.path.basename(self.file_path))[0]
            filename = f"{base_name}.md"

        output_path = os.path.join(self.output_dir, filename)
        markdown_content = self.convert()

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(markdown_content)

        return os.path.abspath(output_path)
