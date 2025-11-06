import pymupdf4llm
from stratadl.core.ingestion.file.base import BaseFileIngestor
import requests
from typing import Optional

class PDFDocumentIngestor(BaseFileIngestor):
    """
    Utilise pymupdf4llm qui a des heuristiques avancées pour la structure
    """

    def __init__(self, file_path: str, output_dir: str = "output"):
        super().__init__(file_path, output_dir)
        self.markdown_version = None

    def convert_to_markdown(self):
        """Convertit le PDF en Markdown avec structure préservée"""
        # pymupdf4llm analyse la position, la taille, le style pour déduire la hiérarchie
        self.markdown_version = pymupdf4llm.to_markdown(
            self.file_path,
            page_chunks=False,  # Ne pas diviser en chunks
            write_images=False,  # Ignorer les images
            show_progress=False
        )
        return self.markdown_version

    def parser_tika(self, tika_url: Optional[str] = "http://localhost:9998", **kwargs):
        """Parse PDF avec Tika via requests"""
        with open(self.file_path, 'rb') as f:
            response = requests.put(
                f"{tika_url}/tika",
                data=f,
                headers={'Accept': 'text/plain'}
            )

        # Pour les métadonnées
        with open(self.file_path, 'rb') as f:
            meta_response = requests.put(
                f"{tika_url}/meta",
                data=f,
                headers={'Accept': 'application/json'}
            )

        return meta_response.json(), response.text, response.status_code

    def convert(self):
        return self.convert_to_markdown()