import mammoth
from markdownify import markdownify as md

from stratadl.core.ingestion.file.base import BaseFileIngestor

class WordDocumentIngestor(BaseFileIngestor):
    """
    Classe d'ingestion de documents Word (.docx) pour extraction du contenu tout gardant la hierarchie des informations
    """

    def __init__(self, file_path: str, output_dir: str = "output"):
        super().__init__(file_path, output_dir)

        self.html_version = None
        self.markdown_version = None

    def convert_to_html(self):
        """
            Permet de convertir le contenu docx en html
        """

        with open(self.file_path, "rb") as docx_file:
            result = mammoth.convert_to_html(docx_file)
            self.html_version = f"<html><body>{result.value}</body></html>"

        return self.html_version

    def convert_to_markdown(self, strip = ['img']):
        """
            Permet de convertir le contenu html en markdown
        """
        if self.html_version is None:
            self.convert_to_html()

        self.markdown_version = md(self.html_version, strip=strip)

        return self.markdown_version


    def convert(self):
        return self.convert_to_markdown()
