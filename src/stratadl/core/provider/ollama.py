from stratadl.core.provider.base import BaseProvider
from stratadl.core.provider.stream import OllamaStreamResult, OllamaDownloadResult
import requests
import json
from typing import List
import sys


class OllamaProvider(BaseProvider):

    def __init__(self, config=None):
        super().__init__(config)
        self.api_url = self.config.get("api_url", "http://localhost:11434")
        self.model = self.config.get("model", "llama3.2")

    def set_model(self, model_name:str):
        """
        Permet de setter un modèle
        """

        self.model = model_name

    def generate(self, prompt: str, stream: bool = False, **kwargs):
        """
        Génère une réponse depuis Ollama.

        Args:
            prompt: Le texte d'entrée
            stream: Si True, affiche en temps réel et retourne un générateur
            **kwargs: Arguments supplémentaires pour l'API Ollama

        Returns:
            Si stream=False: str (texte complet)
            Si stream=True: générateur qui yield chaque chunk
        """
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": stream,
                **kwargs
            }

            response = requests.post(
                f"{self.api_url}/api/generate",
                json=payload,
                stream=stream
            )
            response.raise_for_status()

            if stream:
                return self._handle_stream(response)
            else:
                return self._handle_non_stream(response)

        except requests.exceptions.HTTPError as e:
            print("HTTP error:", e)
            if e.response is not None:
                print("Response content:", e.response.text)
            return None
        except requests.exceptions.RequestException as e:
            print("Request failed:", e)
            return None

    def chat(self, messages:List[dict], stream:bool = False, **kwargs):
        """
        Génère le message suivant à une conversation
        """

        try:
            payload = {
                "model": self.model,
                "messages": messages,
                "stream": stream,
                **kwargs
            }

            response = requests.post(
                f"{self.api_url}/api/chat",
                json=payload,
                stream=stream
            )
            response.raise_for_status()

            if stream:
                return self._handle_stream(response)
            else:
                return self._handle_non_stream(response)

        except requests.exceptions.HTTPError as e:
            print("HTTP error:", e)
            if e.response is not None:
                print("Response content:", e.response.text)
            return None
        except requests.exceptions.RequestException as e:
            print("Request failed:", e)
            return None

    def _handle_stream(self, response):
        def stream_generator():
            for line in response.iter_lines():
                if line:
                    chunk = json.loads(line.decode('utf-8'))
                    yield chunk

                    if chunk.get('done', False):
                        break

        return OllamaStreamResult(stream_generator())

    def _handle_non_stream(self, response):
        """Gère la réponse non-streamée (JSON unique)"""

        try:
            data = json.loads(response.text)
        except json.JSONDecodeError as e:
            print("Debug: JSON decode error:", e)
            print("Debug: Raw response:", response.text)
            return ""

        # Accès direct au texte dans le champ message.content
        message = data.get("message", {})
        content = message.get("content", "")

        return content.strip()

    def embed(self, text: str, **kwargs) -> list[float]:
        try:
            payload = {"model": self.model, "input": text, **kwargs}
            response = requests.post(f"{self.api_url}/api/embed", json=payload)
            response.raise_for_status()
            return response.json()

        except requests.exceptions.HTTPError as e:
            print("HTTP error:", e)
            if e.response is not None:
                print("Response content:", e.response.text)
            return None
        except requests.exceptions.RequestException as e:
            print("Request failed:", e)
            return None

    def list_models(self):
        """
            Permet de retourner la liste des modèles télécharger
        """

        try:

            response = requests.get(f"{self.api_url}/api/tags")
            response.raise_for_status()
            return response.json()

        except requests.exceptions.HTTPError as e:
            print("HTTP error:", e)
            if e.response is not None:
                print("Response content:", e.response.text)
            return None
        except requests.exceptions.RequestException as e:
            print("Request failed:", e)
            return None

    def model_info(self, verbose:bool = False):
        """
            Permet de retourner les informations du modèle en cours
        """
        try:
            payload = {"model": self.model, "verbose": verbose}
            response = requests.post(f"{self.api_url}/api/show", json=payload)
            response.raise_for_status()
            return response.json()

        except requests.exceptions.HTTPError as e:
            print("HTTP error:", e)
            if e.response is not None:
                print("Response content:", e.response.text)
            return None
        except requests.exceptions.RequestException as e:
            print("Request failed:", e)
            return None

    def pull_model(self, model: str, stream: bool = False, insecure: bool = False, **kwargs):
        """
        Télécharge un modèle depuis la bibliothèque Ollama

        Args:
            model: nom du modèle à télécharger
            stream: si False, retourne la réponse finale uniquement
            insecure: permet les connexions non sécurisées
            **kwargs: arguments supplémentaires

        Returns:
            Si stream=False: dict avec le statut final
            Si stream=True: OllamaDownloadResult (générateur)
        """
        try:
            payload = {
                "model": model,
                "stream": stream,
                "insecure": insecure,
                **kwargs
            }

            response = requests.post(
                f"{self.api_url}/api/pull",
                json=payload,
                stream=stream
            )
            response.raise_for_status()

            if stream:
                return self._handle_pull_stream(response)
            else:
                return self._handle_pull_non_stream(response)

        except requests.exceptions.HTTPError as e:
            print("HTTP error:", e)
            if e.response is not None:
                print("Response content:", e.response.text)
            return None
        except requests.exceptions.RequestException as e:
            print("Request failed:", e)
            return None

    def _handle_pull_stream(self, response):
        """Gère le téléchargement en mode streaming"""
        def download_generator():
            for line in response.iter_lines():
                if line:
                    chunk = json.loads(line.decode('utf-8'))
                    yield chunk

                    if chunk.get('status') == 'success':
                        break

        return OllamaDownloadResult(download_generator())

    def download(self, model_name:str):
        """
            Permet de lancer le téléchargement tout en affichant la progression
        """
        result = self.pull_model(model_name, stream=True)

        for chunk in result:
            status = chunk.get('status', '')

            if status == 'pulling manifest':
                print("📥 Récupération du manifest...")

            elif "pulling" in status:
                digest = chunk.get('digest', '')[:12]
                completed = chunk.get('completed', 0)
                total = chunk.get('total', 1)
                percentage = (completed / total) * 100

                # Barre de progression
                bar_length = 40
                filled = int(bar_length * completed / total)
                bar = '█' * filled + '░' * (bar_length - filled)

                mb_completed = completed / (1024 * 1024)
                mb_total = total / (1024 * 1024)

                sys.stdout.write(f'\r{digest}: [{bar}] {percentage:.1f}% ({mb_completed:.1f}/{mb_total:.1f} MB)')
                sys.stdout.flush()

            elif status == 'verifying sha256 digest':
                print("\n✓ Vérification de l'intégrité...")

            elif status == 'writing manifest':
                print("✓ Écriture du manifest...")

            elif status == 'removing any unused layers':
                print("✓ Nettoyage des couches inutilisées...")

            elif status == 'success':
                print("✅ Téléchargement terminé avec succès!")

        # Statistiques finales
        stats = result.get_stats()
        print(f"\nStatistiques:")
        print(f"  - Taille totale: {stats['total_size_mb']} MB")
        print(f"  - Statut: {stats['status']}")

    def _handle_pull_non_stream(self, response):
        """Gère le téléchargement en mode non-streamé"""
        result = response.json()
        return result

    def delete(self, model_name:str):
        """
            Permet de supprimer un modèle
        """

        try:
            payload = {"model": model_name}
            response = requests.delete(f"{self.api_url}/api/delete", json=payload)
            response.raise_for_status()
            return response.text

        except requests.exceptions.HTTPError as e:
            print("HTTP error:", e)
            if e.response is not None:
                print("Response content:", e.response.text)
            return None
        except requests.exceptions.RequestException as e:
            print("Request failed:", e)
            return None

    def version(self):
        """
            Permet de retourner la version d'ollama
        """

        try:
            response = requests.get(f"{self.api_url}/api/version")
            response.raise_for_status()
            return response.json()

        except requests.exceptions.HTTPError as e:
            print("HTTP error:", e)
            if e.response is not None:
                print("Response content:", e.response.text)
            return None
        except requests.exceptions.RequestException as e:
            print("Request failed:", e)
            return None


"""

Memo pour l'implémentation des flux stream dans les autres providers

class OllamaProvider(BaseProvider):
    # ... code existant ...

    def _handle_stream(self, response):
        def stream_generator():
            for line in response.iter_lines():
                if line:
                    chunk = json.loads(line.decode('utf-8'))
                    yield chunk

                    if chunk.get('done', False):
                        break

        return OllamaStreamResult(stream_generator())


class OpenAIProvider(BaseProvider):
    # ... code existant ...

    def _handle_stream(self, response):
        def stream_generator():
            for line in response.iter_lines():
                if line:
                    line_text = line.decode('utf-8')
                    if line_text.startswith('data: '):
                        if line_text.strip() == 'data: [DONE]':
                            break
                        chunk = json.loads(line_text[6:])
                        yield chunk

        return OpenAIStreamResult(stream_generator())


class AnthropicProvider(BaseProvider):
    # ... code existant ...

    def _handle_stream(self, response):
        def stream_generator():
            for line in response.iter_lines():
                if line:
                    line_text = line.decode('utf-8')
                    if line_text.startswith('data: '):
                        chunk = json.loads(line_text[6:])
                        yield chunk

                        if chunk.get('type') == 'message_stop':
                            break

        return AnthropicStreamResult(stream_generator())


"""