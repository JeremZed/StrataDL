from abc import ABC, abstractmethod

class BaseStreamResult(ABC):
    """Classe de base pour tous les résultats streamés"""

    def __init__(self, generator):
        self.generator = generator
        self.full_response = ""
        self.chunks = []
        self.metadata = None
        self._consumed = False

    def __iter__(self):
        if self._consumed:
            for chunk in self.chunks:
                yield chunk
        else:
            for chunk in self.generator:
                text = self._extract_text(chunk)
                self.full_response += text
                self.chunks.append(chunk)

                if self._is_done(chunk):
                    self.metadata = self._extract_metadata(chunk)

                yield chunk

            self._consumed = True

    @abstractmethod
    def _extract_text(self, chunk) -> str:
        """Extrait le texte d'un chunk (spécifique au provider)"""
        pass

    @abstractmethod
    def _is_done(self, chunk) -> bool:
        """Détermine si c'est le dernier chunk"""
        pass

    @abstractmethod
    def _extract_metadata(self, chunk) -> dict:
        """Extrait les métadonnées du dernier chunk"""
        pass

    def get_full_response(self):
        if not self._consumed:
            for _ in self:
                pass
        return self.full_response

    def get_metadata(self):
        if not self._consumed:
            for _ in self:
                pass
        return self.metadata

    def get_stats(self):
        """À implémenter dans les sous-classes pour formater les stats"""
        if not self._consumed:
            for _ in self:
                pass
        return self.metadata


class OllamaStreamResult(BaseStreamResult):
    """Implémentation pour Ollama"""

    def _extract_text(self, chunk) -> str:

        if chunk.get('message', None) is not None:
            return chunk.get('message').get('content', '')

        return chunk.get('response', '')

    def _is_done(self, chunk) -> bool:
        return chunk.get('done', False)

    def _extract_metadata(self, chunk) -> dict:
        return {
            'total_duration': chunk.get('total_duration'),
            'load_duration': chunk.get('load_duration'),
            'prompt_eval_count': chunk.get('prompt_eval_count'),
            'prompt_eval_duration': chunk.get('prompt_eval_duration'),
            'eval_count': chunk.get('eval_count'),
            'eval_duration': chunk.get('eval_duration'),
            'context': chunk.get('context'),
            'done_reason': chunk.get('done_reason'),
            'model': chunk.get('model'),
            'created_at': chunk.get('created_at')
        }

    def get_stats(self):
        if not self._consumed:
            for _ in self:
                pass

        if not self.metadata:
            return None

        total_time = self.metadata.get('total_duration', 0) / 1e9
        eval_time = self.metadata.get('eval_duration', 0) / 1e9
        tokens_generated = self.metadata.get('eval_count', 0)

        return {
            'total_time_seconds': round(total_time, 2),
            'generation_time': round(eval_time, 2),
            'tokens_generated': tokens_generated,
            'tokens_per_second': round(tokens_generated / eval_time, 2) if eval_time > 0 else 0,
            'model': self.metadata.get('model')
        }


class OpenAIStreamResult(BaseStreamResult):
    """Implémentation pour OpenAI"""

    def _extract_text(self, chunk) -> str:
        if 'choices' in chunk and len(chunk['choices']) > 0:
            delta = chunk['choices'][0].get('delta', {})
            return delta.get('content', '')
        return ''

    def _is_done(self, chunk) -> bool:
        if 'choices' in chunk and len(chunk['choices']) > 0:
            return chunk['choices'][0].get('finish_reason') is not None
        return False

    def _extract_metadata(self, chunk) -> dict:
        return {
            'usage': chunk.get('usage', {}),
            'model': chunk.get('model'),
            'finish_reason': chunk.get('choices', [{}])[0].get('finish_reason')
        }

    def get_stats(self):
        if not self._consumed:
            for _ in self:
                pass

        if not self.metadata:
            return None

        usage = self.metadata.get('usage', {})
        return {
            'prompt_tokens': usage.get('prompt_tokens', 0),
            'completion_tokens': usage.get('completion_tokens', 0),
            'total_tokens': usage.get('total_tokens', 0),
            'model': self.metadata.get('model'),
            'finish_reason': self.metadata.get('finish_reason')
        }


class AnthropicStreamResult(BaseStreamResult):
    """Implémentation pour Anthropic (Claude)"""

    def _extract_text(self, chunk) -> str:
        if chunk.get('type') == 'content_block_delta':
            return chunk.get('delta', {}).get('text', '')
        return ''

    def _is_done(self, chunk) -> bool:
        return chunk.get('type') == 'message_stop'

    def _extract_metadata(self, chunk) -> dict:
        return {
            'usage': chunk.get('usage', {}),
            'model': chunk.get('model'),
            'stop_reason': chunk.get('stop_reason')
        }

    def get_stats(self):
        if not self._consumed:
            for _ in self:
                pass

        if not self.metadata:
            return None

        usage = self.metadata.get('usage', {})
        return {
            'input_tokens': usage.get('input_tokens', 0),
            'output_tokens': usage.get('output_tokens', 0),
            'model': self.metadata.get('model'),
            'stop_reason': self.metadata.get('stop_reason')
        }


class OllamaDownloadResult(BaseStreamResult):
    """Implémentation pour le téléchargement de modèles Ollama"""

    def __init__(self, generator):
        super().__init__(generator)
        self.download_progress = []
        self.current_status = None
        self.total_size = 0
        self.downloaded_size = 0

    def _extract_text(self, chunk) -> str:
        """Pour le download, on retourne le statut"""
        return chunk.get('status', '')

    def _is_done(self, chunk) -> bool:
        """Le téléchargement est terminé quand status = 'success'"""
        return chunk.get('status') == 'success'

    def _extract_metadata(self, chunk) -> dict:
        """Extrait les métadonnées finales du téléchargement"""
        return {
            'status': chunk.get('status'),
            'total_downloaded': self.downloaded_size,
            'total_size': self.total_size,
            'progress_history': self.download_progress
        }

    def __iter__(self):
        if self._consumed:
            for chunk in self.chunks:
                yield chunk
        else:
            for chunk in self.generator:
                self.current_status = chunk.get('status', '')
                self.chunks.append(chunk)

                # Suivi de la progression du téléchargement
                if 'total' in chunk:
                    self.total_size = max(self.total_size, chunk.get('total', 0))
                if 'completed' in chunk:
                    self.downloaded_size = chunk.get('completed', 0)
                    self.download_progress.append({
                        'digest': chunk.get('digest', ''),
                        'completed': self.downloaded_size,
                        'total': chunk.get('total', 0),
                        'percentage': round((self.downloaded_size / chunk.get('total', 1)) * 100, 2)
                    })

                if self._is_done(chunk):
                    self.metadata = self._extract_metadata(chunk)

                yield chunk

            self._consumed = True

    def get_stats(self):
        """Retourne les statistiques du téléchargement"""
        if not self._consumed:
            for _ in self:
                pass

        if not self.metadata:
            return None

        return {
            'status': self.metadata.get('status'),
            'total_size_bytes': self.metadata.get('total_size'),
            'total_size_mb': round(self.metadata.get('total_size', 0) / (1024 * 1024), 2),
            'num_progress_updates': len(self.download_progress)
        }

    def get_progress_percentage(self):
        """Retourne le pourcentage de progression actuel"""
        if self.total_size == 0:
            return 0
        return round((self.downloaded_size / self.total_size) * 100, 2)