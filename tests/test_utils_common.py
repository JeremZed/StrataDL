import pytest
import torch
import pandas as pd
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import tempfile
import shutil

# Import des fonctions à tester
from stratadl.core.utils.common import get_device, show_summary, save_results_to_csv


class TestGetDevice:
    """Tests pour la fonction get_device"""

    @patch('torch.cuda.is_available')
    def test_get_device_cuda_available(self, mock_cuda):
        """Test quand CUDA est disponible"""
        mock_cuda.return_value = True
        device = get_device()
        assert device.type == 'cuda'
        mock_cuda.assert_called_once()

    @patch('torch.cuda.is_available')
    def test_get_device_cuda_not_available(self, mock_cuda):
        """Test quand CUDA n'est pas disponible"""
        mock_cuda.return_value = False
        device = get_device()
        assert device.type == 'cpu'
        mock_cuda.assert_called_once()


class TestShowSummary:
    """Tests pour la fonction show_summary"""

    @patch('stratadl.core.utils.common.summary')
    def test_show_summary_calls_torchsummary(self, mock_summary):
        """Test que show_summary appelle bien summary avec les bons paramètres"""
        mock_model = Mock()
        input_size = (3, 224, 224)

        show_summary(mock_model, input_size)

        mock_summary.assert_called_once_with(mock_model, input_size)

    @patch('stratadl.core.utils.common.summary')
    def test_show_summary_different_input_sizes(self, mock_summary):
        """Test avec différentes tailles d'entrée"""
        mock_model = Mock()
        test_sizes = [(1, 28, 28), (3, 64, 64), (1, 100)]

        for size in test_sizes:
            show_summary(mock_model, size)

        assert mock_summary.call_count == len(test_sizes)


class TestSaveResultsToCSV:
    """Tests pour la fonction save_results_to_csv"""

    @pytest.fixture
    def temp_dir(self):
        """Crée un répertoire temporaire pour les tests"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def sample_results(self):
        """Résultats d'exemple pour les tests"""
        return {
            'config1': {
                'accuracy': 0.95,
                'loss': 0.05,
                'params': {
                    'learning_rate': 0.001,
                    'batch_size': 32
                },
                'epochs': [1, 2, 3]
            }
        }

    def test_save_results_creates_directory(self, temp_dir, sample_results):
        """Test que la fonction crée le répertoire si nécessaire"""
        csv_path = Path(temp_dir) / 'results' / 'test.csv'

        save_results_to_csv(sample_results, str(csv_path))

        assert csv_path.parent.exists()
        assert csv_path.exists()

    def test_save_results_creates_new_file(self, temp_dir, sample_results):
        """Test la création d'un nouveau fichier CSV"""
        csv_path = Path(temp_dir) / 'test.csv'

        save_results_to_csv(sample_results, str(csv_path))

        assert csv_path.exists()
        df = pd.read_csv(csv_path)
        assert len(df) == 1
        assert df['config_name'].iloc[0] == 'config1'

    def test_save_results_flattens_nested_dict(self, temp_dir, sample_results):
        """Test que les dictionnaires imbriqués sont bien aplatis"""
        csv_path = Path(temp_dir) / 'test.csv'

        save_results_to_csv(sample_results, str(csv_path))

        df = pd.read_csv(csv_path)
        assert 'params_learning_rate' in df.columns
        assert 'params_batch_size' in df.columns
        assert df['params_learning_rate'].iloc[0] == 0.001
        assert df['params_batch_size'].iloc[0] == 32

    def test_save_results_handles_lists(self, temp_dir, sample_results):
        """Test que les listes sont converties en JSON"""
        csv_path = Path(temp_dir) / 'test.csv'

        save_results_to_csv(sample_results, str(csv_path))

        df = pd.read_csv(csv_path)
        assert 'epochs' in df.columns
        epochs = json.loads(df['epochs'].iloc[0])
        assert epochs == [1, 2, 3]

    def test_save_results_appends_to_existing(self, temp_dir, sample_results):
        """Test l'ajout de résultats à un fichier existant"""
        csv_path = Path(temp_dir) / 'test.csv'

        # Première sauvegarde
        save_results_to_csv(sample_results, str(csv_path))

        # Deuxième sauvegarde
        new_results = {
            'config2': {
                'accuracy': 0.90,
                'loss': 0.10
            }
        }
        save_results_to_csv(new_results, str(csv_path))

        df = pd.read_csv(csv_path)
        assert len(df) == 2
        assert df['config_name'].tolist() == ['config1', 'config2']

    def test_save_results_handles_empty_file(self, temp_dir, sample_results):
        """Test le cas d'un fichier vide ou corrompu"""
        csv_path = Path(temp_dir) / 'test.csv'

        # Créer un fichier vide
        csv_path.touch()

        save_results_to_csv(sample_results, str(csv_path))

        df = pd.read_csv(csv_path)
        assert len(df) == 1
        assert df['config_name'].iloc[0] == 'config1'

    def test_save_results_handles_corrupted_csv(self, temp_dir, sample_results):
        """Test le cas d'un CSV corrompu"""
        csv_path = Path(temp_dir) / 'test.csv'

        # Créer un fichier avec du contenu invalide
        csv_path.write_text("invalid,csv,data\n,,,")

        # La fonction devrait gérer l'erreur et créer un nouveau fichier
        save_results_to_csv(sample_results, str(csv_path))

        df = pd.read_csv(csv_path)
        assert 'config_name' in df.columns

    def test_save_results_multiple_configs(self, temp_dir):
        """Test avec plusieurs configurations en une fois"""
        csv_path = Path(temp_dir) / 'test.csv'
        results = {
            'config1': {'accuracy': 0.95},
            'config2': {'accuracy': 0.90},
            'config3': {'accuracy': 0.85}
        }

        save_results_to_csv(results, str(csv_path))

        df = pd.read_csv(csv_path)
        assert len(df) == 3
        assert set(df['config_name']) == {'config1', 'config2', 'config3'}

    def test_save_results_complex_nested_structure(self, temp_dir):
        """Test avec une structure imbriquée complexe"""
        csv_path = Path(temp_dir) / 'test.csv'
        results = {
            'config1': {
                'metrics': {
                    'train': {
                        'accuracy': 0.95,
                        'loss': 0.05
                    },
                    'val': {
                        'accuracy': 0.90,
                        'loss': 0.10
                    }
                }
            }
        }

        save_results_to_csv(results, str(csv_path))

        df = pd.read_csv(csv_path)
        assert 'metrics_train_accuracy' in df.columns
        assert 'metrics_val_accuracy' in df.columns
        assert df['metrics_train_accuracy'].iloc[0] == 0.95
        assert df['metrics_val_accuracy'].iloc[0] == 0.90

    @patch('stratadl.core.utils.common.logging')
    def test_save_results_logs_success(self, mock_logging, temp_dir, sample_results):
        """Test que la sauvegarde est loggée"""
        csv_path = Path(temp_dir) / 'test.csv'

        save_results_to_csv(sample_results, str(csv_path))

        mock_logging.debug.assert_called_once()
        call_args = mock_logging.debug.call_args[0][0]
        assert 'Résultats sauvegardés' in call_args

    def test_save_results_default_path(self, temp_dir, sample_results, monkeypatch):
        """Test avec le chemin par défaut"""
        # Change le répertoire de travail
        monkeypatch.chdir(temp_dir)

        save_results_to_csv(sample_results)

        default_path = Path('results/training_results.csv')
        assert default_path.exists()


# Tests d'intégration
class TestIntegration:
    """Tests d'intégration entre les fonctions"""

    def test_full_workflow(self, tmp_path):
        """Test un workflow complet de sauvegarde"""
        csv_path = tmp_path / 'results.csv'

        # Première série de résultats
        results1 = {
            'experiment1': {
                'accuracy': 0.95,
                'config': {'lr': 0.001}
            }
        }
        save_results_to_csv(results1, str(csv_path))

        # Deuxième série
        results2 = {
            'experiment2': {
                'accuracy': 0.90,
                'config': {'lr': 0.01}
            }
        }
        save_results_to_csv(results2, str(csv_path))

        # Vérification finale
        df = pd.read_csv(csv_path)
        assert len(df) == 2
        assert df['accuracy'].tolist() == [0.95, 0.90]


if __name__ == '__main__':
    pytest.main([__file__, '-v'])