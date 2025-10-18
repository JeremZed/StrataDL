import pytest
from typer.testing import CliRunner
from stratadl.cli.main import app

runner = CliRunner()

def test_show_version_dev(monkeypatch):
    """Test de la commande show-version quand le package n'est pas installé."""
    import stratadl.cli.main as main
    monkeypatch.setattr(main, "version", lambda _: (_ for _ in ()).throw(main.PackageNotFoundError()))
    result = runner.invoke(app, ["show-version"])
    assert result.exit_code == 0
    assert "StrataDL (dev mode)" in result.output


def test_show_version_ok(monkeypatch):
    """Test de la commande show-version avec version simulée."""
    monkeypatch.setattr("stratadl.cli.main.version", lambda _: "0.0.1")
    result = runner.invoke(app, ["show-version"])
    assert result.exit_code == 0
    assert "StrataDL version 0.0.1" in result.output

def test_serve_defaults():
    """Test de la commande serve avec valeurs par défaut."""
    result = runner.invoke(app, ["serve"])
    assert result.exit_code == 0
    assert "Lancement FastAPI" in result.output
    assert "Lancement UI" in result.output

def test_serve_custom_ports():
    """Test de la commande serve avec ports personnalisés."""
    result = runner.invoke(app, ["serve", "--port-api", "9000", "--port-ui", "9100"])
    assert result.exit_code == 0
    assert "http://127.0.0.1:9000" in result.output
    assert "http://127.0.0.1:9100" in result.output

def test_main_version_flag(monkeypatch):
    """Test du flag global --version."""
    monkeypatch.setattr("stratadl.cli.main.show_version", lambda: print("StrataDL version test"))
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "StrataDL version test" in result.output
