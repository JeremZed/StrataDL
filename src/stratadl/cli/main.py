import typer
from importlib.metadata import version, PackageNotFoundError

app = typer.Typer(help="StrataDL Framework CLI")

@app.command()
def show_version():
    """
    Permet d'afficher la version du package.
    """
    try:
        typer.echo(f"StrataDL version {version('stratadl')}")
    except PackageNotFoundError:
        typer.echo("StrataDL (dev mode)")

@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host", help="Adresse IP pour l'API et UI"),
    port_ui: int = typer.Option(8500, "--port-ui", help="Port UI"),
    port_api: int = typer.Option(8000, "--port-api", help="Port FastAPI API")
):
    """
    Permet de lancer API + UI
    """
    typer.echo("Simulation du lancement de l'API et de l'UI...")
    typer.echo(f"- Lancement FastAPI sur  http://{host}:{port_api}...")
    typer.echo(f"- Lancement UI sur http://{host}:{port_ui}...")

@app.callback(invoke_without_command=True)
def main(
    version_flag: bool = typer.Option(False, "--version", help="Afficher la version")
    ):

    if version_flag:
        show_version()