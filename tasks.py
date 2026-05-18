from __future__ import annotations

from pathlib import Path

from invoke import task


PYTHON = ".venv\\Scripts\\python" if Path(".venv\\Scripts\\python.exe").exists() else "python"
UVICORN = ".venv\\Scripts\\uvicorn" if Path(".venv\\Scripts\\uvicorn.exe").exists() else "uvicorn"
STREAMLIT = (
    ".venv\\Scripts\\streamlit" if Path(".venv\\Scripts\\streamlit.exe").exists() else "streamlit"
)


def run_python(context, script: str) -> None:
    context.run(f"{PYTHON} {script}", pty=False)


@task
def install(context) -> None:
    """Install project dependencies into the local virtual environment."""
    context.run(f"{PYTHON} -m pip install -r requirements.txt", pty=False)


@task
def ingest(context) -> None:
    """Fetch UCI data, validate it, create features, and load DuckDB."""
    run_python(context, "ingest.py")


@task
def validate(context) -> None:
    """Run data quality checks against the raw local CSV."""
    run_python(context, "validate.py")


@task
def train(context) -> None:
    """Train classification/regression models and generate ML reports."""
    run_python(context, "train_models.py")


@task
def survival(context) -> None:
    """Run Kaplan-Meier survival analysis."""
    run_python(context, "survival_analysis.py")


@task
def lineage(context) -> None:
    """Generate lineage metadata and a DVC-style lock file."""
    run_python(context, "lineage.py")


@task(name="docs")
def docs_task(context) -> None:
    """Generate dataset schema and project documentation."""
    run_python(context, "generate_docs.py")


@task
def all(context) -> None:
    """Run the complete local workflow."""
    run_python(context, "run_all.py")


@task
def api(context, reload: bool = True) -> None:
    """Start the FastAPI prediction service."""
    reload_flag = " --reload" if reload else ""
    context.run(f"{UVICORN} serve_api:app{reload_flag}", pty=False)


@task
def streamlit(context) -> None:
    """Start the Streamlit dashboard."""
    context.run(f"{STREAMLIT} run streamlit_app.py", pty=False)


@task
def compile(context) -> None:
    """Compile all Python scripts as a quick syntax check."""
    context.run(
        f"{PYTHON} -m py_compile "
        "ingest.py validate.py train_models.py survival_analysis.py "
        "serve_api.py streamlit_app.py lineage.py generate_docs.py run_all.py tasks.py",
        pty=False,
    )
