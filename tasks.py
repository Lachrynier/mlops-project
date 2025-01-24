import os
import glob

from invoke import Context, task

# Fix annoying bug https://github.com/pyinvoke/invoke/issues/833
import inspect

WINDOWS = os.name == "nt"
PROJECT_NAME = "proj"
PYTHON_VERSION = "3.11"


if not hasattr(inspect, "getargspec"):
    inspect.getargspec = inspect.getfullargspec


# Setup commands
@task
def create_environment(ctx: Context) -> None:
    """Create a new conda environment for project."""
    ctx.run(
        f"conda create --name {PROJECT_NAME} python={PYTHON_VERSION} pip invoke --no-default-packages --yes",
        echo=True,
        pty=not WINDOWS,
    )


@task
def all_requirements(ctx: Context) -> None:
    """Install project requirements."""
    ctx.run("pip install -U pip setuptools wheel", echo=True, pty=not WINDOWS)
    requirements_files = glob.glob("requirements*.txt")
    for requirements_file in requirements_files:
        ctx.run(f"pip install -r {requirements_file}", echo=True, pty=not WINDOWS)
    ctx.run("pip install -e .", echo=True, pty=not WINDOWS)


@task
def api_requirements(ctx: Context) -> None:
    """Install project requirements."""
    ctx.run("pip install -U pip setuptools wheel", echo=True, pty=not WINDOWS)
    ctx.run("pip install -r requirements_api.txt", echo=True, pty=not WINDOWS)
    ctx.run("pip install -e .", echo=True, pty=not WINDOWS)


@task
def frontend_requirements(ctx: Context) -> None:
    """Install project requirements."""
    ctx.run("pip install -U pip setuptools wheel", echo=True, pty=not WINDOWS)
    ctx.run("pip install -r requirements_frontend.txt", echo=True, pty=not WINDOWS)
    ctx.run("pip install -e .", echo=True, pty=not WINDOWS)


@task
def train_requirements(ctx: Context) -> None:
    """Install project requirements."""
    ctx.run("pip install -U pip setuptools wheel", echo=True, pty=not WINDOWS)
    ctx.run("pip install -r requirements.txt", echo=True, pty=not WINDOWS)
    ctx.run("pip install -e .", echo=True, pty=not WINDOWS)


@task
def dev_requirements(ctx: Context) -> None:
    """Install development requirements."""
    ctx.run('pip install -e .["dev"]', echo=True, pty=not WINDOWS)


# Project commands
@task
def download_data(ctx: Context) -> None:
    """Download data."""
    ctx.run(f"python src/{PROJECT_NAME}/data.py data/raw", echo=True, pty=not WINDOWS)


@task
def train(ctx: Context) -> None:
    """Train model."""
    ctx.run(f"python src/{PROJECT_NAME}/train.py", echo=True, pty=not WINDOWS)


@task
def test(ctx: Context) -> None:
    """Run tests."""
    ctx.run("coverage run -m pytest tests/", echo=True, pty=not WINDOWS)
    ctx.run("coverage report -m", echo=True, pty=not WINDOWS)


@task
def generate_test_dataset(ctx: Context, classes: int = 5, images_per_class: int = 2) -> None:
    """Generate a reduced dataset for unit tests"""
    import proj.data
    import tarfile

    os.makedirs("data/raw_test", exist_ok=True)
    tar = tarfile.open("data/raw_test/256_ObjectCategories.tar", "w|")
    dataset = proj.data.Caltech256("data/raw", download=True)

    selected_counts = classes * [0]

    for img, target in zip(dataset.imgs, dataset.targets):
        if target >= classes:
            continue

        if selected_counts[target] >= images_per_class:
            continue

        member = dataset.tar.getmember(img)
        tar.addfile(member, dataset.tar.extractfile(member))
        selected_counts[target] += 1


@task
def docker_build(ctx: Context, progress: str = "plain") -> None:
    """Build docker images."""
    ctx.run(
        f"docker build -t train:latest . -f dockerfiles/train.dockerfile --progress={progress}",
        echo=True,
        pty=not WINDOWS,
    )
    ctx.run(
        f"docker build -t api:latest . -f dockerfiles/api.dockerfile --progress={progress}", echo=True, pty=not WINDOWS
    )


# Documentation commands
@task(dev_requirements)
def build_docs(ctx: Context) -> None:
    """Build documentation."""
    ctx.run("mkdocs build --config-file docs/mkdocs.yaml --site-dir build", echo=True, pty=not WINDOWS)


@task(dev_requirements)
def serve_docs(ctx: Context) -> None:
    """Serve documentation."""
    ctx.run("mkdocs serve --config-file docs/mkdocs.yaml", echo=True, pty=not WINDOWS)
