import shutil
import subprocess
from pathlib import Path


# Project root
ROOT = Path(__file__).resolve().parent.parent

# Where the raw documentation will be stored
RAW_DIR = ROOT / "data" / "raw"

# Temporary directory for cloned repositories
CLONE_DIR = ROOT / ".docs_repos"


REPOSITORIES = {
    "fastapi": {
        "url": "https://github.com/fastapi/fastapi.git",
        "docs_path": "docs/en/docs",
    },
    "docker": {
        "url": "https://github.com/docker/docs.git",
        "docs_path": "content",
    },
    "pydantic": {
        "url": "https://github.com/pydantic/pydantic.git",
        "docs_path": "docs",
    },
    "sqlalchemy": {
        "url": "https://github.com/sqlalchemy/sqlalchemy.git",
        "docs_path": "doc/build",
    },
    "qdrant": {
        "url": "https://github.com/qdrant/qdrant-client.git",
        "docs_path": "docs",
    },
    "pytorch": {
        "url": "https://github.com/pytorch/pytorch.git",
        "docs_path": "docs",
    },
    "transformers": {
        "url": "https://github.com/huggingface/transformers.git",
        "docs_path": "docs/source/en",
    },
    "postgresql": {
        "url": "https://github.com/postgres/postgres.git",
        "docs_path": "doc",
    },
}


def run_command(command):
    """Run a shell command and stop if it fails."""
    print(f"\n$ {' '.join(command)}")
    subprocess.run(command, check=True)


def clone_repository(name, url):
    """Clone a repository if it doesn't already exist."""
    repo_dir = CLONE_DIR / name

    if repo_dir.exists():
        print(f"\n{name}: repository already exists, skipping clone.")
        return repo_dir

    run_command([
        "git",
        "clone",
        "--depth",
        "1",
        url,
        str(repo_dir),
    ])

    return repo_dir


def copy_markdown_files(name, repo_dir, docs_path):
    """Copy all Markdown files from the documentation directory."""
    source_dir = repo_dir / docs_path
    destination_dir = RAW_DIR / name

    if not source_dir.exists():
        print(f"WARNING: Documentation path not found: {source_dir}")
        return 0

    destination_dir.mkdir(parents=True, exist_ok=True)

    count = 0

    for md_file in source_dir.rglob("*.md"):
        relative_path = md_file.relative_to(source_dir)

        destination_file = destination_dir / relative_path
        destination_file.parent.mkdir(parents=True, exist_ok=True)

        shutil.copy2(md_file, destination_file)

        count += 1

    return count


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    CLONE_DIR.mkdir(parents=True, exist_ok=True)

    total_files = 0

    for name, config in REPOSITORIES.items():

        print("\n" + "=" * 60)
        print(f"Processing: {name}")
        print("=" * 60)

        repo_dir = clone_repository(
            name,
            config["url"],
        )

        count = copy_markdown_files(
            name,
            repo_dir,
            config["docs_path"],
        )

        print(f"Copied {count} Markdown files for {name}")

        total_files += count

    print("\n" + "=" * 60)
    print(f"Finished! Total Markdown files: {total_files}")
    print(f"Documentation stored in: {RAW_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
