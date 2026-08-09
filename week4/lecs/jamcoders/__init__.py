# JamCoders package
import json
import shutil
from pathlib import Path
from typing import List


def clear_cache() -> None:
    """
    Clear all cached data files (JSON files and downloaded n-gram data).
    """
    cache_cleared = []

    # Get the jamcoders package directory
    package_dir = Path(__file__).parent

    # Clear JSON files
    json_files = list(package_dir.glob("*.json"))
    for json_file in json_files:
        json_file.unlink()
        cache_cleared.append(str(json_file.name))

    # Clear ngram_data directory
    ngram_data_dir = package_dir / "ngram_data"
    if ngram_data_dir.exists():
        shutil.rmtree(ngram_data_dir)
        cache_cleared.append("ngram_data/")

    if cache_cleared:
        print(f"Cleared cache: {', '.join(cache_cleared)}")
    else:
        print("No cache files found to clear.")


def build_cache() -> None:
    """
    Build/rebuild all cache files by importing and initializing data.
    """
    print("Building cache...")

    # Import datasets to trigger JSON generation
    try:
        from . import datasets
        # Access the lazy-loaded properties to trigger caching
        _ = datasets.moby
        _ = datasets.patois
        _ = datasets.moby_tokenized  # This might take a while on first run
        print("✓ Built dataset caches: moby.json, patois.json, moby_tokenized.json")
    except Exception as e:
        print(f"✗ Error building dataset caches: {e}")

    # Import models to trigger n-gram data download
    try:
        from . import models
        # Load pretrained models to trigger downloads
        _ = models.load_pretrained_unigram()
        print("✓ Downloaded n-gram data: count_1w.txt")

        # Optionally build bigram model (this will be slower)
        print("Building bigram model from Moby Dick corpus...")
        _ = models.load_pretrained_ngram(2)
        print("✓ Built bigram model")
    except Exception as e:
        print(f"✗ Error building model caches: {e}")

    print("\nCache building complete!")


def list_cache() -> List[str]:
    """
    List all cache files currently present.
    
    Returns:
        List of cache file paths
    """
    cache_files = []

    # Get the jamcoders package directory
    package_dir = Path(__file__).parent

    # List JSON files
    json_files = list(package_dir.glob("*.json"))
    cache_files.extend([str(f.relative_to(package_dir)) for f in json_files])

    # List ngram_data contents
    ngram_data_dir = package_dir / "ngram_data"
    if ngram_data_dir.exists():
        ngram_files = list(ngram_data_dir.glob("*"))
        cache_files.extend([str(f.relative_to(package_dir)) for f in ngram_files])

    return cache_files


# CLI-style interface
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m jamcoders [clear_cache|build_cache|list_cache]")
        sys.exit(1)

    command = sys.argv[1]

    if command == "clear_cache":
        clear_cache()
    elif command == "build_cache":
        build_cache()
    elif command == "list_cache":
        files = list_cache()
        if files:
            print("Cache files:")
            for f in sorted(files):
                print(f"  - {f}")
        else:
            print("No cache files found.")
    else:
        print(f"Unknown command: {command}")
        print("Available commands: clear_cache, build_cache, list_cache")
        sys.exit(1)
