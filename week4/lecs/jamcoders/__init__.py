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


def build_cache() -> bool:
    """
    Build/rebuild all cache files by importing and initializing data.

    Returns:
        True if every cache was built, False if any step failed.
    """
    print("Building cache...")
    ok = True

    # Import datasets to trigger JSON generation
    try:
        from . import datasets
        # Access the lazy-loaded properties to trigger caching. Shakespeare backs
        # Days 2 and 3, so warm it here too -- not just the Day 1 Moby Dick data.
        _ = datasets.moby
        _ = datasets.patois
        _ = datasets.moby_tokenized  # This might take a while on first run
        _ = datasets.shake_sentences
        _ = datasets.shake_words
        print("✓ Built dataset caches: moby.json, patois.json, moby_tokenized.json, "
              "shakespeare.json, shakespeare_words.json")
    except Exception as e:
        print(f"✗ Error building dataset caches: {e}")
        ok = False

    # Import models to trigger n-gram data download
    try:
        from . import models
        # Load pretrained models to trigger downloads
        unigram = models.load_pretrained_unigram()
        print("✓ Downloaded n-gram data: count_1w.txt")

        # tril_model reads this JSON cache
        models._save_tril_model(unigram)
        print("✓ Built unigram cache: ngram_data/norvig_unigram_full.json")

        # Optionally build bigram model (this will be slower)
        print("Building bigram model from the Shakespeare corpus...")
        _ = models.load_pretrained_ngram(2)
        print("✓ Built bigram model")
    except Exception as e:
        print(f"✗ Error building model caches: {e}")
        ok = False

    print("\nCache building complete!" if ok else "\nCache building FAILED -- see errors above.")
    return ok


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
