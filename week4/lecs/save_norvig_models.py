#!/usr/bin/env python3
"""Save full Norvig models as JSON files"""

import json
from pathlib import Path
from jamcoders.models import load_pretrained_unigram, load_pretrained_bigram

print("Loading and saving Norvig models as JSON...")
print("=" * 60)

# Create output directory
output_dir = Path("jamcoders/ngram_data")
output_dir.mkdir(exist_ok=True)

# Load and save unigram model
print("\n1. Loading full unigram model...")
unigram = load_pretrained_unigram()
print(f"   Loaded {len(unigram):,} words")

unigram_file = output_dir / "norvig_unigram_full.json"
print(f"   Saving to {unigram_file}...")
with open(unigram_file, 'w') as f:
    json.dump(unigram, f)
print(f"   ✓ Saved! File size: {unigram_file.stat().st_size / 1024 / 1024:.1f} MB")

# Load and save bigram model
print("\n2. Loading full bigram model...")
bigram = load_pretrained_bigram()
print(f"   Loaded {len(bigram):,} contexts")

bigram_file = output_dir / "norvig_bigram_full.json"
print(f"   Saving to {bigram_file}...")
with open(bigram_file, 'w') as f:
    json.dump(bigram, f)
print(f"   ✓ Saved! File size: {bigram_file.stat().st_size / 1024 / 1024:.1f} MB")

print("\n" + "=" * 60)
print("Both models saved successfully!")