# JamCoders Week 4: Language Models & Word Networks

This repository tracks the **current** state of the JamCoders Week 4 module. Each
year the module is taught, that year's state is frozen as a git tag.

| Year | Status | Snapshot |
|------|--------|----------|
| 2025 | Taught | [`2025-taught`](https://github.com/jamcoders/week4-lecs/tree/2025-taught) |
| 2026 | In progress | `2026-taught` — *TBD, tagged after the final day*; current material lives on [`2026`](https://github.com/jamcoders/week4-lecs/tree/2026) |

## Installation

```bash
# Install uv if not already installed
# curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and navigate to the repository
cd week4-lecs

# Install Python 3.12 (if needed)
uv python install 3.12

# Install all dependencies from pyproject.toml
uv sync

# Run Jupyter notebook (use nbclassic to avoid compatibility issues)
uv run jupyter nbclassic
```

### Internet access

`uv sync` needs internet, but everything after it is offline-friendly: no
dependency is fetched from a URL, so `uv run` works with no network at all.

The datasets and pre-trained models are downloaded **once** and then cached
inside the package (see `jamcoders/*.json` and `jamcoders/ngram_data/`, all
gitignored). Only that first download needs internet; if it fails, the notebook
prints a plain-English message saying so rather than a network traceback.

Before teaching on an unreliable connection, warm the caches while you still
have one:

```bash
uv run python -m jamcoders build_cache   # also: list_cache, clear_cache
```

A student distribution of this material needs only the notebooks, `jamcoders/*.py`, and
the four cache files the notebooks actually open:

    jamcoders/moby.json                            (Day 1a)
    jamcoders/patois.json                          (Day 1a)
    jamcoders/shakespeare_words.json               (Days 2 and 3)
    jamcoders/ngram_data/norvig_unigram_full.json  (Day 2, `tril_model`)

`shakespeare.json` and `ngram_data/count_1w.txt` are needed only for generation (included for future use).
The `gpt2()` demo in Day 2 is `distilgpt2` saved in `~/.cache/huggingface`.

## Lecture Notebooks

### Day 1a: Introduction to Language & Text Processing
- **File**: `w4d1a.ipynb`
- **Topics**: Introduction to language algorithms, working with text data, import statements
- **Key concepts**: String processing, counting words, edge cases in language
- **Datasets introduced**: Moby Dick sentences, Jamaican Patois NLI (Armstrong, Hewitt, Manning; EMNLP Findings 2022)

### Day 1b: Randomness & Probability with Skittles
- **File**: `w4d1b.ipynb`
- **Topics**: Digital randomness, probability distributions, sampling
- **Key concepts**: Building models from data, visualization of distributions
- **Interactive demos**: Skittles sampling simulation

### Day 2: Language Modeling - From Skittles to Shakespeare
- **File**: `lec_w4d2.ipynb`
- **Topics**: Language models, unigrams, bigrams, n-grams
- **Key concepts**: Context in language, probability-based text generation

### Day 3: Problem Solving Strategies and Memoization
- **File**: `lec_w4d3.ipynb`
- **Topics**: Problem-solving methodology, dynamic programming, memoization

---

## Wrapper Modules

The `jamcoders` package contains three pedagogically useful wrapper modules that provide convenient access to datasets and utilities:

### datasets.py
- Includes:
  - `moby`: Sentences from Moby Dick (used Day 1 only)
  - `patois`: Jamaican Patois NLI sentences (Armstrong, Hewitt, Manning; EMNLP Findings 2022)
  - `shake`: All Shakespeare words as a flat list (from John DeNero, via Peter Norvig's website)
  - `shake_sentences`: Shakespeare sentences (from John DeNero, via Peter Norvig's website)
  - `shake_words`: Shakespeare tokenized by sentence
- **Tokenization**: `tokenize(text)` lowercases and splits on punctuation, keeping
  contractions and possessives whole (`isn't`, `hamlet's`, `o'er`). It is a single
  regex so students can read it, and re-tokenizing all of
  Shakespeare takes under a second. Cached token files record `tokenizer_version`
  and rebuild themselves if that constant in `datasets.py` changes.

### models.py
- Includes:
  - `build_ngram_model_from_corpus(sentences, n)`
  - `build_better_ngram_model(...)`: same, with rare n-grams filtered out
  - `visualize_model()`: Bar chart visualization of word distributions
  - `generate_from_ngram_model()`: Text generation from n-gram models
  - `gpt2()`: Simple wrapper for GPT-2 text generation
- **Pre-trained models**:
  - `tril_model`: Unigram model trained on 1 trillion words (from Google, via Peter Norvig's website)
  - `load_pretrained_ngram(n)`: Load n-gram models (bigram, trigram, etc.)

### random.py
- Includes:
  - `sample_from_list()`: Sample from list uniformly at random
  - `sample_from_dict()`: Sample key according to probabilities in values
  - `visualize()`: Skittles visualization
