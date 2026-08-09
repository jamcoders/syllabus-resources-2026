# JamCoders Week 4: Language Models & Word Networks

## Installation

```bash
# Install uv if not already installed
# curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and navigate to the repository
cd week4-lecs-2025

# Install Python 3.12 (if needed)
uv python install 3.12

# Install all dependencies from pyproject.toml
uv sync

# Run Jupyter notebook (use nbclassic to avoid compatibility issues)
uv run jupyter nbclassic
```

## Lecture Notebooks

### Day 1a: Introduction to Language & Text Processing
- **File**: `lec_w4d1a.ipynb`
- **Topics**: Introduction to language algorithms, working with text data, import statements
- **Key concepts**: String processing, counting words, edge cases in language
- **Datasets introduced**: Moby Dick sentences, Jamaican Patois NLI (Armstrong, Hewitt, Manning; EMNLP Findings 2022)

### Day 1b: Randomness & Probability with Skittles
- **File**: `lec_w4d1b.ipynb`
- **Topics**: Digital randomness, probability distributions, sampling
- **Key concepts**: Building models from data, visualization of distributions
- **Interactive demos**: Skittles sampling simulation showing convergence to true distribution

### Day 2: Language Modeling - From Skittles to Shakespeare
- **File**: `lec_w4d2.ipynb`
- **Topics**: Language models, unigrams, bigrams, n-grams
- **Key concepts**: Context in language, probability-based text generation
- **Progression**: Unigram (bag of words) → Bigram → Trigram → 4-gram models
- **Includes**: GPT-2 demo, comparison of different n-gram models

### Day 3: Problem Solving Strategies and Memoization
- **File**: `lec_w4d3.ipynb`
- **Topics**: Problem-solving methodology, memoization, word segmentation
- **Key concepts**: 5-step coding approach, recursive algorithms, optimization
- **Classic problem**: Word segmentation ("applemonapp" → ["apple", "lemon", "app"])

## Labs

The corresponding labs for this week, created by the wonderful TA team, will be available at https://github.com/jamcoders/labs-2025

---

## Wrapper Modules

The `jamcoders` package contains three pedagogically useful wrapper modules that provide convenient access to datasets and utilities:

### datasets.py
- **Purpose**: Provides easy access to text datasets without overwhelming students with file I/O
- **Datasets available**:
  - `moby`: Sentences from Moby Dick (used Day 1 only)
  - `patois`: Jamaican Patois NLI sentences (Armstrong, Hewitt, Manning; EMNLP Findings 2022)
  - `shake`: All Shakespeare words as a flat list (from John DeNero, via Peter Norvig's website)
  - `shake_sentences`: Shakespeare sentences (from John DeNero, via Peter Norvig's website)
  - `shake_words`: Shakespeare tokenized by sentence
- **Features**: Lazy loading, automatic downloading, caching for performance

### models.py
- **Purpose**: Language modeling utilities and pre-trained models
- **Key functions**:
  - `build_model()`: Creates probability distributions from data
  - `visualize_model()`: Bar chart visualization of word distributions
  - `generate_from_ngram_model()`: Text generation from n-gram models
  - `gpt2()`: Simple wrapper for GPT-2 text generation
- **Pre-trained models**:
  - `tril_model`: Unigram model trained on 1 trillion words (from Google, via Peter Norvig's website)
  - `load_pretrained_ngram(n)`: Load n-gram models (bigram, trigram, etc.)

### random.py
- **Purpose**: Random sampling utilities with pedagogical visualizations
- **Key functions**:
  - `sample_from_list()`: Uniform sampling from lists
  - `sample_from_dict()`: Weighted sampling from probability dictionaries
  - `visualize()`: Interactive Skittles visualization showing sampling convergence
- **Teaching tool**: The visualization helps students understand how sampling approaches true distributions

---