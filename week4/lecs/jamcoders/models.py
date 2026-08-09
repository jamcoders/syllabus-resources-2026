# Language modeling utilities for JamCoders
import random
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import urllib.request
from tqdm import tqdm
from better_profanity import profanity
import json

# Constants for vocabulary size limits
UNIGRAM_VOCAB_SIZE = 1000  # Top k words to keep for unigram model
BIGRAM_VOCAB_SIZE = 1000  # Top k words to consider for bigram model (up to 1M possible bigrams)
TRIGRAM_VOCAB_SIZE = 500  # Default vocab size for trigrams
FOURGRAM_VOCAB_SIZE = 500  # Default vocab size for 4-grams

# Initialize profanity filter
profanity.load_censor_words()

# Cache for Norvig models
_tril_model_cache = None
_norvig_bigrams_cache = None


def _get_data_dir():
    """Get the directory for storing downloaded data"""
    data_dir = Path(__file__).parent / 'ngram_data'
    data_dir.mkdir(exist_ok=True)
    return data_dir


def _download_norvig_data():
    """Download Norvig's n-gram data if not already present"""
    data_dir = _get_data_dir()
    base_url = "https://norvig.com/ngrams/"

    files_to_download = {
        'count_1w.txt': 'count_1w.txt',
        'count_2w.txt': 'count_2w.txt'
    }

    for filename, url_path in files_to_download.items():
        file_path = data_dir / filename
        if not file_path.exists():
            print(f"Downloading {filename}...")
            url = base_url + url_path
            urllib.request.urlretrieve(url, str(file_path))
            print(f"Downloaded {filename}")

    return data_dir


def _load_tril_model(max_items=None, filter_bad_words=True):
    """Load Norvig's unigram counts from count_1w.txt
    
    Returns:
        Dictionary mapping words to probabilities
    """
    data_dir = _download_norvig_data()
    file_path = data_dir / 'count_1w.txt'

    counts = {}
    total = 0
    items_loaded = 0

    with open(file_path, 'r', encoding='utf-8') as f:
        # Read more lines than needed to account for filtering
        read_limit = max_items * 2 if max_items and filter_bad_words else max_items

        # Handle both real file objects and mocked iterators
        if hasattr(f, 'readlines'):
            lines = f.readlines()[:read_limit] if read_limit else f.readlines()
        else:
            # For mocked iterators, convert to list
            lines = list(f)

        for line in tqdm(lines, desc="Loading unigrams", unit="words"):
            if max_items and items_loaded >= max_items:
                break

            parts = line.strip().split('\t')
            if len(parts) == 2:
                word, count = parts[0], int(parts[1])

                # Skip bad words if filtering is enabled
                if filter_bad_words and profanity.contains_profanity(word):
                    continue

                counts[word] = count
                total += count
                items_loaded += 1

    # Convert to probabilities
    probs = {word: count / total for word, count in counts.items()}
    return probs


def visualize_model(word_data: Dict[str, float], top_n: int = 15) -> None:
    """
    Visualize word distribution as a bar chart.
    
    Args:
        word_data: Dictionary mapping words to counts or probabilities
        top_n: Number of top words to display
    """
    # If values are counts, convert to probabilities
    total = sum(word_data.values())
    if total > 1.1:  # Likely counts, not probabilities
        word_probs = {word: count / total for word, count in word_data.items()}
    else:
        word_probs = word_data

    # Get top N words
    top_words = sorted(word_probs.items(), key=lambda x: x[1], reverse=True)[:top_n]
    words, probs = zip(*top_words)

    # Create bar chart
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(words)), probs)
    plt.xticks(range(len(words)), words, rotation=45, ha='right')
    plt.ylabel('Probability')
    plt.title(f'Top {top_n} Words by Frequency')
    plt.tight_layout()

    # Add value labels on bars
    for bar, prob in zip(bars, probs):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{prob:.3f}', ha='center', va='bottom', fontsize=8)

    plt.show()


def infer_ngram_order(model: Dict) -> int:
    """
    Infer the order (n) of an n-gram model from its structure and validate it.
    
    Args:
        model: N-gram model dictionary
        
    Returns:
        The order n of the n-gram model
        
    Raises:
        AssertionError: If the model structure is invalid or probabilities don't sum to 1
    """
    assert isinstance(model, dict), f"Model must be a dictionary, not {type(model)}"
    assert len(model) > 0, f"Model must not be empty."

    first_key = next(iter(model))
    first_value = model[first_key]

    if isinstance(first_key, str):
        if isinstance(first_value, (int, float)):
            # Unigram model: {word: probability}
            # Validate all entries
            total_prob = 0.0
            for word, prob in model.items():
                assert isinstance(word, str), f"Unigram keys must be strings, got {type(word)}"
                assert isinstance(prob, (int, float)), f"Unigram values must be numbers, got {type(prob)}"
                assert prob >= 0, f"Probability must be non-negative, got {prob}"
                total_prob += prob
            assert abs(total_prob - 1.0) < 0.01, f"Unigram probabilities must sum to 1.0, got {total_prob}"
            return 1

        elif isinstance(first_value, dict):
            # Bigram model: {word: {next_word: probability}}
            # Validate all entries
            for context, next_words in model.items():
                assert isinstance(context, str), f"Bigram keys must be strings, got {type(context)}"
                assert isinstance(next_words, dict), f"Bigram values must be dictionaries, got {type(next_words)}"

                # Validate inner dictionary
                total_prob = 0.0
                for word, prob in next_words.items():
                    assert isinstance(word, str), f"Next word must be string, got {type(word)}"
                    assert isinstance(prob, (int, float)), f"Probability must be number, got {type(prob)}"
                    assert prob >= 0, f"Probability must be non-negative, got {prob}"
                    total_prob += prob
                assert abs(
                    total_prob - 1.0) < 0.01, f"Probabilities for context '{context}' must sum to 1.0, got {total_prob}"
            return 2

        else:
            raise ValueError(
                f"Invalid model structure for string keys: values must be numbers (unigram) or dicts (bigram), got {type(first_value)}")

    elif isinstance(first_key, tuple):
        # N-gram model for n > 2
        n = len(first_key) + 1
        assert n >= 3, f"Invalid n-gram model: tuple keys must have length >= 2, got {len(first_key)}"

        # Validate all entries
        expected_tuple_len = len(first_key)
        for context, next_words in model.items():
            assert isinstance(context, tuple), f"N-gram keys must be tuples, got {type(context)}"
            assert len(
                context) == expected_tuple_len, f"All context tuples must have length {expected_tuple_len}, got {len(context)}"
            assert all(isinstance(w, str) for w in context), f"Context tuple must contain only strings"
            assert isinstance(next_words, dict), f"N-gram values must be dictionaries, got {type(next_words)}"

            # Validate inner dictionary
            total_prob = 0.0
            for word, prob in next_words.items():
                assert isinstance(word, str), f"Next word must be string, got {type(word)}"
                assert isinstance(prob, (int, float)), f"Probability must be number, got {type(prob)}"
                assert prob >= 0, f"Probability must be non-negative, got {prob}"
                total_prob += prob
            assert abs(
                total_prob - 1.0) < 0.01, f"Probabilities for context {context} must sum to 1.0, got {total_prob}"
        return n

    else:
        raise ValueError(
            f"Invalid model structure: keys must be strings (unigram/bigram) or tuples (n-gram), got {type(first_key)}")


def generate_from_ngram_model(model: Dict, prefix: str, gen_length: int,
                              avoid_early_end: bool = True) -> str:
    """
    Generate text by sampling from an n-gram model starting from a given prefix.
    
    Args:
        model: N-gram model (automatically detects n from model structure)
               - Unigram: {word: probability}
               - Bigram: {word: {next_word: probability}}
               - N-gram (n>2): {tuple_context: {next_word: probability}}
        prefix: Prefix text to start generation (must be lowercase words separated by spaces)
        gen_length: Number of words to generate from the prefix
        avoid_early_end: If True, reduce probability of selecting <END> until near target length
        
    Returns:
        Generated text as a string (prefix + generated words)
    """
    # Assert prefix is in correct format (lowercase words and spaces only)
    assert all(c.islower() or c == ' ' for c in prefix), "Prefix must contain only lowercase letters and spaces"

    # Infer n from model structure
    n = infer_ngram_order(model)

    # Split prefix into words
    prefix_words = prefix.split() if prefix else []

    # Special handling for unigrams
    if n == 1:
        # For unigram, just sample independently
        generated = list(prefix_words)  # Start with prefix

        words = list(model.keys())
        probs = list(model.values())

        # Generate gen_length words
        generated_words = random.choices(words, weights=probs, k=gen_length)
        generated.extend(generated_words)

        return ' '.join(generated)

    # Special handling for bigrams
    elif n == 2:
        # Start with last word of prefix or <START>
        current = prefix_words[-1] if prefix_words else '<START>'
        generated = list(prefix_words)  # Start with prefix

        for _ in range(gen_length):
            if current not in model:
                break

            next_words = list(model[current].keys())
            next_probs = list(model[current].values())

            # Assert probabilities are normalized
            prob_sum = sum(next_probs)
            assert abs(prob_sum - 1.0) < 0.01, f"Probabilities not normalized: sum = {prob_sum}"

            current = random.choices(next_words, weights=next_probs)[0]

            if current == '<END>':
                break
            generated.append(current)

        return ' '.join(generated)

    # For n > 2, use the general n-gram approach
    if prefix_words:
        # Use last (n-1) words from prefix as context
        context = prefix_words[-(n - 1):] if len(prefix_words) >= n - 1 else prefix_words
        # Pad with <START> if needed
        while len(context) < n - 1:
            context = ['<START>'] + context
        context = tuple(context)
    else:
        context = tuple(['<START>'] * (n - 1))

    # Start with the prefix words
    generated = list(prefix_words)

    # Generate gen_length additional words
    for i in range(gen_length):
        # Try to find a valid context, backing off if necessary
        current_context = context
        found = False

        # Try progressively shorter contexts (backoff)
        for backoff_level in range(n - 1):
            if current_context in model:
                found = True
                break
            # Back off by removing the first word
            if len(current_context) > 1:
                current_context = current_context[1:]

        if not found:
            # If we can't find any context, try to restart from a random valid context
            # Prefer contexts that don't immediately lead to END
            valid_contexts = [ctx for ctx in model.keys()
                              if ctx[0] != '<END>' and not (len(model[ctx]) == 1 and '<END>' in model[ctx])]
            if not valid_contexts:
                valid_contexts = [ctx for ctx in model.keys() if ctx[0] != '<END>']
            if valid_contexts:
                context = random.choice(valid_contexts)
                current_context = context
            else:
                break

        next_words = list(model[current_context].keys())
        next_probs = list(model[current_context].values())

        if not next_words:
            break

        # Assert probabilities are normalized (before any modification)
        prob_sum = sum(next_probs)
        assert abs(prob_sum - 1.0) < 0.01, f"Probabilities not normalized: sum = {prob_sum}"

        # Optionally reduce END probability if we're not near the target length
        if avoid_early_end and '<END>' in next_words and i < gen_length * 0.7:
            end_idx = next_words.index('<END>')
            # Reduce END probability by 90% if we're far from target
            next_probs[end_idx] *= 0.1
            # Renormalize
            total = sum(next_probs)
            next_probs = [p / total for p in next_probs]

        next_word = random.choices(next_words, weights=next_probs)[0]

        if next_word == '<END>':
            break

        generated.append(next_word)

        # Update context by sliding window
        context = context[1:] + (next_word,)

    return ' '.join(generated)


# Core n-gram building functions (based on pedagogical code)
def prepare_for_ngrams(sentences: List[List[str]], n: int) -> List[List[str]]:
    """
    Prepares sentences for n-gram modeling by adding special tokens.
    
    Args:
        sentences: List of sentences, where each sentence is a list of words
        n: The n-gram size (2 for bigrams, 3 for trigrams, etc.)
    
    Returns:
        List of padded sentences ready for n-gram extraction
    """
    prepared = []
    for sentence in tqdm(sentences, desc=f"Preparing sentences for {n}-grams", unit="sentences"):
        # Use <START> and <END> for consistency with existing code
        padded = ['<START>'] * (n - 1) + sentence + ['<END>']
        prepared.append(padded)
    return prepared


def build_ngram_counts(sentences: List[List[str]], n: int, vocab_limit: Optional[int] = None) -> Dict:
    """
    Build n-gram counts from prepared sentences.
    
    Args:
        sentences: List of prepared sentences (already padded)
        n: Size of n-grams
        vocab_limit: If set, only include n-grams where all words are in top vocab_limit words
        
    Returns:
        Dictionary mapping context tuples to word counts
    """
    # If vocab_limit is set, first get vocabulary
    valid_vocab = None
    if vocab_limit:
        # Count word frequencies (excluding special tokens)
        word_counts = defaultdict(int)
        for sentence in tqdm(sentences, desc="Counting word frequencies", unit="sentences"):
            for word in sentence:
                if word not in ['<START>', '<END>']:
                    word_counts[word] += 1

        # Get top vocab_limit words
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        valid_vocab = set(word for word, _ in sorted_words[:vocab_limit])
        valid_vocab.update(['<START>', '<END>'])  # Always include special tokens

    ngram_counts = defaultdict(lambda: defaultdict(int))

    for sentence in tqdm(sentences, desc=f"Building {n}-gram counts", unit="sentences"):
        for i in range(len(sentence) - n + 1):
            context = tuple(sentence[i:i + n - 1])
            next_word = sentence[i + n - 1]

            # Skip if using vocab limit and any word not in valid vocab
            if valid_vocab:
                context_words = list(context) + [next_word]
                if not all(w in valid_vocab for w in context_words):
                    continue

            ngram_counts[context][next_word] += 1

    return dict(ngram_counts)


def ngram_counts_to_probs(ngram_counts: Dict) -> Dict:
    """Convert n-gram counts to probability distributions"""
    ngram_probs = {}

    for context, word_counts in tqdm(ngram_counts.items(), desc="Converting to probabilities", unit="contexts"):
        total = sum(word_counts.values())
        if total > 0:
            ngram_probs[context] = {word: count / total for word, count in word_counts.items()}

    return ngram_probs


def build_ngram_model_from_corpus(tokenized_sentences: List[List[str]], n: int,
                                  vocab_limit: Optional[int] = None) -> Dict:
    """
    Build an n-gram model from tokenized sentences.
    
    Args:
        tokenized_sentences: List of lists, where each inner list is tokens from a sentence
        n: The n in n-gram (2 for bigram, 3 for trigram, etc.)
        vocab_limit: If set, only use the top N most frequent words
        
    Returns:
        N-gram model as nested dictionary
    """
    # Prepare sentences
    prepared = prepare_for_ngrams(tokenized_sentences, n)

    # Build counts
    ngram_counts = build_ngram_counts(prepared, n, vocab_limit)

    # Convert to probabilities
    ngram_probs = ngram_counts_to_probs(ngram_counts)

    # For bigram models, convert from tuple keys to nested dict format
    if n == 2:
        bigram_model = defaultdict(dict)
        for (context_word,), next_word_probs in ngram_probs.items():
            bigram_model[context_word] = next_word_probs
        return dict(bigram_model)

    return ngram_probs


def load_pretrained_unigram() -> Dict[str, float]:
    """
    Load a pre-trained unigram model trained on clean English text.

    Returns:
        Dictionary mapping words to probabilities
    """
    # Load ALL words from Norvig's data
    # Filter out inappropriate words
    return _load_tril_model(max_items=None, filter_bad_words=True)


def build_better_ngram_model(tokenized_sentences: List[List[str]], n: int,
                             vocab_limit: Optional[int] = None,
                             min_count: int = 2) -> Dict:
    """
    Build a better n-gram model with filtering for rare n-grams.
    
    Args:
        tokenized_sentences: List of tokenized sentences
        n: The n in n-gram
        vocab_limit: Vocabulary size limit (if None, uses defaults)
        min_count: Minimum count for an n-gram to be included
        
    Returns:
        N-gram model
    """
    if vocab_limit is None:
        vocab_limit = {
            2: BIGRAM_VOCAB_SIZE,
            3: TRIGRAM_VOCAB_SIZE,
            4: FOURGRAM_VOCAB_SIZE
        }.get(n, 500)

    # Prepare sentences
    prepared = prepare_for_ngrams(tokenized_sentences, n)

    # Build counts
    ngram_counts = build_ngram_counts(prepared, n, vocab_limit)

    # Filter out rare n-grams
    filtered_counts = {}
    for context, word_counts in ngram_counts.items():
        # Filter out rare transitions
        filtered_words = {word: count for word, count in word_counts.items()
                          if count >= min_count or word == '<END>'}  # Always keep END
        if filtered_words:
            filtered_counts[context] = filtered_words

    # Convert to probabilities
    ngram_probs = ngram_counts_to_probs(filtered_counts)

    # For bigram models, convert format
    if n == 2:
        bigram_model = defaultdict(dict)
        for (context_word,), next_word_probs in ngram_probs.items():
            bigram_model[context_word] = next_word_probs
        return dict(bigram_model)

    return ngram_probs


def load_pretrained_ngram(n: int, use_better_model: bool = False, verbose: bool = False) -> Dict:
    """
    Load a pre-trained n-gram model.
    
    Args:
        n: The n in n-gram (2 for bigram, 3 for trigram, etc.)
        use_better_model: If True, use filtered model for better generation
        verbose: If True, print loading progress (default: False)
        
    Returns:
        N-gram model (format depends on n)
    """
    if n == 1:
        return load_pretrained_unigram()
    elif n == 2:
        return load_pretrained_bigram()

    # For n > 2, build from Shakespeare corpus
    from .datasets import shake_words

    # Use vocabulary limits to control model size
    vocab_limit = 500  # Smaller vocab for higher n

    if not verbose:
        # Suppress all output including tqdm progress bars
        import sys
        import os
        old_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        old_stderr = sys.stderr
        sys.stderr = open(os.devnull, 'w')
        try:
            if use_better_model and n >= 3:
                min_count = 2 if n == 3 else 3
                model = build_better_ngram_model(shake_words, n, vocab_limit, min_count)
            else:
                model = build_ngram_model_from_corpus(shake_words, n, vocab_limit)
        finally:
            sys.stdout.close()
            sys.stdout = old_stdout
            sys.stderr.close()
            sys.stderr = old_stderr
    else:
        print(f"Loading Shakespeare corpus for {n}-gram model...")
        if use_better_model and n >= 3:
            min_count = 2 if n == 3 else 3
            model = build_better_ngram_model(shake_words, n, vocab_limit, min_count)
        else:
            model = build_ngram_model_from_corpus(shake_words, n, vocab_limit)

        # Print statistics
        total_ngrams = sum(len(v) for v in model.values())
        print(f"Loaded {n}-gram model: {len(model)} contexts, {total_ngrams} total transitions")

    return model


# Lazy-loaded Norvig models
def get_tril_model():
    """Load pre-saved Norvig unigram model from JSON (lazy-loaded)"""
    global _tril_model_cache
    if _tril_model_cache is None:
        json_path = _get_data_dir() / "norvig_unigram_full.json"
        if json_path.exists():
            print(f"Loading Norvig unigrams from {json_path}...")
            with open(json_path, 'r') as f:
                _tril_model_cache = json.load(f)
            print(f"Loaded {len(_tril_model_cache):,} words")
        else:
            print(f"JSON file not found at {json_path}. Loading from source...")
            _tril_model_cache = load_pretrained_unigram()
    return _tril_model_cache


def gpt2(prefix, max_length=20):
    import torch
    from transformers import pipeline
    from better_profanity import profanity
    import logging
    # Suppress transformer warnings
    logging.getLogger("transformers").setLevel(logging.ERROR)
    # Set up the model with Apple Silicon optimization
    device = 0 if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else -1)
    generator = pipeline('text-generation', model='distilgpt2', device=device)

    # Rejection sampling - try up to 10 times
    max_attempts = 10

    for attempt in range(max_attempts):
        result = generator(
            prefix,
            max_length=max_length,  # Shorter to avoid repetition
            temperature=0.9,  # Higher for more variety
            do_sample=True,
            top_p=0.95,
            pad_token_id=generator.tokenizer.eos_token_id,
            repetition_penalty=1.2  # Penalize repetition
        )

        generated_text = result[0]['generated_text']

        # Clean up weird characters and excessive newlines
        generated_text = generated_text.replace('‹', '').replace('\n\n\n', '\n\n')

        # Check for quality issues
        lines = generated_text.split('\n')

        # Reject if too many repeated lines
        if len(lines) > len(set(lines)) * 1.5:  # Too much repetition
            continue

        # Reject if contains profanity
        if profanity.contains_profanity(generated_text):
            continue

        # Good enough!
        return generated_text.strip()

    # If all attempts failed, return the prefix with a safe continuation
    return prefix + " [content filtered]"


# Module-level __getattr__ for lazy loading
def __getattr__(name):
    if name == 'tril_model':
        return get_tril_model()
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
