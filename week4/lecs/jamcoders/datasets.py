# Pre-loaded datasets for JamCoders students
import json
import os
import urllib.request
import re
from datetime import date


def _get_json_path(filename):
    """Get the path to a JSON file in the datasets module directory"""
    return os.path.join(os.path.dirname(__file__), filename)


def _load_json_data(filename, data_key='sentences', create_func=None):
    """Generic function to load data from a JSON file, creating it if needed"""
    if create_func:
        create_func()
    json_path = _get_json_path(filename)
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data[data_key] if data_key else data


def _save_json_data(filename, data):
    """Generic function to save data to a JSON file"""
    json_path = _get_json_path(filename)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


class DatasetDownloadError(RuntimeError):
    """A dataset could not be downloaded -- usually because there is no internet."""


def _offline_message(what, url, cache_path, reason):
    """Build the message students see when a first-run download fails"""
    return (
        f"\nCould not download {what}.\n"
        f"  Source: {url}\n"
        f"  Reason: {reason}\n\n"
        f"You are probably offline. {what} is downloaded once and then cached in\n"
        f"{cache_path}, so this only affects the very first run.\n\n"
        f"What to do:\n"
        f"  1. Connect to the internet and re-run this cell, or\n"
        f"  2. Ask an instructor to copy {cache_path} onto your machine.\n"
    )


def _download_text(url, what, cache_path):
    """Download text from a URL, turning connection errors into a readable message"""
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            return response.read().decode('utf-8')
    except Exception as e:
        # `from None` hides the urllib traceback -- students only need the advice.
        raise DatasetDownloadError(_offline_message(what, url, cache_path, e)) from None


# Bump this whenever the tokenizer below changes: cached *_words / *_tokenized
# files record the version they were built with and are rebuilt when it differs.
TOKENIZER_VERSION = 2

# A word is a run of letters/digits, keeping internal apostrophes so contractions
# and possessives survive whole ("isn't", "hamlet's", "o'er"). Decimals like
# "3.14" are matched first so they don't split. Everything else separates words.
_WORD_RE = re.compile(r"\d+\.\d+|\w+(?:['’]\w+)*")


def tokenize(text):
    """Split text into lowercase words, dropping punctuation and whitespace.

    >>> tokenize("It's a beautiful day, isn't it?")
    ["it's", 'a', 'beautiful', 'day', "isn't", 'it']
    """
    return _WORD_RE.findall(text.lower())


def _fetch_shakespeare():
    """Fetch and process Shakespeare text from Norvig"""
    print("Fetching Shakespeare from Norvig...")
    url = "https://norvig.com/ngrams/shakespeare.txt"
    full_text = _download_text(url, "the Shakespeare corpus", "jamcoders/shakespeare.json")

    # Split into sentences - Shakespeare uses different formatting
    lines = full_text.split('\n')

    # Clean up and create sentences
    sentences = []
    current_sentence = []

    for line in lines:
        line = line.strip()
        if not line:
            # Empty line - if we have accumulated text, save it as a sentence
            if current_sentence:
                sentences.append(' '.join(current_sentence))
                current_sentence = []
        else:
            # Check if line ends with sentence-ending punctuation
            if line.endswith(('.', '!', '?')):
                current_sentence.append(line)
                sentences.append(' '.join(current_sentence))
                current_sentence = []
            else:
                # Accumulate lines that don't end with punctuation
                current_sentence.append(line)

    # Don't forget the last sentence if there is one
    if current_sentence:
        sentences.append(' '.join(current_sentence))

    # Filter out very short sentences and title-like lines
    clean_sentences = []
    for s in sentences:
        s = s.strip()
        if len(s) > 10 and not s.isupper():  # Skip all-caps titles
            clean_sentences.append(s)

    return clean_sentences


def _fetch_patois():
    """Fetch Patois sentences from HuggingFace"""
    print("Loading JamPatoisNLI dataset from HuggingFace...")
    url = "https://huggingface.co/datasets/Ruth-Ann/jampatoisnli"
    try:
        from datasets import load_dataset  # HuggingFace's `datasets`, not this module
        ds = load_dataset('Ruth-Ann/jampatoisnli')
    except Exception as e:
        raise DatasetDownloadError(
            _offline_message("the JamPatoisNLI dataset", url, "jamcoders/patois.json", e)
        ) from None
    return [example['premise'] for example in ds['train']]


def _create_shakespeare_json():
    """Create shakespeare.json if it doesn't exist"""
    json_path = _get_json_path('shakespeare.json')
    if not os.path.exists(json_path):
        sentences = _fetch_shakespeare()
        metadata = {
            "title": "The Complete Works of William Shakespeare",
            "author": "William Shakespeare",
            "publication_year": "1564-1616",
            "license": "Public domain",
            "source": "Peter Norvig's website (from John DeNero)",
            "source_url": "https://norvig.com/ngrams/shakespeare.txt",
            "description": "Complete works of Shakespeare from Norvig's n-grams collection",
            "fetched_date": date.today().isoformat()
        }
        data = {
            "metadata": metadata,
            "sentences": sentences
        }
        _save_json_data('shakespeare.json', data)


def _create_patois_json():
    """Create patois.json if it doesn't exist"""
    def fetch_with_metadata():
        sentences = _fetch_patois()
        metadata = {
            "title": "Jamaican Patois Natural Language Inference Dataset - Train Premises",
            "dataset_authors": ["Ruth-Ann Armstrong", "John Hewitt", "Christopher Manning"],
            "institution": "Stanford University",
            "year": 2022,
            "license": "Other. See jampatoisnli.github.io",
            "source": f"All {len(sentences)} premise sentences from JamPatoisNLI train split",
            "source_url": "https://huggingface.co/datasets/Ruth-Ann/jampatoisnli/tree/main",
            "description": "Sample sentences in Jamaican Patois",
            "language": "Jamaican Patois (jam)",
            "fetched_date": date.today().isoformat()
        }
        return sentences, metadata
    
    json_path = _get_json_path('patois.json')
    if not os.path.exists(json_path):
        sentences, metadata = fetch_with_metadata()
        data = {
            "metadata": metadata,
            "sentences": sentences
        }
        _save_json_data('patois.json', data)


def _tokenize_sentences(sentences, description="Tokenizing"):
    """Common tokenization logic for any sentence list"""
    from tqdm import tqdm
    return [tokenize(s) for s in tqdm(sentences, desc=description, unit=" sentences")]


def _load_tokenized_json(filename, build):
    """Load tokenized sentences, rebuilding the cache if missing or stale.

    `build` is called only when the cache is absent or was written by an older
    tokenizer, so switching tokenizers never silently serves the old tokens.
    """
    json_path = _get_json_path(filename)
    if os.path.exists(json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, dict) and data.get('tokenizer_version') == TOKENIZER_VERSION:
            return data['sentences']
        print(f"Tokenizer changed since {filename} was written -- rebuilding it...")

    tokenized_sentences = build()
    _save_json_data(filename, {
        'tokenizer_version': TOKENIZER_VERSION,
        'sentences': tokenized_sentences,
    })
    print(f"Saved tokenized data to {json_path}")
    return tokenized_sentences


def _build_shakespeare_words():
    """Tokenize the Shakespeare sentences"""
    global _shake_sentences_cache
    if _shake_sentences_cache is None:
        _shake_sentences_cache = _load_json_data('shakespeare.json', create_func=_create_shakespeare_json)

    return _tokenize_sentences(_shake_sentences_cache, "Tokenizing Shakespeare")


def _create_moby_json():
    """Create moby.json if it doesn't exist by fetching from Project Gutenberg"""
    json_path = _get_json_path('moby.json')
    if not os.path.exists(json_path):
        print("Fetching Moby Dick from Project Gutenberg...")
        url = "https://www.gutenberg.org/files/2701/2701-0.txt"
        full_text = _download_text(url, "Moby Dick", "jamcoders/moby.json")

        # Find the start of the actual book (skip Gutenberg headers)
        start_marker = "Call me Ishmael."
        start_index = full_text.find(start_marker)
        if start_index == -1:
            raise ValueError("Could not find start of book")

        # Find the end of the actual book (skip Gutenberg footer)
        end_marker = "*** END OF THE PROJECT GUTENBERG EBOOK"
        end_index = full_text.find(end_marker)
        if end_index == -1:
            book_text = full_text[start_index:]
        else:
            book_text = full_text[start_index:end_index]

        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', book_text)

        # Clean up sentences
        clean_sentences = []
        for s in sentences:
            s = s.strip()
            if s and not s.startswith('***'):  # Skip Gutenberg markers
                clean_sentences.append(s)

        moby_data = {
            "metadata": {
                "title": "Moby-Dick; or, The Whale",
                "author": "Herman Melville",
                "publication_year": 1851,
                "license": "Public domain (USA); distributed via Project Gutenberg under the Project Gutenberg License. See https://www.gutenberg.org/policy/license.html",
                "source": "Project Gutenberg",
                "source_url": url,
                "description": "Complete text from the classic American novel",
            },
            "sentences": clean_sentences,
        }
        _save_json_data('moby.json', moby_data)


def _build_moby_tokenized():
    """Tokenize the Moby Dick sentences"""
    global _moby_cache
    if _moby_cache is None:
        _moby_cache = _load_json_data('moby.json', create_func=_create_moby_json)

    return _tokenize_sentences(_moby_cache, "Tokenizing Moby Dick")


# Lazy-load the datasets using __getattr__
_moby_cache = None
_patois_cache = None
_shake_sentences_cache = None
_moby_tokenized_cache = None
_shake_words_cache = None
_shake_flat_cache = None


def __getattr__(name):
    """Lazy loading of datasets"""
    global _moby_cache, _patois_cache, _shake_sentences_cache, _moby_tokenized_cache
    global _shake_words_cache, _shake_flat_cache

    if name == 'moby':
        if _moby_cache is None:
            _moby_cache = _load_json_data('moby.json', create_func=_create_moby_json)
        return _moby_cache
    elif name == 'patois':
        if _patois_cache is None:
            _patois_cache = _load_json_data('patois.json', create_func=_create_patois_json)
        return _patois_cache
    elif name == 'shake_sentences':
        if _shake_sentences_cache is None:
            _shake_sentences_cache = _load_json_data('shakespeare.json', create_func=_create_shakespeare_json)
        return _shake_sentences_cache
    elif name == 'shake':
        # Return all words concatenated into one list
        if _shake_flat_cache is None:
            if _shake_words_cache is None:
                _shake_words_cache = _load_tokenized_json('shakespeare_words.json', _build_shakespeare_words)
            # Flatten once, not on every access -- this is ~900k words
            _shake_flat_cache = [word for sentence in _shake_words_cache for word in sentence]
        return _shake_flat_cache
    elif name == 'moby_tokenized':
        if _moby_tokenized_cache is None:
            _moby_tokenized_cache = _load_tokenized_json('moby_tokenized.json', _build_moby_tokenized)
        return _moby_tokenized_cache
    elif name == 'shake_words':
        if _shake_words_cache is None:
            _shake_words_cache = _load_tokenized_json('shakespeare_words.json', _build_shakespeare_words)
        return _shake_words_cache
    else:
        raise AttributeError(f"module {__name__} has no attribute {name}")