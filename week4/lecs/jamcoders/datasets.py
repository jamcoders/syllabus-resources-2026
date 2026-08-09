# Pre-loaded datasets for JamCoders students
import json
import os
import urllib.request
import re


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




def _fetch_shakespeare():
    """Fetch and process Shakespeare text from Norvig"""
    print("Fetching Shakespeare from Norvig...")
    url = "https://norvig.com/ngrams/shakespeare.txt"
    with urllib.request.urlopen(url) as response:
        full_text = response.read().decode('utf-8')

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
    from datasets import load_dataset  # From HuggingFace datasets library
    print("Loading JamPatoisNLI dataset from HuggingFace...")
    ds = load_dataset('Ruth-Ann/jampatoisnli')
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
            "fetched_date": "2025-07-21"
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
            "fetched_date": "2025-07-20"
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
    # Load spacy model with optimizations for speed
    import spacy
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])  # We only need tokenization

    tokenized_sentences = []
    # Process in batches for better performance
    batch_size = 1000

    from tqdm import tqdm
    for i in tqdm(range(0, len(sentences), batch_size), desc=f"{description} batches"):
        batch = sentences[i:i + batch_size]
        # Process batch with pipe for efficiency
        for doc in nlp.pipe(batch, batch_size=50):
            tokens = [token.text.lower() for token in doc if not token.is_punct and not token.is_space]
            tokenized_sentences.append(tokens)

    return tokenized_sentences


def _create_shakespeare_words_json():
    """Create shakespeare_words.json if it doesn't exist"""
    json_path = _get_json_path('shakespeare_words.json')
    if not os.path.exists(json_path):
        print("Tokenizing Shakespeare (this will be cached for future use)...")

        # Ensure shakespeare is loaded
        global _shake_sentences_cache
        if _shake_sentences_cache is None:
            _shake_sentences_cache = _load_json_data('shakespeare.json', create_func=_create_shakespeare_json)

        # Use shared tokenization function
        tokenized_sentences = _tokenize_sentences(_shake_sentences_cache, "Tokenizing Shakespeare")

        # Save to JSON
        _save_json_data('shakespeare_words.json', tokenized_sentences)

        print(f"Saved tokenized data to {json_path}")


# Lazy-load the datasets using __getattr__
_moby_cache = None
_patois_cache = None
_shake_sentences_cache = None
_moby_tokenized_cache = None
_shake_words_cache = None


def __getattr__(name):
    """Lazy loading of datasets"""
    global _moby_cache, _patois_cache, _shake_sentences_cache, _moby_tokenized_cache, _shake_words_cache

    if name == 'moby':
        if _moby_cache is None:
            _moby_cache = _load_json_data('moby.json')
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
        if _shake_words_cache is None:
            _shake_words_cache = _load_json_data('shakespeare_words.json', data_key=None, create_func=_create_shakespeare_words_json)
        # Flatten the list of lists into a single list
        return [word for sentence in _shake_words_cache for word in sentence]
    elif name == 'moby_tokenized':
        if _moby_tokenized_cache is None:
            _moby_tokenized_cache = _load_json_data('moby_tokenized.json', data_key=None)
        return _moby_tokenized_cache
    elif name == 'shake_words':
        if _shake_words_cache is None:
            _shake_words_cache = _load_json_data('shakespeare_words.json', data_key=None, create_func=_create_shakespeare_words_json)
        return _shake_words_cache
    else:
        raise AttributeError(f"module {__name__} has no attribute {name}")