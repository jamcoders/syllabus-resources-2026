import pytest
import spacy


@pytest.fixture
def nlp():
    """Load spacy model once for all tests"""
    return spacy.load('en_core_web_sm')


def tokenize_text(text, nlp):
    """Tokenize text using the same scheme as _tokenize_moby"""
    doc = nlp(text)
    tokens = [token.text.lower() for token in doc if not token.is_punct and not token.is_space]
    return tokens


def test_basic_tokenization(nlp):
    """Test basic tokenization removes punctuation and lowercases"""
    text = "Hello, World!"
    tokens = tokenize_text(text, nlp)
    assert tokens == ['hello', 'world']


def test_multiple_punctuation(nlp):
    """Test handling of multiple punctuation marks"""
    text = "Wait... What?!?"
    tokens = tokenize_text(text, nlp)
    assert tokens == ['wait', 'what']


def test_apostrophes(nlp):
    """Test handling of contractions with apostrophes"""
    text = "It's a beautiful day, isn't it?"
    tokens = tokenize_text(text, nlp)
    assert tokens == ['it', "'s", 'a', 'beautiful', 'day', 'is', "n't", 'it']


def test_numbers_preserved(nlp):
    """Test that numbers are preserved"""
    text = "I have 42 apples and 3.14 pies."
    tokens = tokenize_text(text, nlp)
    assert tokens == ['i', 'have', '42', 'apples', 'and', '3.14', 'pies']


def test_newlines_removed(nlp):
    """Test that newlines and whitespace are removed"""
    text = "First line\nSecond line\r\nThird line"
    tokens = tokenize_text(text, nlp)
    assert tokens == ['first', 'line', 'second', 'line', 'third', 'line']


def test_empty_string(nlp):
    """Test tokenization of empty string"""
    text = ""
    tokens = tokenize_text(text, nlp)
    assert tokens == []


def test_only_punctuation(nlp):
    """Test string with only punctuation"""
    text = "...!!!"
    tokens = tokenize_text(text, nlp)
    assert tokens == []


def test_mixed_case(nlp):
    """Test mixed case handling"""
    text = "The QUICK Brown FOX"
    tokens = tokenize_text(text, nlp)
    assert tokens == ['the', 'quick', 'brown', 'fox']


def test_special_characters(nlp):
    """Test handling of special characters"""
    text = "Email: test@example.com (contact us!)"
    tokens = tokenize_text(text, nlp)
    assert tokens == ['email', 'test@example.com', 'contact', 'us']


def test_quotes(nlp):
    """Test handling of quotes"""
    text = '"Hello," she said.'
    tokens = tokenize_text(text, nlp)
    assert tokens == ['hello', 'she', 'said']