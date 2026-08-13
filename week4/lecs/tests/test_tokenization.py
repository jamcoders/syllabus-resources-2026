from jamcoders.datasets import tokenize


def test_basic_tokenization():
    """Test basic tokenization removes punctuation and lowercases"""
    assert tokenize("Hello, World!") == ['hello', 'world']


def test_multiple_punctuation():
    """Test handling of multiple punctuation marks"""
    assert tokenize("Wait... What?!?") == ['wait', 'what']


def test_contractions_stay_whole():
    """Contractions and possessives stay single words -- "n't" is not a word"""
    assert tokenize("It's a beautiful day, isn't it?") == [
        "it's", 'a', 'beautiful', 'day', "isn't", 'it']
    assert tokenize("Hamlet's father") == ["hamlet's", 'father']


def test_early_modern_apostrophes():
    """Shakespeare's elisions survive tokenization"""
    assert tokenize("O'er the lov'd hills") == ["o'er", 'the', "lov'd", 'hills']


def test_curly_apostrophe():
    """Typographic apostrophes (from Project Gutenberg) behave like plain ones"""
    assert tokenize("It’s here") == ["it’s", 'here']


def test_numbers_preserved():
    """Test that numbers, including decimals, are preserved"""
    assert tokenize("I have 42 apples and 3.14 pies.") == [
        'i', 'have', '42', 'apples', 'and', '3.14', 'pies']


def test_newlines_removed():
    """Test that newlines and whitespace are removed"""
    assert tokenize("First line\nSecond line\r\nThird line") == [
        'first', 'line', 'second', 'line', 'third', 'line']


def test_empty_string():
    """Test tokenization of empty string"""
    assert tokenize("") == []


def test_only_punctuation():
    """Test string with only punctuation"""
    assert tokenize("...!!!") == []


def test_mixed_case():
    """Test mixed case handling"""
    assert tokenize("The QUICK Brown FOX") == ['the', 'quick', 'brown', 'fox']


def test_special_characters():
    """Punctuation inside an email splits it -- fine for word-level models"""
    assert tokenize("Email: test@example.com (contact us!)") == [
        'email', 'test', 'example', 'com', 'contact', 'us']


def test_quotes():
    """Test handling of quotes"""
    assert tokenize('"Hello," she said.') == ['hello', 'she', 'said']


def test_hyphenated_words_split():
    """Hyphens separate words, so "sperm-whale" becomes two tokens"""
    assert tokenize("the sperm-whale") == ['the', 'sperm', 'whale']
