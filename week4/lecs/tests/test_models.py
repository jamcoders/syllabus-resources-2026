import pytest
from jamcoders.models import (
    load_pretrained_unigram,
    load_pretrained_ngram,
    visualize_model,
    prepare_for_ngrams,
    build_ngram_counts,
    ngram_counts_to_probs,
    build_ngram_model_from_corpus,
    generate_from_ngram_model,
)
from unittest.mock import patch, MagicMock
import matplotlib.pyplot as plt


class TestNgramPreparation:
    def test_prepare_for_bigrams(self):
        """Test preparing sentences for bigram modeling"""
        sentences = [['hello', 'world'], ['good', 'morning']]
        prepared = prepare_for_ngrams(sentences, n=2)
        
        assert len(prepared) == 2
        assert prepared[0] == ['<START>', 'hello', 'world', '<END>']
        assert prepared[1] == ['<START>', 'good', 'morning', '<END>']
    
    def test_prepare_for_trigrams(self):
        """Test preparing sentences for trigram modeling"""
        sentences = [['hello', 'world']]
        prepared = prepare_for_ngrams(sentences, n=3)
        
        assert prepared[0] == ['<START>', '<START>', 'hello', 'world', '<END>']
    
    def test_prepare_empty_sentence(self):
        """Test preparing empty sentences"""
        sentences = [[]]
        prepared = prepare_for_ngrams(sentences, n=2)
        
        assert prepared[0] == ['<START>', '<END>']


class TestNgramCounting:
    def test_build_bigram_counts(self):
        """Test building bigram counts"""
        prepared = [['<START>', 'hello', 'world', '<END>'], 
                   ['<START>', 'hello', 'there', '<END>']]
        counts = build_ngram_counts(prepared, n=2)
        
        assert counts[('<START>',)]['hello'] == 2
        assert counts[('hello',)]['world'] == 1
        assert counts[('hello',)]['there'] == 1
        assert counts[('world',)]['<END>'] == 1
    
    def test_build_trigram_counts(self):
        """Test building trigram counts"""
        prepared = [['<START>', '<START>', 'the', 'cat', 'sat', '<END>']]
        counts = build_ngram_counts(prepared, n=3)
        
        assert counts[('<START>', '<START>')]['the'] == 1
        assert counts[('<START>', 'the')]['cat'] == 1
        assert counts[('the', 'cat')]['sat'] == 1
    
    def test_vocab_limit(self):
        """Test vocabulary limiting in n-gram counting"""
        prepared = [['<START>', 'common', 'rare', 'common', '<END>'],
                   ['<START>', 'common', 'common', '<END>']]
        
        # With vocab_limit=1, only 'common' should be kept
        counts = build_ngram_counts(prepared, n=2, vocab_limit=1)
        
        # Should have transitions involving 'common' and special tokens
        assert ('<START>',) in counts
        assert ('common',) in counts
        # 'rare' should not appear as a context
        assert ('rare',) not in counts


class TestNgramProbabilities:
    def test_counts_to_probs(self):
        """Test converting counts to probabilities"""
        counts = {
            ('the',): {'cat': 2, 'dog': 1},
            ('a',): {'cat': 1}
        }
        probs = ngram_counts_to_probs(counts)
        
        assert abs(probs[('the',)]['cat'] - 2/3) < 0.001
        assert abs(probs[('the',)]['dog'] - 1/3) < 0.001
        assert probs[('a',)]['cat'] == 1.0


class TestCorpusModeling:
    def test_build_bigram_model(self):
        """Test building complete bigram model from corpus"""
        sentences = [['the', 'cat'], ['the', 'dog'], ['a', 'cat']]
        model = build_ngram_model_from_corpus(sentences, n=2)
        
        # Check it's in the right format for bigrams
        assert isinstance(model, dict)
        assert '<START>' in model
        assert 'the' in model
        
        # Check probabilities
        assert abs(model['<START>']['the'] - 2/3) < 0.001
        assert abs(model['<START>']['a'] - 1/3) < 0.001
        assert abs(model['the']['cat'] - 0.5) < 0.001
        assert abs(model['the']['dog'] - 0.5) < 0.001
    
    def test_build_trigram_model(self):
        """Test building trigram model from corpus"""
        sentences = [['the', 'cat', 'sat'], ['the', 'cat', 'ran']]
        model = build_ngram_model_from_corpus(sentences, n=3)
        
        # Check format - should use tuple keys
        assert ('<START>', '<START>') in model
        assert ('the', 'cat') in model
        
        # Check probabilities
        assert model[('the', 'cat')]['sat'] == 0.5
        assert model[('the', 'cat')]['ran'] == 0.5


class TestTextGeneration:
    
    def test_generate_from_ngram_trigram(self):
        """Test generating from trigram model"""
        model = {
            ('<START>', '<START>'): {'the': 0.5, 'a': 0.5},
            ('<START>', 'the'): {'cat': 0.7, 'dog': 0.3},
            ('the', 'cat'): {'sat': 0.6, 'ran': 0.4},
            ('cat', 'sat'): {'<END>': 1.0},
            ('the', 'dog'): {'ran': 1.0},
            ('dog', 'ran'): {'<END>': 1.0}
        }
        
        text = generate_from_ngram_model(model, "", 10)
        assert isinstance(text, str)
        # Should generate something like "the cat sat" or "the dog ran"


class TestNgramFunctions:
    def test_ngram_train_generate(self):
        """Test n-gram model training and generation"""
        sentences = [['the', 'cat', 'sat'], ['the', 'dog', 'ran'], ['a', 'cat', 'ran']]
        
        # Test bigram
        bigram_model = build_ngram_model_from_corpus(sentences, n=2)
        generated_text = generate_from_ngram_model(bigram_model, "", 10)
        generated_words = generated_text.split()
        assert isinstance(generated_words, list)
        assert len(generated_words) <= 10
        
        # Test trigram
        trigram_model = build_ngram_model_from_corpus(sentences, n=3)
        generated_text = generate_from_ngram_model(trigram_model, "", 10)
        generated_words = generated_text.split()
        assert isinstance(generated_words, list)
    
    def test_ngram_probabilities(self):
        """Test getting probabilities from n-gram models"""
        sentences = [['the', 'cat'], ['the', 'dog']]
        
        # Build bigram model
        model = build_ngram_model_from_corpus(sentences, n=2)
        
        # Get probabilities for context 'the'
        probs = model.get('the', {})
        assert 'cat' in probs
        assert 'dog' in probs
        assert abs(probs['cat'] - 0.5) < 0.001
        assert abs(probs['dog'] - 0.5) < 0.001
    
    def test_ngram_direct_access(self):
        """Test direct dictionary access to n-gram models"""
        sentences = [['the', 'cat', 'sat'], ['the', 'dog', 'ran']]
        
        # Build models
        bigram_model = build_ngram_model_from_corpus(sentences, n=2)
        trigram_model = build_ngram_model_from_corpus(sentences, n=3)
        
        # Test bigram access
        assert 'the' in bigram_model
        assert 'cat' in bigram_model['the']
        
        # Test trigram access
        assert ('the', 'cat') in trigram_model
        assert 'sat' in trigram_model[('the', 'cat')]


class TestPretrainedModels:
    @patch('jamcoders.models._download_norvig_data')
    @patch('builtins.open')
    def test_load_pretrained_unigram(self, mock_open, mock_download):
        """Test loading pretrained unigram model"""
        # Mock the download
        mock_download.return_value = MagicMock()
        
        # Mock file content
        mock_file = MagicMock()
        mock_file.__enter__.return_value = iter([
            'the\t1000\n',
            'of\t500\n',
            'and\t400\n'
        ])
        mock_open.return_value = mock_file
        
        model = load_pretrained_unigram()
        
        assert 'the' in model
        assert 'of' in model
        assert 'and' in model
        # Check probabilities sum to 1
        assert abs(sum(model.values()) - 1.0) < 0.001
    
    @patch('jamcoders.models.build_ngram_model_from_corpus')
    def test_load_pretrained_bigram(self, mock_build):
        """Test loading pretrained bigram model"""
        # Mock the model building
        mock_model = {
            '<START>': {'the': 0.5, 'a': 0.5},
            'the': {'cat': 0.7, 'dog': 0.3}
        }
        mock_build.return_value = mock_model
        
        with patch('builtins.print'):  # Suppress print output
            model = load_pretrained_ngram(2)
        
        assert model == mock_model


class TestVisualization:
    @patch('matplotlib.pyplot.show')
    def test_visualize_model(self, mock_show):
        """Test visualization function"""
        # Test with counts
        word_counts = {'the': 100, 'of': 80, 'and': 70, 'to': 60, 'a': 50}
        visualize_model(word_counts, top_n=3)
        
        # Check that show was called
        mock_show.assert_called_once()
        
        # Test with probabilities
        word_probs = {'the': 0.1, 'of': 0.08, 'and': 0.07}
        visualize_model(word_probs, top_n=2)


class TestEdgeCases:
    def test_empty_corpus(self):
        """Test building model from empty corpus"""
        model = build_ngram_model_from_corpus([], n=2)
        assert model == {}
    
    def test_single_word_sentences(self):
        """Test with single-word sentences"""
        sentences = [['hello'], ['world']]
        model = build_ngram_model_from_corpus(sentences, n=2)
        
        assert '<START>' in model
        assert model['<START>']['hello'] == 0.5
        assert model['<START>']['world'] == 0.5
        assert model['hello']['<END>'] == 1.0
        assert model['world']['<END>'] == 1.0
    
    def test_repeated_words(self):
        """Test with repeated words in sentences"""
        sentences = [['the', 'the', 'the']]
        model = build_ngram_model_from_corpus(sentences, n=2)
        
        assert model['the']['the'] == 2/3  # 2 out of 3 times 'the' follows 'the'
        assert model['the']['<END>'] == 1/3  # 1 out of 3 times '<END>' follows 'the'