import re
from collections import Counter
from math import log
from typing import Dict, List, Optional
import numpy as np
from functools import lru_cache
from tweetnlp import Tokenizer, NER
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from multiprocessing import Pool, cpu_count
import spacy
from tqdm import tqdm
import scipy.sparse as sp

# --- Configurações globais ---
EMOTICON_PATTERN = r'[:;=8][\-^]?[)DpP(]'
URL_PATTERN = r'http\S+|www\S+|https\S+'
MENTION_PATTERN = r'@\w+'
HASHTAG_PATTERN = r'#\w+'

# --- Inicialização lazy de modelos ---
_tokenizer = None
_ner_model = None

def get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = Tokenizer()
    return _tokenizer

def get_ner_model():
    global _ner_model
    if _ner_model is None:
        _ner_model = NER()
    return _ner_model

@lru_cache(maxsize=5000)
def get_cached_ner(text):
    return get_ner_model().predict(text)

# --- Funções de tokenização ---
def preprocess_text(text: str) -> List[str]:
    """Tokeniza o texto usando Twokenizer e normaliza para lowercase"""
    tokenizer = get_tokenizer()
    tokens = tokenizer.tokenize(text)
    return [token.lower() for token in tokens if token.strip()]

def preprocess_text_A(text: str) -> List[str]:
    """Tokeniza o texto usando Twokenizer"""
    tokenizer = get_tokenizer()
    tokens = tokenizer.tokenize(text)
    return [token for token in tokens if token.strip()]

def batch_preprocess_texts(texts: List[str]) -> List[List[str]]:
    tokenizer = get_tokenizer()
    all_tokens = []
    for text in texts:
        tokens = tokenizer.tokenize(text)
        all_tokens.append([token.lower() for token in tokens if token.strip()])
    return all_tokens

# --- Módulo 1: Features Lexicais ---
def extract_lexical_features(text: str, words: List[str]) -> Dict:
    word_count = len(words)
    unique_words = set(words)
    char_count = sum(len(word) for word in words)
    
    return {
        'lexical_type_token_ratio': len(unique_words) / max(1, word_count),
        'lexical_word_count': word_count,
        'lexical_unique_words': len(unique_words),
        'lexical_avg_word_length': char_count / max(1, word_count),
        'lexical_word_length_variance': np.var([len(word) for word in words]) if word_count > 1 else 0,
        'lexical_stopword_ratio': sum(1 for word in words if word in STOPWORDS) / max(1, word_count),
    }

# --- Definindo stopwords básicas para tweets (pode ser expandido) ---
STOPWORDS = set([
    "a", "an", "the", "and", "or", "but", "if", "in", "on", "for", "to", "of", 
    "at", "by", "with", "is", "are", "was", "were", "be", "been", "being", "i",
    "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them"
])

# Módulo 2: Features Sintáticas
def extract_syntactic_features(words: List[str]) -> Dict:
    """Extrai features sintáticas de uma lista de tokens usando spaCy"""
    word_count = len(words)

    # Criar doc spaCy a partir dos tokens
    nlp = spacy.blank("en")
    doc = spacy.tokens.Doc(nlp.vocab, words=words)
    for name, proc in nlp.pipeline:
        doc = proc(doc)

    # POS tags e bigramas
    pos_tags = [token.pos_ for token in doc]
    pos_bigrams = list(zip(pos_tags[:-1], pos_tags[1:]))

    # Comprimento das sentenças
    sentences = list(doc.sents)
    sentence_lengths = [
        len([t for t in sent if not t.is_punct and not t.is_space])
        for sent in sentences
    ]

    def entropy(counter, total):
        return -sum((count / total) * log(count / total)
                    for count in counter.values()) if total else 0

    return {
        'syntactic_pos_tag_entropy': entropy(Counter(pos_tags), len(pos_tags)),
        'syntactic_pos_bigram_entropy': entropy(Counter(pos_bigrams), len(pos_bigrams)),
        'syntactic_avg_sentence_length': np.mean(sentence_lengths) if sentence_lengths else 0,
        'syntactic_subordinating_conj': sum(1 for token in doc if token.dep_ == 'mark') / max(1, len(sentences)),
        'syntactic_comma_ratio': sum(1 for token in doc if token.text == ',') / max(1, len(sentences)),
        'syntactic_punct_ratio': sum(1 for token in doc if token.is_punct) / max(1, word_count),
    }




# --- Módulo 3: Features Estilísticas ---
def extract_stylistic_features(text: str, word_count: int) -> Dict:
    return {
        'stylistic_repeated_chars': int(bool(re.search(r'(.)\1{2,}', text))),
        'stylistic_repeated_words': int(bool(re.search(r'\b(\w+)\s+\1\b', text.lower()))),
        'stylistic_exclamation_density': text.count('!') / max(1, word_count),
        'stylistic_question_density': text.count('?') / max(1, word_count),
        'stylistic_ellipsis_count': text.count('...'),
        'stylistic_emoji_density': len(re.findall(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF]', text)) / max(1, word_count),
        'stylistic_emoticon_density': len(re.findall(EMOTICON_PATTERN, text)) / max(1, word_count),
    }
def extract_stylistic_featuresA(text: str, word_count: int) -> Dict:
    return {
        'stylistic_random_uppercase': int(bool(re.search(r'\b[a-zA-Z]*[A-Z][a-zA-Z]*[A-Z][a-zA-Z]*\b', text))),
        'stylistic_capitalization_inconsistency': sum(
            1 for word in text.split() if not word.isupper() and not word.islower() and not word.istitle()
        ) / max(1, word_count),
    }


# --- Módulo 4: Features Estruturais ---
def extract_structural_features(text: str, words: List[str], word_count: int) -> Dict:
    return {
        'structural_has_url': int(bool(re.search(URL_PATTERN, text))),
        'structural_has_mention': int(bool(re.search(MENTION_PATTERN, text))),
        'structural_has_hashtag': int(bool(re.search(HASHTAG_PATTERN, text))),
        'structural_is_retweet': int(text.strip().startswith('RT')),
        'structural_url_density': len(re.findall(URL_PATTERN, text)) / max(1, word_count),
        'structural_mention_density': len(re.findall(MENTION_PATTERN, text)) / max(1, word_count),
        'structural_hashtag_density': len(re.findall(HASHTAG_PATTERN, text)) / max(1, word_count),
        'structural_extra_spaces': len(re.findall(r'\s{2,}', text)) / max(1, word_count),
        'structural_temporal_markers': len([word for word in words if word.lower() in {'today', 'yesterday', 'tomorrow', 'now', 'later'}]) / max(1, word_count),
    }

# --- Módulo 5: Features de NLP (NER) ---
def extract_tweetnlp_features(text: str, word_count: int) -> Dict:
    features = {}
    try:
        entities = get_cached_ner(text)
        entity_types = [e['type'] for e in entities]
        positions = [e['offset'][0] / len(text) for e in entities] if entities else []

        features.update({
            'ner_count': len(entities),
            'ner_ratio': len(entities) / max(1, word_count),
            'ner_type_diversity': len(set(entity_types)) / max(1, len(entity_types)) if entity_types else 0,
            'ner_position_mean': np.mean(positions) if positions else 0,
            'ner_position_std': np.std(positions) if len(positions) > 1 else 0,
            **{f'ner_{type}_count': count for type, count in Counter(entity_types).items()}
        })
    except Exception as e:
        pass
    return features

# --- Módulo 8: Extração de N-grams ---
def extract_ngrams_features(texts: List[str], max_features: int = 300) -> pd.DataFrame:
    processed_texts = [" ".join(tokens) for tokens in batch_preprocess_texts(texts)]
    vectorizer = CountVectorizer(ngram_range=(1, 3), max_features=max_features)
    X_ngrams = vectorizer.fit_transform(processed_texts)
    return pd.DataFrame.sparse.from_spmatrix(X_ngrams, columns=vectorizer.get_feature_names_out())

