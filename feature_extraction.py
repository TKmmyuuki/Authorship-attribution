import re
from collections import Counter
from math import log
from typing import Dict, List, Optional
import numpy as np
from functools import lru_cache
from tweetnlp import NER
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
nlp = spacy.load("en_core_web_sm")

def get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        import nltk
        nltk.download('punkt')
        from nltk.tokenize import TweetTokenizer
        _tokenizer = TweetTokenizer()
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
    
    # Juntar os tokens em texto para processamento adequado
    text = " ".join(words)
    doc = nlp(text)

    def entropy(counter, total):
        if total <= 1:
            return 0
        return -sum((count / total) * log(count / total) for count in counter.values())

    # POS tags e bigramas (ignorar espaços)
    pos_tags = [token.pos_ for token in doc if not token.is_space]
    pos_bigrams = list(zip(pos_tags[:-1], pos_tags[1:])) if len(pos_tags) > 1 else []

    # Comprimento das sentenças
    sentence_lengths = [
        len([t for t in sent if not t.is_punct and not t.is_space])
        for sent in doc.sents
    ]

    # Features que dependem do parser (usar try/except como fallback)
    try:
        subordinating_conj = sum(1 for token in doc if token.dep_ == 'mark') / max(1, len(list(doc.sents)))
    except:
        subordinating_conj = 0  # Fallback se o parser não estiver disponível

    return {
        'syntactic_pos_tag_entropy': entropy(Counter(pos_tags), len(pos_tags)),
        'syntactic_pos_bigram_entropy': entropy(Counter(pos_bigrams), len(pos_bigrams)) if pos_bigrams else 0,
        'syntactic_avg_sentence_length': np.mean(sentence_lengths) if sentence_lengths else 0,
        'syntactic_subordinating_conj': subordinating_conj,
        'syntactic_comma_ratio': sum(1 for token in doc if token.text == ',') / max(1, word_count),
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
def extract_ngrams_features(processed_texts: List[List[str]], max_features: int = 300) -> pd.DataFrame:
    """Extrai features de n-grams a partir de textos já tokenizados"""
    # Juntar os tokens em strings para o CountVectorizer
    text_strings = [" ".join(tokens) for tokens in processed_texts]
    
    vectorizer = CountVectorizer(ngram_range=(1, 3), max_features=max_features)
    X_ngrams = vectorizer.fit_transform(text_strings)
    
    return pd.DataFrame.sparse.from_spmatrix(X_ngrams, columns=vectorizer.get_feature_names_out())

# --- CONFIGURAÇÕES ADICIONAIS ---
COMMON_WORDS = set([
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "i", "it", "for", "not", "on", "with", 
    "he", "as", "you", "do", "at", "this", "but", "his", "by", "from", "they", "we", "say", "her", "she"
])

# --- NOVAS FUNÇÕES DE FEATURES ---

def detect_topic_shifts(words: List[str], window_size: int = 5) -> float:
    """Detecta mudanças abruptas de tópico (simplificado)"""
    if len(words) < window_size * 2:
        return 0.0
    
    topic_shifts = 0
    for i in range(len(words) - window_size):
        window1 = set(words[i:i+window_size])
        window2 = set(words[i+window_size:i+window_size*2])
        similarity = len(window1.intersection(window2)) / len(window1.union(window2))
        if similarity < 0.2:  # Pouca sobreposição
            topic_shifts += 1
    
    return topic_shifts / max(1, len(words) - window_size * 2)

def count_hedging_words(words: List[str]) -> int:
    """Conta palavras que indicam hesitação ou incerteza"""
    hedging_words = {
        'maybe', 'perhaps', 'possibly', 'probably', 'likely', 'unlikely',
        'might', 'could', 'would', 'should', 'seem', 'appear', 'suggest',
        'potentially', 'presumably', 'arguably', 'generally', 'usually'
    }
    return sum(1 for word in words if word.lower() in hedging_words)

def calculate_syntactic_depth(doc) -> float:
    """Calcula profundidade sintática média (simplificado)"""
    if not hasattr(doc[0], 'dep_'):  # Verifica se tem parser
        return 0.0
    
    depths = []
    for sent in doc.sents:
        for token in sent:
            if token.dep_ in ('nsubj', 'dobj', 'prep'):
                depth = 0
                current = token
                while current.head != current:
                    depth += 1
                    current = current.head
                depths.append(depth)
    
    return np.mean(depths) if depths else 0.0

import re
import numpy as np
from typing import Dict, List

# --- REGEX COMPILADOS (helpers reutilizáveis) ---
RE_STRUCTURED = re.compile(r'\d+\.\s|-\s|\*\s|\n\d+\.\s|\n\*\s|\n-\s')
RE_POLITE = re.compile(r'\b(please|thank you|kindly|appreciate|wonderful|great|excellent|'
                       r'i apologize|my apologies|grateful)\b', re.IGNORECASE)
RE_DISCLAIMER = re.compile(r'\b(disclaimer|note that|important|warning|caution)\b', re.IGNORECASE)
RE_AI_REF = re.compile(r'\b(ai|artificial intelligence|machine learning|model|algorithm)\b', re.IGNORECASE)

# Específicos ChatGPT
RE_CHATGPT_REF = re.compile(r'\b(as an ai|i\'m|i am|language model|openai|chatgpt|gpt-4|gpt-3)\b', re.IGNORECASE)
RE_ASSISTANT_PATTERNS = re.compile(r'\b(assistant|helpful|happy to help|how can I assist)\b', re.IGNORECASE)

# Específicos Mistral
RE_MISTRAL_REF = re.compile(r'\b(mistral|mixtral|8x7b|moe|mixture[-\s]of[-\s]experts)\b', re.IGNORECASE)
RE_TECH_JARGON = re.compile(r'\b(transformer|embedding|latent|inference|parameter|gradient|'
                            r'optimization|algorithm|architecture|dataset|training|fine.tuning)\b', re.IGNORECASE)
RE_NON_ENGLISH = re.compile(r'\b(le|la|les|un|une|des|él|ella|los|las|y|o|pero|'
                            r'nous|vous|ils|elles|merci|bonjour|au revoir|'
                            r'gracias|por favor|de nada|buenos días)\b', re.IGNORECASE)
RE_STEP_REASONING = re.compile(r'\b(first|second|third|next|then|finally|therefore|thus|'
                               r'step\s*\d+|phase\s*\d+|reasoning|logic)\b', re.IGNORECASE)
RE_ETHICAL = re.compile(r'\b(ethical|moral|responsible|caution|warning|disclaimer|'
                        r'offensive|appropriate|sensitive)\b', re.IGNORECASE)

# --- HELPERS ---
def ratio(count: int, total: int) -> float:
    return count / max(1, total)

def count_matches(pattern: re.Pattern, text: str) -> int:
    return len(pattern.findall(text))

# --- 1. FEATURES ESPECÍFICAS PARA DETECÇÃO DE IA ---
def extract_ai_specific_features(text: str, words: List[str]) -> Dict:
    word_count = len(words)
    unique_words = set(words)

    try:
        topic_shifts = detect_topic_shifts(words)
    except:
        topic_shifts = 0

    hedging_count = count_hedging_words(words) if "count_hedging_words" in globals() else 0

    return {
        'ai_perplexity_score': ratio(len(unique_words), word_count),
        'ai_repeated_ngrams': ratio(sum(1 for i in range(len(words)-2) if words[i:i+3] in words[i+3:]), word_count),
        'ai_topic_shifts': topic_shifts,
        'ai_safety_disclaimers': int(any(kw in text.lower() for kw in ["as an ai", "i cannot", "i shouldn't", "ethical", "responsible"])),
        'ai_hedging_language': ratio(hedging_count, word_count),
    }

# --- 2. FEATURES DE COMPLEXIDADE TEXTUAL ---
def extract_complexity_features(text: str, words: List[str]) -> Dict:
    word_count = len(words)
    try:
        doc = nlp(text)
    except:
        return {}

    try:
        syntactic_depth = calculate_syntactic_depth(doc)
    except:
        syntactic_depth = 0

    sentences = list(doc.sents)
    return {
        'complexity_subordinating_ratio': ratio(sum(1 for t in doc if t.dep_ == 'mark'), len(sentences)),
        'complexity_long_sentences': ratio(sum(1 for s in sentences if len(s) > 15), len(sentences)),
        'complexity_rare_words': ratio(sum(1 for w in words if w not in COMMON_WORDS), word_count),
        'complexity_syntactic_depth': syntactic_depth,
    }

# --- 3. FEATURES TEMPORAIS ---
def extract_temporal_context_features(text: str) -> Dict:
    word_count = len(text.split())
    refs = {
        'recent': len(re.findall(r'\b(today|yesterday|tomorrow|now|recently)\b', text.lower())),
        'past': len(re.findall(r'\b(last|ago|previous|before|earlier)\b', text.lower())),
        'future': len(re.findall(r'\b(next|soon|future|will|going to)\b', text.lower())),
    }
    return {f'temporal_{k}_ratio': ratio(v, word_count) for k, v in refs.items()}

# --- 4. FEATURES GENÉRICAS DE LLM ---
def extract_general_llm_features(text: str, words: List[str]) -> Dict:
    word_count = len(words)
    return {
        'llm_structured_output': ratio(count_matches(RE_STRUCTURED, text), word_count),
        'llm_overly_polite': ratio(count_matches(RE_POLITE, text), word_count),
        'llm_disclaimer_density': ratio(count_matches(RE_DISCLAIMER, text), word_count),
        'llm_ai_references': ratio(count_matches(RE_AI_REF, text), word_count),
    }

# --- 5. ESPECÍFICAS DO CHATGPT ---
def extract_chatgpt_specific_features(text: str, words: List[str]) -> Dict:
    word_count = len(words)
    return {
        'chatgpt_self_ref': ratio(count_matches(RE_CHATGPT_REF, text), word_count),
        'chatgpt_structured_output': ratio(count_matches(RE_STRUCTURED, text), word_count),
        'chatgpt_overly_polite': ratio(count_matches(RE_POLITE, text), word_count),
        'chatgpt_disclaimer_density': ratio(text.lower().count('disclaimer'), word_count),
        'chatgpt_assistant_patterns': ratio(count_matches(RE_ASSISTANT_PATTERNS, text), word_count),
    }

# --- 6. ESPECÍFICAS DO MISTRAL ---
def extract_mistral_specific_features(text: str, words: List[str]) -> Dict:
    word_count = len(words)
    return {
        'mistral_self_ref': ratio(count_matches(RE_MISTRAL_REF, text), word_count),
        'mistral_structured_density': ratio(count_matches(RE_STRUCTURED, text), word_count),
        'mistral_technical_jargon': ratio(count_matches(RE_TECH_JARGON, text), word_count),
        'mistral_non_english_density': ratio(count_matches(RE_NON_ENGLISH, text), word_count),
        'mistral_step_reasoning': ratio(count_matches(RE_STEP_REASONING, text), word_count),
        'mistral_low_ethical_disclaimers': ratio(count_matches(RE_ETHICAL, text), word_count),
    }

# --- 7. PROCESSAMENTO EM LOTE ---
def extract_features_batch(texts: List[str], processed_texts: List[List[str]]) -> List[Dict]:
    results = []
    for text, words in zip(texts, processed_texts):
        features = {}
        features.update(extract_ai_specific_features(text, words))
        features.update(extract_complexity_features(text, words))
        features.update(extract_temporal_context_features(text))
        features.update(extract_general_llm_features(text, words))
        features.update(extract_chatgpt_specific_features(text, words))
        features.update(extract_mistral_specific_features(text, words))
        results.append(features)
    return results

