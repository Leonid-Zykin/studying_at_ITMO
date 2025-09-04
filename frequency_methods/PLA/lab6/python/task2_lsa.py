import os
import re
import json
import csv
import argparse
from typing import List, Tuple, Dict
import inspect
from collections import namedtuple

# Compatibility for libraries expecting inspect.getargspec on Python 3.12+
if not hasattr(inspect, 'getargspec'):
    def _compat_getargspec(func):
        fs = inspect.getfullargspec(func)
        ArgSpec = namedtuple('ArgSpec', 'args varargs keywords defaults')
        return ArgSpec(fs.args, fs.varargs, fs.varkw, fs.defaults)
    inspect.getargspec = _compat_getargspec

import numpy as np
import matplotlib.pyplot as plt
try:
    from wordcloud import WordCloud  # optional
except Exception:
    WordCloud = None

# NLP-lite: no nltk
import pymorphy2

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
IMAGES_DIR = os.path.join(BASE_DIR, 'images', 'task2')
OUTPUT_JSON = os.path.join(IMAGES_DIR, 'lsa_results.json')

# Minimal Russian stopwords fallback (subset) for offline use
RU_STOPWORDS_FALLBACK = {
    'и','в','во','не','что','он','на','я','с','со','как','а','то','все','она','так','его','но','да','ты','к','у','же','вы','за','бы','по','только','ее','мне','было','вот','от','меня','еще','нет','о','из','ему','теперь','когда','даже','ну','вдруг','ли','если','уже','или','ни','быть','был','него','до','вас','нибудь','опять','уж','вам','ведь','там','потом','себя','ничего','ей','может','они','тут','где','есть','надо','ней','для','мы','тебя','их','чем','была','сам','чтоб','без','будто','чего','раз','тоже','себе','под','будет','ж','тогда','кто','этот','того','потому','этого','какой','совсем','ним','здесь','этом','один','почти','мой','тем','чтобы','нее','сейчас','были','куда','зачем','всех','никогда','можно','при','наконец','два','об','её','мной','там','хоть','после','над','больше','тот','через','эти','нас','про','всего','них','какая','много','разве','три','эту','моя','впрочем','хорошо','свою','этой','перед','иногда','лучше','чуть','том','нельзя','такой','им','более','всегда','конечно'
}


def ensure_dirs():
    os.makedirs(IMAGES_DIR, exist_ok=True)


def load_documents(input_path: str) -> List[str]:
    ext = os.path.splitext(input_path)[1].lower()
    if ext == '.json':
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # Expect list of strings or list of dicts with 'text'
        if isinstance(data, list):
            if all(isinstance(x, str) for x in data):
                return data
            if all(isinstance(x, dict) and 'text' in x for x in data):
                return [x['text'] for x in data]
        raise ValueError('Unsupported JSON structure. Use list[str] or list[{text: str}].')
    if ext == '.csv':
        texts = []
        with open(input_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            if 'text' not in reader.fieldnames:
                raise ValueError('CSV must contain a "text" column')
            for row in reader:
                texts.append(row['text'])
        return texts
    # Fallback: plain text, one document per line
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    return lines


def normalize_text(s: str) -> str:
    # Lowercase, remove punctuation/numbers except intra-word hyphens
    s = s.lower()
    s = re.sub(r"[\d]+", " ", s)
    s = re.sub(r"[^a-zа-яё\-\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def preprocess_docs(docs: List[str]) -> Tuple[List[List[str]], List[str]]:
    stop_ru = set(RU_STOPWORDS_FALLBACK)

    # Enforce pymorphy2 lemmatization as per lab requirements
    try:
        morph = pymorphy2.MorphAnalyzer()
    except Exception as e:
        raise RuntimeError('pymorphy2 is required for lemmatization in this lab') from e

    tokenized_docs: List[List[str]] = []
    vocab_set: set = set()

    for doc in docs:
        norm = normalize_text(doc)
        tokens = re.findall(r"[a-zа-яё]+", norm)
        lemmas: List[str] = []
        for t in tokens:
            if t in stop_ru:
                continue
            p = morph.parse(t)[0]
            lemma = p.normal_form
            if lemma in stop_ru:
                continue
            lemmas.append(lemma)
        tokenized_docs.append(lemmas)
        vocab_set.update(lemmas)

    vocab = sorted(vocab_set)
    return tokenized_docs, vocab


def build_term_doc_matrix(tokenized_docs: List[List[str]], vocab: List[str]) -> np.ndarray:
    index: Dict[str, int] = {w: i for i, w in enumerate(vocab)}
    V = len(vocab)
    D = len(tokenized_docs)
    M = np.zeros((V, D), dtype=np.float64)
    for j, doc in enumerate(tokenized_docs):
        for w in doc:
            i = index.get(w)
            if i is not None:
                M[i, j] += 1.0
    # TF-IDF-like scaling (optional but useful)
    # idf = log(D / (1 + df)) where df = number of docs containing term
    df = np.count_nonzero(M > 0, axis=1)
    idf = np.log((D + 1) / (1 + df)) + 1.0
    M = (M.T * idf).T
    return M


def analyze_svd(M: np.ndarray, vocab: List[str], doc_titles: List[str], top_k: int = 2, top_words: int = 5):
    U, S, Vt = np.linalg.svd(M, full_matrices=False)

    results = {
        'singular_values': S.tolist(),
        'topics': []
    }

    for t in range(top_k):
        topic = {}
        u_vec = U[:, t]
        v_vec = Vt[t, :]

        # Words most associated with the topic (by absolute weight)
        top_word_idx = np.argsort(np.abs(u_vec))[::-1][:top_words]
        topic_words = [(vocab[i], float(u_vec[i])) for i in top_word_idx]

        # Documents most associated with the topic
        top_doc_idx = np.argsort(np.abs(v_vec))[::-1][:min(5, len(doc_titles))]
        topic_docs = [(doc_titles[j], float(v_vec[j])) for j in top_doc_idx]

        topic['index'] = t
        topic['top_words'] = topic_words
        topic['top_docs'] = topic_docs
        results['topics'].append(topic)

    return results, U, S, Vt


def save_wordcloud(words_weights: List[Tuple[str, float]], path: str):
    freqs = {w: abs(wt) for w, wt in words_weights}
    if WordCloud is not None:
        wc = WordCloud(width=1200, height=800, background_color='white', collocations=False)
        img = wc.generate_from_frequencies(freqs)
        img.to_file(path)
        return
    # Fallback: simple bar chart if WordCloud not available
    words = list(freqs.keys())
    vals = list(freqs.values())
    plt.figure(figsize=(8, 4), dpi=150)
    plt.bar(words, vals)
    plt.xticks(rotation=30, ha='right')
    plt.title('Word weights')
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def save_global_wordcloud(tokenized_docs: List[List[str]], path: str):
    freqs: Dict[str, float] = {}
    for doc in tokenized_docs:
        for w in doc:
            freqs[w] = freqs.get(w, 0.0) + 1.0
    if WordCloud is not None:
        wc = WordCloud(width=1400, height=900, background_color='white', collocations=False)
        img = wc.generate_from_frequencies(freqs)
        img.to_file(path)
        return
    # Fallback: show top-N words by frequency
    items = sorted(freqs.items(), key=lambda x: x[1], reverse=True)[:30]
    words = [w for w, _ in items]
    vals = [v for _, v in items]
    plt.figure(figsize=(10, 6), dpi=150)
    plt.barh(range(len(words)), vals)
    plt.yticks(range(len(words)), words)
    plt.gca().invert_yaxis()
    plt.title('Global word frequencies (fallback)')
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_top_words(words_weights: List[Tuple[str, float]], path: str, title: str):
    words = [w for w, _ in words_weights]
    vals = [abs(v) for _, v in words_weights]
    plt.figure(figsize=(8, 4), dpi=150)
    plt.bar(words, vals)
    plt.xticks(rotation=30, ha='right')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Path to input dataset: JSON (list[str] or list[{text}]), CSV (with column text), or TXT (one doc per line).')
    parser.add_argument('--encoding', default='utf-8')
    args = parser.parse_args()

    ensure_dirs()

    docs = load_documents(args.input)
    doc_titles = [f'Doc {i+1}' for i in range(len(docs))]

    tokenized_docs, vocab = preprocess_docs(docs)
    if len(vocab) == 0:
        raise ValueError('Empty vocabulary after preprocessing. Check input file encoding/content.')

    M = build_term_doc_matrix(tokenized_docs, vocab)

    results, U, S, Vt = analyze_svd(M, vocab, doc_titles, top_k=2, top_words=5)

    # Save topic visualizations
    for topic in results['topics']:
        idx = topic['index']
        words_weights = topic['top_words']
        bar_path = os.path.join(IMAGES_DIR, f'topic{idx+1}_top_words.png')
        wc_path = os.path.join(IMAGES_DIR, f'topic{idx+1}_wordcloud.png')
        plot_top_words(words_weights, bar_path, title=f'Topic {idx+1}: top-5 words')
        save_wordcloud(words_weights, wc_path)

    # Save global word cloud over the entire corpus vocabulary
    save_global_wordcloud(tokenized_docs, os.path.join(IMAGES_DIR, 'global_wordcloud.png'))

    # Save singular values plot
    plt.figure(figsize=(7, 5), dpi=150)
    plt.plot(np.arange(1, len(S)+1), S, marker='o')
    plt.xlabel('Index')
    plt.ylabel('Singular value')
    plt.title('Singular values of term-document matrix')
    plt.grid(True, ls=':')
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, 'singular_values.png'))
    plt.close()

    # Persist JSON summary for report
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Processed {len(docs)} documents; vocabulary size={len(vocab)}")
    print(f"Saved figures into {IMAGES_DIR}")


if __name__ == '__main__':
    main() 