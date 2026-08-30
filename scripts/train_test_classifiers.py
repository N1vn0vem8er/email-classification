import os
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    matthews_corrcoef, cohen_kappa_score, confusion_matrix
)
from gensim.models import KeyedVectors, Word2Vec, FastText
from sklearn.model_selection import train_test_split
from gensim.utils import simple_preprocess

INPUT_FILE = "../spam_Emails_data.csv"
TEXT_COLUMN = "text"

df = pd.read_csv(INPUT_FILE)

df[TEXT_COLUMN] = df[TEXT_COLUMN].apply(
    lambda x: ' '.join(simple_preprocess(str(x)))
)


PATH_GLOVE = 'glove_to_word2vec_vectorsalldata.txt'
CACHE_DIR = "embedding_cache_cbow"
MODELS_CACHE = "models_cache_42_cbow"
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(MODELS_CACHE, exist_ok=True)


class GensimEmbedder:
    def __init__(self, method='word2vec', vector_size=100, pretrained_path=None):
        self.method = method
        self.vector_size = vector_size
        self.model = None
        self.pretrained_path = pretrained_path

    def fit(self, texts):
        tokenized = [str(t).split() for t in texts]
        if self.method == 'word2vec':
            self.model = Word2Vec(sentences=tokenized, vector_size=self.vector_size, min_count=10, workers=1, seed=42, sg=1)
        elif self.method == 'fasttext':
            self.model = FastText(sentences=tokenized, vector_size=self.vector_size, min_count=10, workers=1, seed=42, sg=1)
        elif self.method == 'glove':
            if self.pretrained_path and os.path.exists(self.pretrained_path):
                self.model = KeyedVectors.load_word2vec_format(self.pretrained_path)
                self.vector_size = self.model.vector_size
            else:
                self.model = Word2Vec(sentences=tokenized, vector_size=self.vector_size, min_count=1)

    def transform(self, texts):
        embeddings = []
        for text in texts:
            words = str(text).split()
            vectors = []
            for word in words:
                if self.method == 'glove':
                    if word in self.model: vectors.append(self.model[word])
                else:
                    if word in self.model.wv: vectors.append(self.model.wv[word])

            if len(vectors) > 0:
                embeddings.append(np.mean(vectors, axis=0))
            else:
                embeddings.append(np.zeros(self.vector_size))
        return np.array(embeddings)

def manage_embeddings(embedder, emb_name, i, X_train_raw, X_test_raw):
    file_train = os.path.join(CACHE_DIR, f"emb_{emb_name}_iter{i}_train.npy")
    file_test = os.path.join(CACHE_DIR, f"emb_{emb_name}_iter{i}_test.npy")

    if os.path.exists(file_train) and os.path.exists(file_test):
        return np.load(file_train), np.load(file_test)

    print(f"Generating embeddings: {emb_name}")

    X_train_vec = embedder.transform(X_train_raw)
    X_test_vec = embedder.transform(X_test_raw)

    np.save(file_train, X_train_vec)
    np.save(file_test, X_test_vec)

    return X_train_vec, X_test_vec

def get_classifier(name):
    if name == 'RF': return RandomForestClassifier(n_estimators=100, random_state=42)
    if name == 'SVM': return LinearSVC(dual='auto', max_iter=2000, random_state=42)
    if name == 'MLP': return MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
    return None

classifiers = ['RF', 'SVM', 'MLP']

print(f"Start eksperymentu.")
trained_models = [
                  {"word2vec": "../vec100_sg_rs12/word2vecvec100_sg_rs12.bin", "fasttext": "../vec100_sg_rs12/fasttextvec100_sg_rs12.bin", "rs": 12, "vec_size": 100},
                  {"word2vec": "../vec300_sg_rs12/word2vecvec300_sg_rs12.bin", "fasttext": "../vec300_sg_rs12/fasttextvec300_sg_rs12.bin", "rs": 12, "vec_size": 300},
                  {"word2vec": "../vec100_sg_rs22/word2vecvec100_sg_rs22.bin", "fasttext": "../vec100_sg_rs22/fasttextvec100_sg_rs22.bin", "rs": 22, "vec_size": 100},
                  {"word2vec": "../vec300_sg_rs22/word2vecvec300_sg_rs22.bin", "fasttext": "../vec300_sg_rs22/fasttextvec300_sg_rs22.bin", "rs": 22, "vec_size": 300},
                  {"word2vec": "../vec100_sg_rs32/word2vecvec100_sg_rs32.bin", "fasttext": "../vec100_sg_rs32/fasttextvec100_sg_rs32.bin", "rs": 32, "vec_size": 100},
                  {"word2vec": "../vec300_sg_rs32/word2vecvec300_sg_rs32.bin", "fasttext": "../vec300_sg_rs32/fasttextvec300_sg_rs32.bin", "rs": 32, "vec_size": 300},
                  {"word2vec": "../vec100_sg_rs42/word2vecvec100_sg_rs42.bin", "fasttext": "../vec100_sg_rs42/fasttextvec100_sg_rs42.bin", "rs": 42, "vec_size": 100},
                  {"word2vec": "../vec300_sg_rs42/word2vecvec300_sg_rs42.bin", "fasttext": "../vec300_sg_rs42/fasttextvec300_sg_rs42.bin", "rs": 42, "vec_size": 300},
                  {"word2vec": "../vec100_sg_rs52/word2vecvec100_sg_rs52.bin", "fasttext": "../vec100_sg_rs52/fasttextvec100_sg_rs52.bin", "rs": 52, "vec_size": 100},
                  {"word2vec": "../vec300_sg_rs52/word2vecvec300_sg_rs52.bin", "fasttext": "../vec300_sg_rs52/fasttextvec300_sg_rs52.bin", "rs": 52, "vec_size": 300},
                  {"word2vec": "../vec100_sg_rs62/word2vecvec100_sg_rs62.bin", "fasttext": "../vec100_sg_rs62/fasttextvec100_sg_rs62.bin", "rs": 62, "vec_size": 100},
                  {"word2vec": "../vec300_sg_rs62/word2vecvec300_sg_rs62.bin", "fasttext": "../vec300_sg_rs62/fasttextvec300_sg_rs62.bin", "rs": 62, "vec_size": 300},
                  {"word2vec": "../vec100_sg_rs72/word2vecvec100_sg_rs72.bin", "fasttext": "../vec100_sg_rs72/fasttextvec100_sg_rs72.bin", "rs": 72, "vec_size": 100},
                  {"word2vec": "../vec300_sg_rs72/word2vecvec300_sg_rs72.bin", "fasttext": "../vec300_sg_rs72/fasttextvec300_sg_rs72.bin", "rs": 72, "vec_size": 300},
                  {"word2vec": "../vec100_sg_rs82/word2vecvec100_sg_rs82.bin", "fasttext": "../vec100_sg_rs82/fasttextvec100_sg_rs82.bin", "rs": 82, "vec_size": 100},
                  {"word2vec": "../vec300_sg_rs82/word2vecvec300_sg_rs82.bin", "fasttext": "../vec300_sg_rs82/fasttextvec300_sg_rs82.bin", "rs": 82, "vec_size": 300},
                  {"word2vec": "../vec100_sg_rs92/word2vecvec100_sg_rs92.bin", "fasttext": "../vec100_sg_rs92/fasttextvec100_sg_rs92.bin", "rs": 92, "vec_size": 100},
                  {"word2vec": "../vec300_sg_rs92/word2vecvec300_sg_rs92.bin", "fasttext": "../vec300_sg_rs92/fasttextvec300_sg_rs92.bin", "rs": 92, "vec_size": 300},
                  {"word2vec": "../vec100_sg_rs102/word2vecvec100_sg_rs102.bin", "fasttext": "../vec100_sg_rs102/fasttextvec100_sg_rs102.bin", "rs": 102, "vec_size": 100},
                  {"word2vec": "../vec300_sg_rs102/word2vecvec300_sg_rs102.bin", "fasttext": "../vec300_sg_rs102/fasttextvec300_sg_rs102.bin", "rs": 102, "vec_size": 300},
                  {"word2vec": "../vec100_cbow_rs12/word2vecvec100_cbow_rs12.bin", "fasttext": "../vec100_cbow_rs12/fasttextvec100_cbow_rs12.bin", "rs": 12, "vec_size": 100},
                  {"word2vec": "../vec300_cbow_rs12/word2vecvec300_cbow_rs12.bin", "fasttext": "../vec300_cbow_rs12/fasttextvec300_cbow_rs12.bin", "rs": 12, "vec_size": 300},
                  {"word2vec": "../vec100_cbow_rs22/word2vecvec100_cbow_rs22.bin", "fasttext": "../vec100_cbow_rs22/fasttextvec100_cbow_rs22.bin", "rs": 22, "vec_size": 100},
                  {"word2vec": "../vec300_cbow_rs22/word2vecvec300_cbow_rs22.bin", "fasttext": "../vec300_cbow_rs22/fasttextvec300_cbow_rs22.bin", "rs": 22, "vec_size": 300},
                  {"word2vec": "../vec100_cbow_rs32/word2vecvec100_cbow_rs32.bin", "fasttext": "../vec100_cbow_rs32/fasttextvec100_cbow_rs32.bin", "rs": 32, "vec_size": 100},
                  {"word2vec": "../vec300_cbow_rs32/word2vecvec300_cbow_rs32.bin", "fasttext": "../vec300_cbow_rs32/fasttextvec300_cbow_rs32.bin", "rs": 32, "vec_size": 300},
                  {"word2vec": "../vec100_cbow_rs42/word2vecvec100_cbow_rs42.bin", "fasttext": "../vec100_cbow_rs42/fasttextvec100_cbow_rs42.bin", "rs": 42, "vec_size": 100},
                  {"word2vec": "../vec300_cbow_rs42/word2vecvec300_cbow_rs42.bin", "fasttext": "../vec300_cbow_rs42/fasttextvec300_cbow_rs42.bin", "rs": 42, "vec_size": 300},
                  {"word2vec": "../vec100_cbow_rs52/word2vecvec100_cbow_rs52.bin", "fasttext": "../vec100_cbow_rs52/fasttextvec100_cbow_rs52.bin", "rs": 52, "vec_size": 100},
                  {"word2vec": "../vec300_cbow_rs52/word2vecvec300_cbow_rs52.bin", "fasttext": "../vec300_cbow_rs52/fasttextvec300_cbow_rs52.bin", "rs": 52, "vec_size": 300},
                  {"word2vec": "../vec100_cbow_rs62/word2vecvec100_cbow_rs62.bin", "fasttext": "../vec100_cbow_rs62/fasttextvec100_cbow_rs62.bin", "rs": 62, "vec_size": 100},
                  {"word2vec": "../vec300_cbow_rs62/word2vecvec300_cbow_rs62.bin", "fasttext": "../vec300_cbow_rs62/fasttextvec300_cbow_rs62.bin", "rs": 62, "vec_size": 300},
                  {"word2vec": "../vec100_cbow_rs72/word2vecvec100_cbow_rs72.bin", "fasttext": "../vec100_cbow_rs72/fasttextvec100_cbow_rs72.bin", "rs": 72, "vec_size": 100},
                  {"word2vec": "../vec300_cbow_rs72/word2vecvec300_cbow_rs72.bin", "fasttext": "../vec300_cbow_rs72/fasttextvec300_cbow_rs72.bin", "rs": 72, "vec_size": 300},
                  {"word2vec": "../vec100_cbow_rs82/word2vecvec100_cbow_rs82.bin", "fasttext": "../vec100_cbow_rs82/fasttextvec100_cbow_rs82.bin", "rs": 82, "vec_size": 100},
                  {"word2vec": "../vec300_cbow_rs82/word2vecvec300_cbow_rs82.bin", "fasttext": "../vec300_cbow_rs82/fasttextvec300_cbow_rs82.bin", "rs": 82, "vec_size": 300},
                  {"word2vec": "../vec100_cbow_rs92/word2vecvec100_cbow_rs92.bin", "fasttext": "../vec100_cbow_rs92/fasttextvec100_cbow_rs92.bin", "rs": 92, "vec_size": 100},
                  {"word2vec": "../vec300_cbow_rs92/word2vecvec300_cbow_rs92.bin", "fasttext": "../vec300_cbow_rs92/fasttextvec300_cbow_rs92.bin", "rs": 92, "vec_size": 300},
                  {"word2vec": "../vec100_cbow_rs102/word2vecvec100_cbow_rs102.bin", "fasttext": "../vec100_cbow_rs102/fasttextvec100_cbow_rs102.bin", "rs": 102, "vec_size": 100},
                  {"word2vec": "../vec300_cbow_rs102/word2vecvec300_cbow_rs102.bin", "fasttext": "../vec300_cbow_rs102/fasttextvec300_cbow_rs102.bin", "rs": 102, "vec_size": 300},]
                  # {"word2vec": "GoogleNews-vectors-negative300.bin", "rs": 42, "vec_size": 300}, {"fasttext": "wiki-news-300d-1M.vec", "rs": 42, "vec_size": 300}]



for embModel in trained_models:

    results = []

    word2vec = GensimEmbedder("word2vec", vector_size=embModel['vec_size'])
    word2vec.model= Word2Vec.load(embModel['word2vec'])

    fasttext = GensimEmbedder("fasttext", vector_size=embModel['vec_size'])
    fasttext.model = FastText.load(embModel['fasttext'])

    embedding_methods = [
        (word2vec, "word2vec"),
        (fasttext, "fasttext"),
    ]

    for i in range(1, 11):
        for embedder, emb_name in embedding_methods:
            Y = df["label"]
            X = df[TEXT_COLUMN]
            X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=i)
            X_train_vec, X_test_vec = manage_embeddings(embedder, f"{emb_name}_{embModel['rs']}_{embModel['vec_size']}", i, X_train, X_test)
            for clf_name in classifiers:
                print(f"Itreration: {i}, Training: {clf_name}, Embedder: {emb_name}")
                clf = get_classifier(clf_name)
                clf.fit(X_train_vec, y_train)
                joblib.dump(clf, f"{MODELS_CACHE}/{clf_name}_{emb_name}_{i}_{embModel['rs']}_{embModel['vec_size']}.bin")
                preds = clf.predict(X_test_vec)
                cm = confusion_matrix(y_test, preds)
                cm_text = f"TN:{cm[0][0]} FP:{cm[0][1]} | FN:{cm[1][0]} TP:{cm[1][1]}"
                results.append({
                    'iteration': i,
                    'embedding': emb_name,
                    'classifier': clf_name,
                    'accuracy': accuracy_score(y_test, preds),
                    'f1_score': f1_score(y_test, preds, pos_label="Spam"),
                    'precision': precision_score(y_test, preds, pos_label="Spam"),
                    'recall': recall_score(y_test, preds, pos_label="Spam"),
                    'mcc': matthews_corrcoef(y_test, preds),
                    'kappa': cohen_kappa_score(y_test, preds),
                    'conf_matrix': cm_text
                })
            pd.DataFrame(results).to_csv('wyniki_partial.csv', index=False)

    df_results = pd.DataFrame(results)
    df_results.to_csv(f"wyniki_finalne_model_{embModel['rs']}_{embModel['vec_size']}.csv", index=False)

    print("\nŚrednie wyniki (Accuracy)")
    pivot = df_results.pivot_table(index=['embedding', 'iteration'], columns='classifier', values='accuracy', aggfunc='mean')
    print(pivot)
