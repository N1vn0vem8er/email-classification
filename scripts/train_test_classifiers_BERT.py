import os
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    matthews_corrcoef, cohen_kappa_score, confusion_matrix
)
from gensim.models import KeyedVectors, Word2Vec, FastText
from transformers import AutoTokenizer, AutoModel, BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
from gensim.utils import simple_preprocess
from tqdm import tqdm
import joblib

INPUT_FILE = "spam_Emails_data.csv"
TEXT_COLUMN = "text"

df = pd.read_csv(INPUT_FILE)

df[TEXT_COLUMN] = df[TEXT_COLUMN].apply(
    lambda x: ' '.join(simple_preprocess(str(x)))
)

CACHE_DIR = "onemodelcachebert"
os.makedirs(CACHE_DIR, exist_ok=True)
MODELS_DIR = "bertmodels_random_state42"
os.makedirs(MODELS_DIR, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Używane urządzenie: {device}")

class TransformerEmbedder:
    def __init__(self, model_name, batch_size=32):
        self.model_name = model_name
        self.batch_size = batch_size
        self.tokenizer = None
        self.model = None

    def fit(self, texts):
        if self.tokenizer is None:
            self.tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-uncased")
            self.model = BertModel.from_pretrained("google-bert/bert-base-uncased").to(device)
            self.model.eval()

    def transform(self, texts):
        all_emb = []
        for i in tqdm(range(0, len(texts), self.batch_size), desc=f"Encoding {self.model_name}", leave=False):
            batch = texts[i:i + self.batch_size]
            inputs = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
            with torch.no_grad():
                out = self.model(**inputs)
                emb = out.last_hidden_state.mean(dim=1).cpu().numpy()
                all_emb.append(emb)
        return np.vstack(all_emb)

def manage_embeddings(embedder, emb_name, i, X_train_raw, X_test_raw):
    file_train = os.path.join(CACHE_DIR, f"emb_{emb_name}_iter{i}_train.npy")
    file_test = os.path.join(CACHE_DIR, f"emb_{emb_name}_iter{i}_test.npy")

    if os.path.exists(file_train) and os.path.exists(file_test):
        return np.load(file_train), np.load(file_test)

    print(f"Generating embeddings: {emb_name}")

    X_train_vec = embedder.transform(X_train_raw.tolist())
    X_test_vec = embedder.transform(X_test_raw.tolist())

    np.save(file_train, X_train_vec)
    np.save(file_test, X_test_vec)

    return X_train_vec, X_test_vec

def get_classifier(name):
    if name == 'RF': return RandomForestClassifier(n_estimators=100, random_state=42)
    if name == 'SVM': return LinearSVC(dual='auto', max_iter=2000, random_state=42)
    if name == 'MLP': return MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
    return None

results = []

classifiers = ['RF', 'SVM', 'MLP']

print(f"Starting Experiment")

bert = TransformerEmbedder("bert")
# bert.fit("")

embedding_methods = [
    (bert, "bert"),
]

for i in range(1, 31):
    for embedder, emb_name in embedding_methods:
        Y = df["label"]
        X = df[TEXT_COLUMN]
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=i)
        X_train_vec, X_test_vec = manage_embeddings(embedder, emb_name, i, X_train, X_test)
        for clf_name in classifiers:
            print(f"Itreration: {i}, Training: {clf_name}, Embedder: {emb_name}")
            clf = get_classifier(clf_name)
            clf.fit(X_train_vec, y_train)
            joblib.dump(clf, f"{MODELS_DIR}/{clf_name}_{i}.bin")
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
df_results.to_csv('wyniki_finalne_model_bert.csv', index=False)

pivot = df_results.pivot_table(index=['embedding', 'iteration'], columns='classifier', values='accuracy', aggfunc='mean')
print(pivot)
