import os
import requests
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    matthews_corrcoef, cohen_kappa_score, confusion_matrix
)
from sklearn.model_selection import train_test_split
from gensim.utils import simple_preprocess
import joblib

INPUT_FILE = "spam_Emails_data.csv"
TEXT_COLUMN = "text"

df = pd.read_csv(INPUT_FILE)

df[TEXT_COLUMN] = df[TEXT_COLUMN].apply(
    lambda x: ' '.join(simple_preprocess(str(x)))
)

NUM_DATASETS = 30
OLLAMA_API_URL = "http://localhost:11434/api/embed"
EMBEDDINGS_CACHE_DIR = "embeddings_cache_embdemma"
MODELS_DIR = "modelsgemma_rs42"

os.makedirs(EMBEDDINGS_CACHE_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


def get_ollama_embeddings(texts, model_name="embeddinggemma", batch_size=100):
    all_embeddings = []
    texts_list = texts.tolist()

    for i in tqdm(range(0, len(texts_list), batch_size), desc=f"Generating vectors ({model_name})"):
        batch = texts_list[i:i + batch_size]
        payload = {
            "model": model_name,
            "input": batch
        }

        response = requests.post(OLLAMA_API_URL, json=payload)

        if response.status_code == 200:
            data = response.json()
            if 'embeddings' in data:
                all_embeddings.extend(data['embeddings'])
            else:
                raise ValueError(f"API error: {data}")
        else:
            raise ConnectionError(f"Ollama API error: {response.status_code} - {response.text}")

    return np.array(all_embeddings)

def manage_embeddings(emb_name, dataset_id, X_train_raw, X_test_raw):
    cache_train_path = f"{EMBEDDINGS_CACHE_DIR}/train_{dataset_id}_{emb_name}.npy"
    cache_test_path = f"{EMBEDDINGS_CACHE_DIR}/test_{dataset_id}_{emb_name}.npy"

    if os.path.exists(cache_train_path) and os.path.exists(cache_test_path):
        print(f"Wczytywanie z cache dla {emb_name}, dataset {dataset_id}")
        X_train_vec = np.load(cache_train_path)
        X_test_vec = np.load(cache_test_path)
    else:
        print(f"Generating vectors for {emb_name}, dataset {dataset_id}")
        X_train_vec = get_ollama_embeddings(X_train_raw, model_name=emb_name)
        X_test_vec = get_ollama_embeddings(X_test_raw, model_name=emb_name)

        np.save(cache_train_path, X_train_vec)
        np.save(cache_test_path, X_test_vec)

    return X_train_vec, X_test_vec

def get_classifier(name):
    if name == 'RF': return RandomForestClassifier(n_estimators=100, random_state=42)
    if name == 'SVM': return LinearSVC(dual='auto', max_iter=2000, random_state=42)
    if name == 'MLP': return MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
    return None

embedding_methods = ["embeddinggemma"]
classifiers = ["RF", "SVM", "MLP"]
results = []

print("Starting experiment")

for dataset_id in range(1, NUM_DATASETS + 1):
    print(f"\nProcessing {dataset_id}/{NUM_DATASETS}")
    for emb_name in embedding_methods:
        try:
            Y = df["label"]
            X = df[TEXT_COLUMN]
            X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=dataset_id)
            X_train_vec, X_test_vec = manage_embeddings(emb_name, dataset_id, X_train, X_test)
        except Exception as e:
            print(f"Błąd przy generowaniu embeddingów {emb_name}: {e}")
            continue

        for clf_name in classifiers:
            print(f"Trenowanie {clf_name} na wektorach {emb_name}")
            clf = get_classifier(clf_name)
            clf.fit(X_train_vec, y_train)
            joblib.dump(clf, f"{MODELS_DIR}/{clf_name}_{dataset_id}.bin")
            preds = clf.predict(X_test_vec)

            cm = confusion_matrix(y_test, preds)
            if cm.shape == (2, 2):
                cm_text = f"TN:{cm[0][0]} FP:{cm[0][1]} | FN:{cm[1][0]} TP:{cm[1][1]}"
            else:
                cm_text = str(cm.tolist())

            results.append({
                'dataset_id': dataset_id,
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

print("\nFinished")
df_results = pd.DataFrame(results)
df_results.to_csv('wyniki_finalne_gemma.csv', index=False)
