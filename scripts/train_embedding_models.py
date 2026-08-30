import os
import numpy as np
import pandas as pd
from gensim.models import KeyedVectors, Word2Vec, FastText
from gensim.utils import simple_preprocess

INPUT_FILE = "spam_Emails_data.csv"
TEXT_COLUMN = "text"

df = pd.read_csv(INPUT_FILE)

df[TEXT_COLUMN] = df[TEXT_COLUMN].apply(
    lambda x: ' '.join(simple_preprocess(str(x)))
)

class GensimEmbedder:
    def __init__(self, method='word2vec', vector_size=100, pretrained_path=None):
        self.method = method
        self.vector_size = vector_size
        self.model = None
        self.pretrained_path = pretrained_path

    def fit(self, texts, seed, arch):
        tokenized = [str(t).split() for t in texts]
        if self.method == 'word2vec':
            self.model = Word2Vec(sentences=tokenized, vector_size=self.vector_size, min_count=10, workers=1, seed=seed, sg=arch)
        elif self.method == 'fasttext':
            self.model = FastText(sentences=tokenized, vector_size=self.vector_size, min_count=10, workers=1, seed=seed, sg=arch)
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

modelssg = [
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
                  {"word2vec": "../vec300_sg_rs102/word2vecvec300_sg_rs102.bin", "fasttext": "../vec300_sg_rs102/fasttextvec300_sg_rs102.bin", "rs": 102, "vec_size": 300},]
modelscbow = [
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

print(f"Start trenowania.")

for model in modelssg:
    print("Trenowanie word2vec")
    word2vec = GensimEmbedder("word2vec", vector_size=model['vec_size'], seed=model['rs'], arch=1)
    word2vec.fit(df[TEXT_COLUMN])
    os.makedirs(os.path.dirname(model['word2vec']), exist_ok=True)
    word2vec.model.save(model['word2vec'])
    print("Trenowanie fasttext")
    fasttext = GensimEmbedder("fasttext", vector_size=model['vec_size'], seed=model['rs'], arch=1)
    fasttext.fit(df[TEXT_COLUMN])
    fasttext.model.save(model['fasttext'])

for model in modelscbow:
    print("Trenowanie word2vec")
    word2vec = GensimEmbedder("word2vec", vector_size=model['vec_size'], seed=model['rs'], arch=0)
    word2vec.fit(df[TEXT_COLUMN])
    os.makedirs(os.path.dirname(model['word2vec']), exist_ok=True)
    word2vec.model.save(model['word2vec'])
    print("Trenowanie fasttext")
    fasttext = GensimEmbedder("fasttext", vector_size=model['vec_size'], seed=model['rs'], arch=0)
    fasttext.fit(df[TEXT_COLUMN])
    fasttext.model.save(model['fasttext'])
