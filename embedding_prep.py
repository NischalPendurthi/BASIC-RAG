import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('BAAI/bge-large-en-v1.5')

CHUNK_SIZE = 10000

reader = pd.read_csv("cleaned_dataset.csv", chunksize=CHUNK_SIZE)

d = 1024  # bge-large dimension
index = faiss.IndexFlatIP(d)  # use cosine (with normalization)

all_texts = []  # metadata storage

for chunk in reader:
    texts = (chunk["title"].fillna("") + " " + chunk["overview"].fillna("")).str.strip().tolist()

    # create embeddings
    embeddings = model.encode(
        texts,
        batch_size=128,
        show_progress_bar=True
    )
    
    embeddings = np.array(embeddings).astype("float32")

    # normalize for cosine similarity
    faiss.normalize_L2(embeddings)

    # add to index
    index.add(embeddings)

    # store metadata
    all_texts.extend(texts)

    print("Indexed:", len(all_texts))
