import time

from transformers import AutoModel, AutoTokenizer
from scipy.spatial.distance import cosine
from datasets import load_dataset
from numpy import argmin, vstack
from tqdm import tqdm


def get_embeddings(texts, model, tokenizer, batch_size=16):
    embeddings = []
    cache = None
    for batch_index in tqdm(range(0, len(texts), batch_size)):
        if not cache:
            batch = texts[batch_index:batch_index + batch_size]
            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
            outputs = model(**inputs)
            cache = outputs
        else:
            outputs = cache

        embeddings.extend(outputs.last_hidden_state[:,0,:].detach().numpy())
    
    return vstack(embeddings)
       
data = load_dataset("imdb", split="test")
texts = [example["text"] for example in data]

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModel.from_pretrained("distilbert-base-uncased")


embeddings = get_embeddings(texts, model, tokenizer)
print(embeddings.shape)

start = time.time()
query_embedding = embeddings[0]
distances = [cosine(query_embedding, embedding) for embedding in embeddings]
query_duration = time.time() - start

print(argmin(distances))
print(f"Query duration: {query_duration}")