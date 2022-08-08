import time

from haystack.document_stores import FAISSDocumentStore
from haystack.nodes import EmbeddingRetriever
from datasets import load_dataset


data = load_dataset("imdb", split="test")
docs = [{"content": example["text"]} for example in data]

document_store = FAISSDocumentStore(sql_url="sqlite://")
document_store.write_documents(docs)

retriever = EmbeddingRetriever(document_store, embedding_model="distilbert-base-uncased")
document_store.update_embeddings(retriever)

start = time.time()
query = docs[0]["content"]
closest_doc = retriever.retrieve(query, top_k=1)
closest_doc_content = closest_doc[0].content
query_duration = time.time() - start

print(f"Query: {query}")
print()
print(f"Closest document: {closest_doc_content}")
print(f"Query duration: {query_duration}")