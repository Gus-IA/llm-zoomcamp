Q1 = "I just discovcered the course, can I still join?"
Q2 = "I just found out abaout the program, can I still enroll?"

from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

v1 = model.encode(Q1)

print(v1.shape)


q1 = "Can I still join the course after the start date?"
v1 = model.encode(q1)

d = "You don't need to register. You're accepted. You can also just start learning and submitting homework without registering."
dv = model.encode(d)

print(v1.dot(dv))

q2 = "How to install Docker on Windows?"
v2 = model.encode(q2)

v2.dot(dv)

print(v2.dot(dv))

import requests

url = "https://raw.githubusercontent.com/DataTalksClub/llm-zoomcamp/main/01-agentic-rag/code/ingest.py"

response = requests.get(url)
response.raise_for_status()

with open("ingest.py", "wb") as f:
    f.write(response.content)

print("Archivo descargado correctamente")


from ingest import load_faq_data

documents = load_faq_data()

print(documents[10])

texts = []

for doc in documents:
    text = doc["question"] + " " + doc["answer"]
    texts.append(text)

from tqdm.auto import tqdm

batch_size = 50
vectors = []

for i in tqdm(range(0, len(texts), batch_size)):
    batch = texts[i : i + batch_size]
    batch_vectors = model.encode(batch)
    vectors.extend(batch_vectors)

print(len(vectors))

scores = []

for i in range(len(vectors)):
    score = v1.dot(vectors[i])
    scores.append(score)


import numpy as np

X = np.array(vectors)
print(X)

scores = X.dot(v1)
print(scores)

idx = np.argmax(scores)
idx, scores[idx]
print(idx)


print(documents[553])

top5 = np.argsort(scores)[-5:]
print(top5)

print(scores[top5])


for idx in top5:
    print(scores[idx])
    print(documents[idx])
    print()


top5negative = np.argsort(-scores)[:5]
print(top5negative)


from minsearch import VectorSearch

vindex = VectorSearch(keyword_fields=["course"])
vindex.fit(X, documents)

print(vindex.search(v1, num_results=5, filter_dict={"course": "llm-zoomcamp"}))
