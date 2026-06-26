import urllib.request

PREFIX = "https://raw.githubusercontent.com/DataTalksClub/llm-zoomcamp/main"

files = [
    "01-agentic-rag/code/ingest.py",
    "01-agentic-rag/code/rag_helper.py",
    "04-evaluation/code/evaluation_utils.py",
]

for file in files:
    url = f"{PREFIX}/{file}"
    output_file = file.split("/")[-1]
    print(f"Descargando {url} -> {output_file}")
    urllib.request.urlretrieve(url, output_file)

print("Descarga completada.")


from ingest import load_faq_data

documents = load_faq_data()

print(documents[10])

documents_llm = []

for doc in documents:
    if doc["course"] == "llm-zoomcamp":
        documents_llm.append(doc)

print(len(documents_llm))

documents = documents_llm

doc = documents[0]
# print(doc["id"])
print(doc["question"])
print(doc["answer"])


from pydantic import BaseModel


class Questions(BaseModel):
    questions: list[str]


data_gen_instructions = """
You emulate a student who's taking our course.
Formulate 5 questions this student might ask based on a FAQ record. The record
should contain the answer to the questions, and the questions should be complete and not too short.
If possible, use as fewer words as possible from the record.

The output should resemble how people ask questions
on the internet. Not too formal, not too short, not too long.
""".strip()


from dotenv import load_dotenv
from groq import Groq

load_dotenv()
groq_client = Groq()


import json

user_prompt = json.dumps(doc)


messages = [
    {"role": "developer", "content": data_gen_instructions},
    {"role": "user", "content": user_prompt},
]

import os

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

response = client.chat.completions.create(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    messages=messages,
)

print(response)
