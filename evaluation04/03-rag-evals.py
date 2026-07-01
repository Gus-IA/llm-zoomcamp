import pandas as pd

df_ground_truth = pd.read_csv("data/ground_truth-new.csv")
ground_truth = df_ground_truth.to_dict(orient="records")


from ingest import load_faq_data, build_index

documents = load_faq_data()

documents_llm = []

for doc in documents:
    if doc["course"] == "llm-zoomcamp":
        documents_llm.append(doc)

documents = documents_llm
index = build_index(documents)

doc_idx = {}

for doc in documents:
    doc_idx[doc["id"]] = doc


from dotenv import load_dotenv
from groq import Groq

load_dotenv()
groq_client = Groq()

from evaluation_utils import RAGWithUsage

assistant = RAGWithUsage(
    index=index,
    llm_client=groq_client,
)

rec = ground_truth[0]
question = rec["question"]

answer_llm = assistant.rag(question)
print(answer_llm)

assistant.total_cost()

doc_id = rec["document"]
original_doc = doc_idx[doc_id]
answer_orig = original_doc["answer"]

print(answer_orig)

rag_result = {
    "question": question,
    "answer_llm": answer_llm,
    "answer_orig": answer_orig,
    "document": doc_id,
}

print(rag_result)


def generate_rag_answer(rec):
    question = rec["question"]
    doc_id = rec["document"]
    original_doc = doc_idx[doc_id]

    answer_llm = assistant.rag(question)
    answer_orig = original_doc["answer"]

    result = {
        "question": question,
        "answer_llm": answer_llm,
        "answer_orig": answer_orig,
        "document": doc_id,
    }

    return result


answer_record = generate_rag_answer(ground_truth[0])
print(answer_record)

assistant.total_cost()

assistant.reset_usage()


with ThreadPoolExecutor(max_workers=6) as pool:
    results = map_progress(pool, ground_truth, generate_rag_answer)

answers = []

for answer_record in results:
    answers.append(answer_record)


df_answers = pd.DataFrame(answers)
df_answers.to_csv("data/rag-answers-new.csv", index=False)
