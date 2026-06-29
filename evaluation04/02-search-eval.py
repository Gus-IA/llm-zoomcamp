import pandas as pd

df_ground_truth = pd.read_csv("data/ground_truth-new.csv")
ground_truth = df_ground_truth.to_dict(orient="records")

print(ground_truth[10])

from ingest import load_faq_data, build_index

documents = load_faq_data()

documents_llm = []

for doc in documents:
    if doc["course"] == "llm-zoomcamp":
        documents_llm.append(doc)

documents = documents_llm
index = build_index(documents)


def text_search(query):
    boost_dict = {"question": 3.0, "section": 0.5}

    return index.search(query, num_results=5, boost_dict=boost_dict)


q = ground_truth[0]
print(q)

doc_id = q["document"]
results = text_search(query=q["question"])

for d in results:
    print(f'{d["id"]} == {doc_id}: {d["id"] == doc_id}')

relevance = []

for d in results:
    relevance.append(int(d["id"] == doc_id))

print(relevance)


def compute_relevance_text(q):
    doc_id = q["document"]
    results = text_search(query=q["question"])

    relevance = []
    for d in results:
        relevance.append(int(d["id"] == doc_id))

    return relevance


q = ground_truth[0]
print(q["question"])
compute_relevance_text(q)
# [1, 0, 0, 0, 0]

q = ground_truth[11]
print(q["question"])
compute_relevance_text(q)
# [1, 0, 0, 0, 0]

q = ground_truth[50]
print(q["question"])
compute_relevance_text(q)
# [1, 0, 0, 0, 0]


from tqdm.auto import tqdm


def compute_relevance_total_text(ground_truth):
    relevance_total = []

    for q in tqdm(ground_truth):
        relevance = compute_relevance_text(q)
        relevance_total.append(relevance)

    return relevance_total


ground_truth_sample = ground_truth[:15]
relevance_total_text = compute_relevance_total_text(ground_truth_sample)

print(relevance_total_text)


def compute_relevance(q, search_function):
    doc_id = q["document"]
    results = search_function(query=q["question"])

    relevance = []
    for d in results:
        relevance.append(int(d["id"] == doc_id))

    return relevance


def compute_relevance_total(ground_truth, search_function):
    relevance_total = []

    for q in tqdm(ground_truth):
        relevance = compute_relevance(q, search_function)
        relevance_total.append(relevance)

    return relevance_total


relevance_total = compute_relevance_total(ground_truth_sample, text_search)
print(relevance_total)


example = [
    [1, 0, 0, 0, 0],
    [0, 1, 0, 0, 0],
    [1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0],
    [1, 0, 0, 0, 0],
    [1, 0, 0, 0, 0],
    [1, 0, 0, 0, 0],
    [1, 0, 0, 0, 0],
    [0, 0, 1, 0, 0],
    [1, 0, 0, 0, 0],
    [1, 0, 0, 0, 0],
    [1, 0, 0, 0, 0],
    [1, 0, 0, 0, 0],
    [1, 0, 0, 0, 0],
]


cnt = 0

for line in example:
    if 1 in line:
        cnt = cnt + 1

print(cnt)
print(cnt / len(example))


def hit_rate(relevance):
    cnt = 0

    for line in relevance:
        if 1 in line:
            cnt = cnt + 1

    return cnt / len(relevance)


print(hit_rate(example))

print(example[1])


total_score = 0.0

for line in example:
    for rank in range(len(line)):
        if line[rank] == 1:
            total_score = total_score + 1 / (rank + 1)
            break

print(total_score)

print(total_score / len(example))


def mrr(relevance):
    total_score = 0.0

    for line in relevance:
        for rank in range(len(line)):
            if line[rank] == 1:
                total_score = total_score + 1 / (rank + 1)
                break

    return total_score / len(relevance)


print(mrr(example))


def evaluate(ground_truth, search_function):
    relevance_total = compute_relevance_total(ground_truth, search_function)

    return {
        "hit_rate": hit_rate(relevance_total),
        "mrr": mrr(relevance_total),
    }


print(evaluate(ground_truth, text_search))
