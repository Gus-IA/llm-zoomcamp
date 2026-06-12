from dotenv import load_dotenv
import os

load_dotenv()

from groq import Groq

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def llm(prompt):
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile", messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content


llm("Hey, what's up?")

question = "I just discovered the course. Can I join now?"
answer = llm(question)
print(answer)

context = """
I just discovered the course. Can I still join?
Yes, but if you want to receive a certificate, you need to submit your project while we're still accepting submissions.

Course: I have registered for the LLM Zoomcamp. When can I expect to receive the confirmation email?
You don't need it. You're accepted. You can also just start learning and submitting homework (while the form is open) without registering. It is not checked against any registered list. Registration is just to gauge interest before the start date.

What is the video/zoom link to the stream for the "Office Hours" or live/workshop sessions?
The zoom link is only published to instructors/presenters/TAs. Students participate via YouTube Live and submit questions to Slido.

Cloud alternatives with GPU
Check the quota and reset cycle carefully. Potential options include Google Colab, Kaggle, Databricks.
"""

prompt = f"""
your task is to answer questions from the course participants based on the provided context.

Use the context to find relevant information and provide accurate answers. If the answer is not found in the context, respond with "I don't know."

Question
{question}

Context:
{context}
"""

question = "I just discovered the course. Can I join now?"
answer = llm(prompt)
print(answer)


import requests

docs_url = "https://datatalks.club/faq/json/courses.json"
response = requests.get(docs_url)
courses_raw = response.json()

documents = []
url_prefix = "https://datatalks.club/faq"

for course in courses_raw:
    course_url = f"""{url_prefix}{course["path"]}"""

    course_response = requests.get(course_url)
    course_response.raise_for_status()
    course_data = course_response.json()

    documents.extend(course_data)

print(len(documents))


from minsearch import Index

index = Index(text_fields=["question", "section", "answer"], keyword_fields=["course"])

index.fit(documents)

search_results = index.search(
    question,
    boost_dict={"question": 2.0},
    filter_dict={"course": "llm-zoomcamp"},
    num_results=5,
)

print(search_results)


def search(question, course="llm-zoomcamp"):
    boost_dict = {"question": 2.0, "section": 0.5}
    filter_dict = {"course": course}

    return index.search(
        question, boost_dict=boost_dict, filter_dict=filter_dict, num_results=5
    )


search_results = search(question)
print(search_results)

INSTRUCTIONS = """
Your task is to answer questions from the course participants
based on the provided context.

Use the context to find relevant information and provide accurate
answers. If the answer is not found in the context,
respond with "I don't know."
"""

USER_PROMPT_TEMPLATE = """
Question:
{question}

Context:
{context}
"""


def build_context(search_results):
    lines = []

    for doc in search_results:
        lines.append(doc["section"])
        lines.append("Q: " + doc["question"])
        lines.append("A: " + doc["answer"])
        lines.append("")

    return "\n".join(lines).strip()


context = build_context(search_results)
print(context)


def build_prompt(question, search_results):
    context = build_context(search_results)
    prompt = USER_PROMPT_TEMPLATE.format(question=question, context=context)
    return prompt.strip()


prompt = build_prompt(question, search_results)
print(prompt)

response = client.chat.completions.create(
    model="llama-3.3-70b-versatile", messages=[{"role": "user", "content": prompt}]
)

print(response)
print(response.usage)

# price
input_price = 0.75 / 1_000_000
output_price = 4.50 / 1_000_000

cost = (
    response.usage.prompt_tokens * input_price
    + response.usage.prompt_tokens * output_price
)

print(cost)

message_history = [
    {"role": "developer", "content": INSTRUCTIONS},
    {"role": "user", "content": prompt},
]

response = client.chat.completions.create(
    model="llama-3.3-70b-versatile", messages=message_history
)


def llm(instructions, user_prompt, model="llama-3.3-70b-versatile"):
    message_history = [
        {"role": "developer", "content": instructions},
        {"role": "user", "content": user_prompt},
    ]

    response = client.chat.completions.create(model=model, messages=message_history)

    return response.choices[0].message.content


def rag(query, model="llama-3.3-70b-versatile"):
    search_results = search(query)
    prompt = build_prompt(query, search_results)
    answer = llm(INSTRUCTIONS, prompt, model=model)
    return answer


answer = rag(question)
print(answer)
