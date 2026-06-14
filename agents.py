def rag(self, query):
    search_results = self.search(query)
    prompt = self.build_prompt(query, search_results)
    answer = self.llm(prompt)
    return answer


import os
from minsearch import Index
from groq import Groq

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

messages = [{"role": "user", "content": "I just discovered the course. Can I join it?"}]

response = client.chat.completions.create(
    model="llama-3.1-70b-versatile",
    messages=messages,
)

print(response.choices[0].message.content)

index = Index(text_fields=["question", "section", "answer"], keyword_fields=["course"])


def search(query):
    boost_dict = {"question": 3.0, "section": 0.5}
    filter_dict = {"course": "llm-zoomcamp"}

    return index.search(
        query, num_results=5, boost_dict=boost_dict, filter_dict=filter_dict
    )


search_tool = {
    "type": "function",
    "function": {
        "name": "search",
        "description": "Search the FAQ database for entries matching the given query.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query text to look up in the course FAQ.",
                }
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
}

response = client.chat.completions.create(
    model="llama-3.1-70b-versatile",
    messages=messages,
    tools=[search_tool],
)

print(response.choices[0].message.content)


import json

call = response.choices[0].message.content
args = json.loads(call.function.arguments)

results = search(**args)
result_json = json.dumps(results, indent=2)

messages.extend(response.choices[0].message.content)

messages.append(
    {
        "role": "tool",
        "tool_call_id": call.id,
        "content": result_json,
    }
)

response = client.chat.completions.create(
    model="llama-3.1 c-70b-versatile",
    messages=messages,
    tools=[search_tool],
)

print(response.choices[0].message.content)
