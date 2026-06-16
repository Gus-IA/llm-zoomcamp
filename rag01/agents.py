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

instructions = """
You're a course teaching assistant.
You're given a question from a course student and your task is to answer it.

If you want to look up information, use the search function. 
Use as many keywords from the user question as possible when making first requests.

Make multiple searches.

Try to expand your search by using new keywords
based on the results you get from the search.

At the end, ask if there are other areas that the user wants to explore.
""".strip()

response = client.chat.completions.create(
    model="llama-3.1 c-70b-versatile",
    messages=messages,
    tools=[search_tool],
)


def make_call(call):
    args = json.loads(call.arguments)

    if call.name == "search":
        result = search(**args)

    result_json = json.dumps(result, indent=2)

    return {
        "type": "function_call_output",
        "call_id": call.call_id,
        "output": result_json,
    }


messages.extend(response.output)

for item in response.output:

    if item.type == "funcion_call":
        print("function_call:", item.name, item.arguments)
        call_output = make_call(item)
        messages.append(call_output)

    elif item.type == "message":
        print("ASSISTANT:")
        print(item.content[0].text)


def agent_loop(instructions, question, model="llama-3.1 c-70b-versatile") -> str:
    messages = [
        {"role": "developer", "content": instructions},
        {"role": "user", "content": question},
    ]

    it = 1

    while True:
        print(f"iteration #{it}...")
        has_function_calls = False

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=[search_tool],
        )

        messages.extend(response.output)

        for item in response.output:
            if item.type == "function_call":
                print("function_call:", item.name, item.arguments)
                call_output = make_call(item)
                messages.append(call_output)
                has_function_calls = True

            elif item.type == "message":
                print("ASSISTANT:")
                last_answer = item.content[0].text
                print(item.content[0].text)

        it = it + 1
        if has_function_calls == False:
            break

    return last_answer


question = "what's queen gambit?"

result = agent_loop(instructions, question)


from toyaikit.llm import OpenAIClient
from toyaikit.tools import Tools
from toyaikit.chat import IPythonChatInterface
from toyaikit.chat.runners import OpenAIResponsesRunner, DisplayingRunnerCallback

agent_tools = Tools()
agent_tools.add_tool(search, search_tool)


def search(query: str) -> dict[str, str]:
    """
    Search the FAQ database for entries matching the given query.
    """
    return index.search(
        query,
        num_results=5,
        boost_dict={"question": 3.0, "section": 0.5},
        filter_dict={"course": "llm-zoomcamp"},
    )


agent_tools = Tools()
agent_tools.add_tool(search, search_tool)

chat_interface = IPythonChatInterface()
callback = DisplayingRunnerCallback(chat_interface)

response = client.chat.completions.create(
    model="llama-3.1-70b-versatile", messages=messages, tools=[search_tool]
)

result = response.loop(
    prompt="How do I run Ollama locally?",
    callback=callback,
)

result2 = response.loop(
    prompt="How do I run a different model?",
    previous_messages=result.all_messages,
    callback=callback,
)

response.run()
