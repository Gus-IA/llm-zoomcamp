from dotenv import load_dotenv

load_dotenv()

from ingest import load_faq_data, build_index
from rag_helper import RAGBase
from groq import Groq

documents = load_faq_data()
index = build_index(documents)

groq_client = Groq()

assistant = RAGBase(
    index=index,
    llm_client=groq_client,
)

answer = assistant.rag("I just discovered the course. Can I join now?")
print(answer)
