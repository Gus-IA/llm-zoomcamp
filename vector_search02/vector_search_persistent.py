from sentence_transformers import SentenceTransformer
from sqlitesearch import VectorSearchIndex

model = SentenceTransformer("all-MiniLM-L6-v2")

vs_index = VectorSearchIndex(
    keyword_fields=["course"], mode="ivf", db_path="faq_vectors2.db"
)


query_vector = model.encode("How do I run Kafka?")
results = vs_index.search(query_vector, num_results=5)

print(results)

from rag_helper import RAGBase
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
groq_client = Groq()


class RAGVector(RAGBase):

    def __init__(self, embedder, **kwargs):
        super().__init__(**kwargs)
        self.embedder = embedder

    def search(self, query, num_results=5):
        query_vector = self.embedder.encode(query)
        filter_dict = {"course": self.course}

        return self.index.search(
            query_vector, num_results=num_results, filter_dict=filter_dict
        )


vector_assistant = RAGVector(
    embedder=model,
    index=vs_index,
    llm_client=groq_client,
)
