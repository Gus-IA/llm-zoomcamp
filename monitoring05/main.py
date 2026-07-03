def main():
    print("Hello from monitoring05!")


if __name__ == "__main__":
    main()

from urllib.request import urlretrieve

PREFIX = "https://raw.githubusercontent.com/DataTalksClub/llm-zoomcamp/main"

files = [
    "01-agentic-rag/code/ingest.py",
    "01-agentic-rag/code/rag_helper.py",
]

for file in files:
    url = f"{PREFIX}/{file}"
    filename = file.split("/")[-1]
    print(f"Descargando {filename}...")
    urlretrieve(url, filename)

print("Descarga completada.")
