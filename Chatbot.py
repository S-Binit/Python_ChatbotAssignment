from ollama import Client
import json
import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os


EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "qwen3:8b-q4_K_M"
COLLECTION_NAME = "articles_demo"

chroma_client = chromadb.PersistentClient()
ollama_client = Client(host="http://localhost:11434")

collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME
)

counter = 0
if os.path.exists("counter.txt"):
    with open("counter.txt", "r") as f:
        counter = int(f.read())

splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=50,
    separators=["\n\n", "\n", "."]
)

print("Reading articles.jsonl and generating embeddings...")

with open("articles.jsonl", "r", encoding="utf-8") as f:
    for article_index, line in enumerate(f):

        if article_index < counter:
            print(f"Skipping article {article_index} (already processed)")
            continue

        line = line.strip()
        if not line:
            print(f"Skipping empty line at index {article_index}")
            continue

        try:
            article = json.loads(line)
        except json.JSONDecodeError as exc:
            print(f"Skipping malformed JSON at index {article_index}: {exc}")
            continue

        title = article.get("title", "")
        content = article.get("content", "")

        chunks = [
            c.strip()
            for c in splitter.split_text(content)
            if len(c.strip()) > 30
        ]

        for chunk_index, chunk in enumerate(chunks):
            embedding = ollama_client.embed(
                model=EMBED_MODEL,
                input=f"search_document: {chunk}"
            )["embeddings"][0]

            collection.add(
                ids=[f"article_{article_index}_chunk_{chunk_index}"],
                documents=[chunk],
                embeddings=[embedding],
                metadatas=[{
                    "title": title,
                    "article_id": article_index,
                    "chunk_id": chunk_index
                }]
            )

        with open("counter.txt", "w") as f_out:
            f_out.write(str(article_index + 1))

print("Database built successfully")

SYSTEM_PROMPT = """You are a helpful assistant.
Answer ONLY using the provided context.
If the context does not contain enough information, say "I don't know."
Keep answers concise and factual.
"""

def ask_chatbot(question, top_k=3):
    query_embed = ollama_client.embed(
        model=EMBED_MODEL,
        input=f"query: {question}"
    )["embeddings"][0]

    results = collection.query(
        query_embeddings=[query_embed],
        n_results=top_k,
    )

    retrieved_docs = results["documents"][0]
    context = "\n\n".join(retrieved_docs)
    prompt = f"""{SYSTEM_PROMPT}

Context:
{context}

Question:
{question}

Answer:
"""
    response = ollama_client.generate(
        model=CHAT_MODEL,
        prompt=prompt,
        options={"temperature": 0.1}
    )

    return response["response"].strip()

print("\nChatbot Ready!")
print("Type your question or type 'exit' to quit\n")

while True:
    user_query = input("Question: ")

    if user_query.lower() in ["exit", "quit"]:
        print("Goodbye!")
        break

    answer = ask_chatbot(user_query)
    print("\nBot Answer:", answer, "\n")