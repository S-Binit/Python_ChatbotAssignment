from chromadb.utils import embedding_functions
import ollama
import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter

raw_text = """
John Smith was born in New York in 1985. He studied computer science at MIT from 2003 to 2007.
After graduation, he worked at Google for 5 years as a software engineer.
In 2012, John moved to San Francisco and started his own company called TechVision.
The company focused on artificial intelligence and machine learning solutions.
TechVision grew rapidly and by 2015 had over 100 employees. The company's main product
was an AI-powered customer service chatbot that could handle complex queries.
In 2018, John decided to sell TechVision to Amazon for $500 million. After the acquisition,
he became the VP of AI Research at Amazon, where he currently leads a team of 50 researchers.
John is married to Sarah Chen, whom he met at MIT. They have two children, Emma and Lucas.
The family lives in Seattle, Washington, where John commutes to Amazon's headquarters daily.
"""

chat_bot = ollama.Client(host="http://localhost:11434")

client = chromadb.PersistentClient()

ollama_ef = embedding_functions.OllamaEmbeddingFunction(
    model_name="nomic-embed-text", url="http://localhost:11434/api/embeddings"
)

collection = client.get_or_create_collection(
    name="info",
    embedding_function=ollama_ef,
)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200, chunk_overlap=20, separators=["\n"]
)

chunks = text_splitter.split_text(raw_text)

for i, chunk in enumerate(chunks):
    collection.add(
        documents=[chunk], ids=[f"chunk_{i}"], metadatas=[{"chunk_index": i}]
    )

query = "What company did John Smith start and who did he sell it to for how much?"

results = collection.query(query_texts=[query], n_results=2)

retrieved_docs = results['documents'][0]
context = "\n\n".join(retrieved_docs)


prompt = f"""You are a helpful assistant. Answer the question based on the context provided. Use the information in the context to form your answer. If context does not have enough information just say "I don't know"

Context: {context}

Question: {query}

Answer:"""

response = chat_bot.generate(
        model="qwen3:4b-q4_K_M",
        prompt=prompt,
        options={
            "temperature": 0.1
        }
    )

answer = response['response']

print(answer)
