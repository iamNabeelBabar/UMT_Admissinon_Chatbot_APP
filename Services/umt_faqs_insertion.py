# import os
import json
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
from uuid import uuid4

# ----------------------------
# Load environment variables
# ----------------------------
load_dotenv()  # Make sure PINECONE_API_KEY, PINECONE_ENV, OPENAI_API_KEY are in .env

# ----------------------------
# Load FAQ JSON
# ----------------------------
data_path = "umt_faqs.json"  # Path to your JSON
with open(data_path, "r", encoding="utf-8") as f:
    data = json.load(f)["faqs"]

# ----------------------------
# Convert JSON into Documents
# ----------------------------
documents = []
for faq in data:
    doc = Document(
        page_content=faq["answer"],
        metadata={
            "question": faq["question"],
            "category": faq["category"]
        }
    )
    documents.append(doc)

print(f"✅ Loaded {len(documents)} documents.")

# ----------------------------
# Push Documents to Pinecone
# ----------------------------
index_name = "irfan-gpt-index"
namespace = "umt-faqs-namespace"

# Create embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
uuids = [str(uuid4()) for _ in range(len(documents))]

# Upsert documents into Pinecone
vectorstore = PineconeVectorStore.from_documents(
    documents=documents,
    embedding=embeddings,
    index_name=index_name,
    namespace="umt-faqs-namespace",
    ids=uuids
)

print(f"✅ Documents upserted to Pinecone index '{index_name}' in namespace '{namespace}'.")
