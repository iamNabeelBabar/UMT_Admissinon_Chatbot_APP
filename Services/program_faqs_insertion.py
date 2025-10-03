import os
import pandas as pd
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
# Load CSV file
# ----------------------------
csv_path = "programs_faq.csv"  # Replace with your CSV file path
df = pd.read_csv(csv_path)

# ----------------------------
# Convert CSV rows to Documents
# ----------------------------
documents = []
for _, row in df.iterrows():
    # Combine relevant fields into text for embedding
    content = (
        f"School: {row['School']}\n"
        f"Program Name: {row['Program Name']}\n"
        f"Duration: {row['Duration']}\n"
        f"Tentative Investment (PKR): {row['Tentative Investment (PKR)']}\n"
        f"Quarterly Installment (PKR): {row['Quarterly Installment (PKR)']}"
    )

    doc = Document(
        page_content=content,
        metadata={
            "school": row["School"],
            "program_name": row["Program Name"],
            "duration": row["Duration"],
            "tentative_investment": row["Tentative Investment (PKR)"],
            "quarterly_installment": row["Quarterly Installment (PKR)"]
        }
    )
    documents.append(doc)

print(f"✅ Loaded {len(documents)} documents from CSV.")

# ----------------------------
# Push Documents to Pinecone
# ----------------------------
index_name = "irfan-gpt-index"
namespace = "umt-programs-namespace"

# Create embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
uuids = [str(uuid4()) for _ in range(len(documents))]

# Upsert documents into Pinecone
vectorstore = PineconeVectorStore.from_documents(
    documents=documents,
    embedding=embeddings,
    index_name=index_name,
    namespace=namespace,
    ids=uuids
)

print(f"✅ Documents upserted to Pinecone index '{index_name}' in namespace '{namespace}'.")
