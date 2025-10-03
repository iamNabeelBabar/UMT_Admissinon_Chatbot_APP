from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
import os

index_name = "irfan-gpt-index"  # Replace with your actual index name
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vectorstore = PineconeVectorStore(
    index_name=index_name,
    embedding=embeddings
)

# Specify the target namespaces for the retriever
target_namespaces = ["umt-faqs-namespace", "umt-programs-namespace"]

def get_faqs_answers(query: str):
    all_docs = []
    
    # Query each namespace separately and combine results
    for namespace in target_namespaces:
        # Create a retriever for the specific namespace
        retriever = vectorstore.as_retriever(
            search_kwargs={"namespace": namespace, "k": 5}  # k=5 means top 5 results per namespace
        )
        
        # Perform retrieval
        docs = retriever.invoke(query)
        all_docs.extend(docs)
    
    # Optionally, sort or deduplicate if needed, but for now just return combined
    return all_docs


# Example usage (uncomment to test):
# docs = get_faqs_answers("What are the FAQs?")
# print("Retrieved documents from multiple namespaces:")
# for doc in docs:
#     print("-" * 20)
#     print(doc.page_content)
#     print("Metadata:", doc.metadata)