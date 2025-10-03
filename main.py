from Services.data_upsertion import get_faqs_answers
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

llm = ChatOpenAI(model="gpt-4o-mini")


query = input("Enter your question: ")


chunks = get_faqs_answers(query)


chunks_with_content = "\n".join([chunk.page_content for chunk in chunks])


prompt = PromptTemplate(
    template="""
You are a helpful assistant. Use only the data provided below to answer the question.

Data:
{chunks_with_content}

Question:
{query}

Instructions:
- Answer accurately and concisely using **only** the provided data.
- If the answer is not in the provided data, respond exactly:
  "I don't know the exact answer. For authoritative information, please visit the UMT admissions website: https://admissions.umt.edu.pk"
- Do NOT invent facts or guess.
- Keep language simple and friendly and response user in their language.
""",
    input_variables=["chunks_with_content", "query"]
)


# Format the prompt with real data
formatted_prompt = prompt.format(
    chunks_with_content=chunks_with_content,
    query=query
)


response = llm.invoke(formatted_prompt)
print("Response:", response.content)