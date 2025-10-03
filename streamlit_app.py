import streamlit as st
from Services.data_upsertion import get_faqs_answers
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

# ----------------------------
# Streamlit page config
# ----------------------------
st.set_page_config(
    page_title="UMT Admission Chatbot",
    page_icon="https://admin.umt.edu.pk/Media/Site/UMT/FileManager/2020/newwebsite/umt-logo.png",
    layout="wide"
)

# ----------------------------
# Sticky top header
# ----------------------------
st.markdown("""
    <div style='position: sticky; top: 0; z-index: 999; text-align: center; 
                font-weight: bold; font-size:38px; background-color:#ffffff; padding:10px; 
                border-bottom: 2px solid #ddd;'>
        UMT ADMISSION CHATBOT
    </div>
""", unsafe_allow_html=True)

# ----------------------------
# Sidebar logo
# ----------------------------
st.sidebar.image(
    "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcS5ctdA-MS6WGj1BAT0fVXi_4I1OKcRY6p1Lg&s",
    use_container_width=True
)

# ----------------------------
# Display main image
# ----------------------------
st.image(
    "https://admin.umt.edu.pk/Media/Site/UMT/FileManager/2020/newwebsite/contact-us.jpg",
    caption="University of Management and Technology (UMT)",
    use_container_width=True
)

# ----------------------------
# Sidebar for API key
# ----------------------------
st.sidebar.header("Configuration")
api_key = st.sidebar.text_input(
    "OpenAI API Key (GPT-4o-mini):",
    type="password",
    help="Enter your OpenAI API key for GPT-4o-mini."
)

# ----------------------------
# Initialize chat history
# ----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ----------------------------
# Display chat messages with lime background for assistant
# ----------------------------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            st.markdown(
                f"<div style='background-color: rgba(0,255,0,0.3); padding:10px; border-radius:5px'>{message['content']}</div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(message["content"])

# ----------------------------
# Chat input
# ----------------------------
if prompt := st.chat_input("Ask a question about UMT admissions..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Check API key
    if not api_key:
        with st.chat_message("assistant"):
            st.error("Please enter your OpenAI API key in the sidebar.")
        st.session_state.messages.append({"role": "assistant", "content": "Please enter your OpenAI API key in the sidebar."})
    else:
        try:
            # Set up LLM
            llm = ChatOpenAI(model="gpt-4o-mini", api_key=api_key)

            # Get FAQ chunks
            with st.chat_message("assistant"):
                with st.spinner("Fetching relevant FAQs..."):
                    chunks = get_faqs_answers(prompt)
                    chunks_with_content = "\n".join([chunk.page_content for chunk in chunks])

                # Prompt template
                prompt_template = PromptTemplate(
                    template="""
You are a helpful assistant. Use only the data provided below to answer the question.

Data:
{chunks_with_content}

Question:
{query}

Instructions:
- Answer accurately and concisely using the provided data.
- If the answer is not in the provided data, respond exactly:
  "I don't know the exact answer. For authoritative information, please visit the UMT admissions website: https://admissions.umt.edu.pk"
- Keep language simple and friendly and respond to user in their language.
""",
                    input_variables=["chunks_with_content", "query"]
                )

                # Format and invoke
                formatted_prompt = prompt_template.format(
                    chunks_with_content=chunks_with_content,
                    query=prompt
                )

                with st.spinner("Generating response..."):
                    response = llm.invoke(formatted_prompt)
                    full_response = response.content

                    # Display assistant response with lime background
                    st.markdown(
                        f"<div style='background-color: rgba(0,255,0,0.3); padding:10px; border-radius:5px'>{full_response}</div>",
                        unsafe_allow_html=True
                    )

            # Add assistant message to history
            st.session_state.messages.append({"role": "assistant", "content": full_response})

        except Exception as e:
            error_msg = f"An error occurred: {str(e)}. Please check your API key and try again."
            with st.chat_message("assistant"):
                st.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})

# ----------------------------
# Footer
# ----------------------------
st.sidebar.markdown("---")
st.sidebar.info("Powered by LangChain, Streamlit, and OpenAI GPT-4o-mini.")
