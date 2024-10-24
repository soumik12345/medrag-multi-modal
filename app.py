import streamlit as st
import weave
from dotenv import load_dotenv

from medrag_multi_modal.assistant import (
    FigureAnnotatorFromPageImage,
    LLMClient,
    MedQAAssistant,
)
from medrag_multi_modal.retrieval import MedCPTRetriever

# Load environment variables
load_dotenv()

# Sidebar for configuration settings
st.sidebar.title("Configuration Settings")
project_name = st.sidebar.text_input(
    "Project Name",
    "ml-colabs/medrag-multi-modal"
)
chunk_dataset_name = st.sidebar.text_input(
    "Text Chunk WandB Dataset Name",
    "grays-anatomy-chunks:v0"
)
index_artifact_address = st.sidebar.text_input(
    "WandB Index Artifact Address",
    "ml-colabs/medrag-multi-modal/grays-anatomy-medcpt:v0",
)
image_artifact_address = st.sidebar.text_input(
    "WandB Image Artifact Address",
    "ml-colabs/medrag-multi-modal/grays-anatomy-images-marker:v6",
)
llm_model_name = st.sidebar.text_input(
    "LLM Client Model Name",
    "gemini-1.5-flash"
)
figure_extraction_model_name = st.sidebar.text_input(
    "Figure Extraction Model Name",
    "pixtral-12b-2409"
)
structured_output_model_name = st.sidebar.text_input(
    "Structured Output Model Name",
    "gpt-4o"
)

# Initialize Weave
weave.init(project_name=project_name)

# Initialize clients and assistants
llm_client = LLMClient(model_name=llm_model_name)
retriever = MedCPTRetriever.from_wandb_artifact(
    chunk_dataset_name=chunk_dataset_name,
    index_artifact_address=index_artifact_address,
)
figure_annotator = FigureAnnotatorFromPageImage(
    figure_extraction_llm_client=LLMClient(model_name=figure_extraction_model_name),
    structured_output_llm_client=LLMClient(model_name=structured_output_model_name),
    image_artifact_address=image_artifact_address,
)
medqa_assistant = MedQAAssistant(
    llm_client=llm_client, retriever=retriever, figure_annotator=figure_annotator
)

# Streamlit app layout
st.title("MedQA Assistant App")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display chat messages from history on app rerun
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat thread section with user input and response
if query := st.chat_input("What medical question can I assist you with today?"):
    # Add user message to chat history
    st.session_state.chat_history.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Process query and get response
    response = medqa_assistant.predict(query=query)
    st.session_state.chat_history.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)
