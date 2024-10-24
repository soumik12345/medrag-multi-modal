import streamlit as st
import weave
from dotenv import load_dotenv

from medrag_multi_modal.assistant import (
    FigureAnnotatorFromPageImage,
    LLMClient,
    MedQAAssistant,
)
from medrag_multi_modal.assistant.llm_client import (
    GOOGLE_MODELS,
    MISTRAL_MODELS,
    OPENAI_MODELS,
)
from medrag_multi_modal.retrieval import MedCPTRetriever

# Load environment variables
load_dotenv()

# Define constants
ALL_AVAILABLE_MODELS = GOOGLE_MODELS + MISTRAL_MODELS + OPENAI_MODELS

# Sidebar for configuration settings
st.sidebar.title("Configuration Settings")
project_name = st.sidebar.text_input(
    label="Project Name",
    value="ml-colabs/medrag-multi-modal",
    placeholder="wandb project name",
    help="format: wandb_username/wandb_project_name",
)
chunk_dataset_name = st.sidebar.text_input(
    label="Text Chunk WandB Dataset Name",
    value="grays-anatomy-chunks:v0",
    placeholder="wandb dataset name",
    help="format: wandb_dataset_name:version",
)
index_artifact_address = st.sidebar.text_input(
    label="WandB Index Artifact Address",
    value="ml-colabs/medrag-multi-modal/grays-anatomy-medcpt:v0",
    placeholder="wandb artifact address",
    help="format: wandb_username/wandb_project_name/wandb_artifact_name:version",
)
image_artifact_address = st.sidebar.text_input(
    label="WandB Image Artifact Address",
    value="ml-colabs/medrag-multi-modal/grays-anatomy-images-marker:v6",
    placeholder="wandb artifact address",
    help="format: wandb_username/wandb_project_name/wandb_artifact_name:version",
)
llm_client_model_name = st.sidebar.selectbox(
    label="LLM Client Model Name",
    options=ALL_AVAILABLE_MODELS,
    index=ALL_AVAILABLE_MODELS.index("gemini-1.5-flash"),
    help="select a model from the list",
)
figure_extraction_model_name = st.sidebar.selectbox(
    label="Figure Extraction Model Name",
    options=ALL_AVAILABLE_MODELS,
    index=ALL_AVAILABLE_MODELS.index("pixtral-12b-2409"),
    help="select a model from the list",
)
structured_output_model_name = st.sidebar.selectbox(
    label="Structured Output Model Name",
    options=ALL_AVAILABLE_MODELS,
    index=ALL_AVAILABLE_MODELS.index("gpt-4o"),
    help="select a model from the list",
)

# Initialize Weave
weave.init(project_name=project_name)

# Initialize clients and assistants
llm_client = LLMClient(model_name=llm_client_model_name)
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
