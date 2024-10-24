# MedQA Assistant App

The MedQA Assistant App is a Streamlit-based application designed to provide a chat interface for medical question answering. It leverages advanced language models (LLMs) and retrieval augmented generation (RAG) techniques to deliver accurate and informative responses to medical queries.

## Features

- **Interactive Chat Interface**: Engage with the app through a user-friendly chat interface.
- **Configurable Settings**: Customize model selection and data sources via the sidebar.
- **Retrieval-Augmented Generation**: Ensures precise and contextually relevant responses.
- **Figure Annotation Capabilities**: Extracts and annotates figures from medical texts.

## Usage

1. **Launch the App**: Start the application using Streamlit:
    ```bash
    streamlit run app.py
    ```
2. **Configure Settings**: Adjust configuration settings in the sidebar to suit your needs.
3. **Ask a Question**: Enter your medical question in the chat input field.
4. **Receive a Response**: Get a detailed answer from the MedQA Assistant.

## Configuration

The app allows users to customize various settings through the sidebar:

- **Project Name**: Specify the WandB project name.
- **Text Chunk WandB Dataset Name**: Define the dataset containing text chunks.
- **WandB Index Artifact Address**: Provide the address of the index artifact.
- **WandB Image Artifact Address**: Provide the address of the image artifact.
- **LLM Client Model Name**: Choose a language model for generating responses.
- **Figure Extraction Model Name**: Select a model for extracting figures from images.
- **Structured Output Model Name**: Choose a model for generating structured outputs.

## Technical Details

The app is built using the following components:

- **Streamlit**: For the user interface.
- **Weave**: For project initialization and artifact management.
- **MedQAAssistant**: For processing queries and generating responses.
- **LLMClient**: For interacting with language models.
- **MedCPTRetriever**: For retrieving relevant text chunks.
- **FigureAnnotatorFromPageImage**: For annotating figures in medical texts.

## Development and Deployment

- **Environment Setup**: Ensure all dependencies are installed as per the `pyproject.toml`.
- **Running the App**: Use Streamlit to run the app locally.
- **Deployment**: coming soon...

## Additional Resources

For more detailed information on the components and their usage, refer to the following documentation sections:

- [MedQA Assistant](/assistant/medqa_assistant)
- [LLM Client](/assistant/llm_client)
- [Figure Annotation](/assistant/figure_annotation)
