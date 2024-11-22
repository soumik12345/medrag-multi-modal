import streamlit as st


load_text_page = st.Page("medrag_app/load_text.py", title="Load Text", icon=":material/file_present:")
question_answering_page = st.Page("medrag_app/question_answer.py", title="Question-Answering Demo", icon=":material/robot:")

page_navigation = st.navigation([load_text_page, question_answering_page])
st.set_page_config(page_title="MedRaG", page_icon=":material/medical_services:")
page_navigation.run()
