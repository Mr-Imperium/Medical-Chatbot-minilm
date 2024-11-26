import streamlit as st
from utils import MedicalKnowledgeBase
import os

# Configuration
PDF_PATH = 'gale.pdf'

@st.cache_resource
def load_knowledge_base():
    return MedicalKnowledgeBase(PDF_PATH)

def main():
    st.title('Medical Information Chatbot')
    
    # Load knowledge base
    kb = load_knowledge_base()
    
    # Chat input
    user_query = st.text_input('Ask a medical question:')
    
    if user_query:
        # Retrieve relevant information
        results = kb.retrieve_relevant_info(user_query)
        
        # Display results
        st.subheader('Relevant Information:')
        for i, result in enumerate(results, 1):
            st.markdown(f"**Result {i}:**\n{result}")
    
    # About section
    st.sidebar.header('About')
    st.sidebar.info(
        'This medical chatbot provides information '
        'based on the Gale Encyclopedia of Medicine. '
        'Always consult a healthcare professional '
        'for medical advice.'
    )

if __name__ == '__main__':
    main()
