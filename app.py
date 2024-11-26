import streamlit as st
from transformers import pipeline

class MedicalChatbot:
    def __init__(self):
        try:
            # Use a smaller, more deployment-friendly model
            self.generator = pipeline(
                'text-generation', 
                model='microsoft/DialoGPT-medium',
                max_length=300
            )
        except Exception as e:
            st.error(f"Model loading error: {e}")
            self.generator = None

    def generate_response(self, prompt):
        if not self.generator:
            return "I'm having trouble generating a response right now."
        
        # Prefix to guide medical context
        medical_prompt = f"Medical advice context: {prompt} Provide a helpful, general response:"
        
        try:
            # Generate response
            response = self.generator(medical_prompt, max_length=300)[0]['generated_text']
            
            # Post-process response
            response = response.split(medical_prompt)[-1].strip()
            
            # Add disclaimer
            response += "\n\n*Note: This is general information and not a substitute for professional medical advice.*"
            
            return response
        except Exception as e:
            st.error(f"Response generation error: {e}")
            return "I apologize, but I'm unable to generate a response at the moment."

def main():
    st.title("🩺 Medical Information Assistant")
    
    # Initialize chatbot
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = MedicalChatbot()
    
    # Sidebar information
    st.sidebar.header("❗ Important Disclaimer")
    st.sidebar.warning(
        "This AI provides general health information only. "
        "It is NOT a substitute for professional medical advice, "
        "diagnosis, or treatment. Always consult a healthcare professional."
    )
    
    # Chat history management
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I'm a medical information assistant. What health-related question can I help you with today?"}
        ]
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # User input
    if prompt := st.chat_input("Ask a medical-related question"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate and display response
        with st.chat_message("assistant"):
            with st.spinner("Generating response..."):
                response = st.session_state.chatbot.generate_response(prompt)
                st.markdown(response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
