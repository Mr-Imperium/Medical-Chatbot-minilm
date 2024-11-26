import streamlit as st
from transformers import AutoModelForCausalLM, LlamaTokenizer
import torch

class MedicalModelChatbot:
    def __init__(self):
        # Use a medical-focused model
        MODEL_NAME = "medalpaca/medalpaca-13b"
        
        try:
            # Explicitly use LlamaTokenizer
            self.tokenizer = LlamaTokenizer.from_pretrained(MODEL_NAME)
            
            # Load model with optimized settings
            self.model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME, 
                device_map='auto',
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True
            )
        except Exception as e:
            st.error(f"Model loading error: {e}")
            self.model = None
            self.tokenizer = None

    def generate_response(self, prompt):
        if not self.model:
            return "Model not initialized. Cannot generate response."
        
        # Enhance prompt with medical context
        full_prompt = f"Medical Question: {prompt}\n\nMedical Answer:"
        
        try:
            # Tokenize input
            inputs = self.tokenizer(full_prompt, return_tensors="pt")
            
            # Generate response
            outputs = self.model.generate(
                **inputs, 
                max_length=300, 
                num_return_sequences=1,
                temperature=0.7
            )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Post-process response
            response = response.split("Medical Answer:")[-1].strip()
            
            # Add disclaimer
            response += "\n\n⚠️ This is general medical information. " \
                        "Always consult a healthcare professional for specific advice."
            
            return response
        except Exception as e:
            return f"Response generation error: {e}"

def main():
    st.title("🩺 Medical Information Assistant")
    
    # Initialize chatbot
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = MedicalModelChatbot()
    
    # Sidebar
    st.sidebar.header("🏥 About This Assistant")
    st.sidebar.warning(
        "Provides AI-generated medical information. "
        "NOT a substitute for professional medical diagnosis or treatment."
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
    if prompt := st.chat_input("Ask a medical question"):
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

