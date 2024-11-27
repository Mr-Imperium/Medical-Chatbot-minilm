import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class MedicalModelChatbot:
    def __init__(self):
        # Alternative medical models to try
        MODELS = [
            "medalpaca/medalpaca-13b",
            "huggyllama/llama-7b",
            "meta-llama/Llama-2-7b-chat-hf"
        ]
        
        self.model = None
        self.tokenizer = None
        
        # Try multiple models
        for MODEL_NAME in MODELS:
            try:
                # Load tokenizer with fallback options
                self.tokenizer = AutoTokenizer.from_pretrained(
                    MODEL_NAME, 
                    use_fast=False,  # Disable fast tokenizer
                    trust_remote_code=True  # Allow custom tokenizer code
                )
                
                # Load model with optimized settings
                self.model = AutoModelForCausalLM.from_pretrained(
                    MODEL_NAME, 
                    device_map='auto',
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True
                )
                
                # If successful, break the loop
                break
            
            except Exception as e:
                st.warning(f"Failed to load {MODEL_NAME}: {e}")
                continue
        
        # Check if any model was successfully loaded
        if self.model is None:
            st.error("Could not load any medical language model.")

    def generate_response(self, prompt):
        if not self.model:
            return "Model not initialized. Cannot generate response."
        
        # Enhance prompt with medical context
        full_prompt = f"""You are a helpful medical information assistant. 
        Provide clear, accurate, and concise medical information.
        Do NOT give specific medical diagnoses.

        Medical Question: {prompt}

        Medical Answer:"""
        
        try:
            # Tokenize input
            inputs = self.tokenizer(full_prompt, return_tensors="pt")
            
            # Generate response
            outputs = self.model.generate(
                **inputs, 
                max_length=300, 
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True
            )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Post-process response
            response = response.split("Medical Answer:")[-1].strip()
            
            # Add disclaimer
            response += "\n\n⚠️ IMPORTANT: " \
                        "This is general medical information. " \
                        "Always consult a healthcare professional for specific medical advice."
            
            return response
        
        except Exception as e:
            return f"Response generation error: {str(e)}"

def main():
    st.set_page_config(
        page_title="Medical Information Assistant",
        page_icon="🩺"
    )
    
    st.title("🩺 Medical Information Assistant")
    
    # Initialize chatbot
    if 'chatbot' not in st.session_state:
        with st.spinner("Loading medical AI model..."):
            st.session_state.chatbot = MedicalModelChatbot()
    
    # Sidebar
    st.sidebar.header("🏥 About This Assistant")
    st.sidebar.warning(
        "This AI provides GENERAL health information. " 
        "It is NOT a substitute for professional medical diagnosis or treatment."
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
            with st.spinner("Generating medical information..."):
                response = st.session_state.chatbot.generate_response(prompt)
                st.markdown(response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
