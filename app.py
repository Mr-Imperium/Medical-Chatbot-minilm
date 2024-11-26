import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Chatbot Configuration
MODEL_NAME = "BioMistral/BioMistral-7B-DARE"

# Load Model and Tokenizer
@st.cache_resource
def load_model():
    try:
        # Use 4-bit quantization for reduced memory usage
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, 
            device_map="auto", 
            load_in_4bit=True,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16
            )
        )
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

# Generate Medical Response
def generate_medical_response(model, tokenizer, prompt):
    try:
        # Prepare input
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # Generate response
        outputs = model.generate(
            **inputs, 
            max_length=300, 
            num_return_sequences=1, 
            temperature=0.7,
            top_p=0.9
        )
        
        # Decode response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response
    except Exception as e:
        st.error(f"Error generating response: {e}")
        return "I'm unable to generate a response right now."

# Streamlit App
def main():
    st.title("🩺 Medical Assistant Chatbot")
    
    # Model Loading
    model, tokenizer = load_model()
    
    if model is None or tokenizer is None:
        st.error("Failed to load the medical AI model.")
        return
    
    # Sidebar for Additional Info
    st.sidebar.header("About the Chatbot")
    st.sidebar.info(
        "This AI provides medical information and should not replace "
        "professional medical advice. Always consult a healthcare professional."
    )
    
    # Chat Interface
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I'm your medical AI assistant. How can I help you today?"}
        ]
    
    # Display Chat Messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # User Input
    if prompt := st.chat_input("Ask a medical question"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate and display assistant response
        with st.chat_message("assistant"):
            response = generate_medical_response(model, tokenizer, prompt)
            st.markdown(response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

# Run the app
if __name__ == "__main__":
    main()