# 🩺 Medical Information Chatbot

## 📝 Project Description

This Medical Information Chatbot is an intelligent Streamlit application designed to provide quick and accessible medical information retrieval using the Gale Encyclopedia of Medicine. The application allows users to ask medical questions and receive relevant, sourced information instantly.

## 🌟 Key Features

- **Interactive Medical Search**: Input medical queries and receive precise, relevant information
- **Comprehensive Knowledge Base**: Powered by the Gale Encyclopedia of Medicine
- **User-Friendly Interface**: Simple and intuitive Streamlit-based design
- **Quick Information Retrieval**: Efficient backend knowledge extraction
- **Informative Sidebar**: Includes application context and important disclaimers

## 🛠 Technology Stack

- **Language**: Python
- **Framework**: Streamlit
- **Data Source**: Gale Encyclopedia of Medicine
- **Key Libraries**: 
  - Streamlit
  - Custom MedicalKnowledgeBase utility

## 🚀 Installation and Setup

### Prerequisites

- Python 3.8+
- pip package manager

### Installation Steps

1. Clone the repository
   ```bash
   git clone https://github.com/mr-imperium/medical-chatbot-minilm.git
   cd medical-chatbot-minilm
   ```

2. Create a virtual environment (optional but recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install required dependencies
   ```bash
   pip install -r requirements.txt
   ```

4. Ensure the Gale Encyclopedia PDF is in the project directory
   - Rename or configure the PDF path in the script if necessary

## 🖥 Running the Application

```bash
streamlit run medical_chatbot.py
```

## 🔍 Usage

1. Launch the application
2. Type your medical question in the input box
3. Browse through the retrieved relevant information
4. Check the sidebar for application context

## ⚠️ Important Disclaimer

**Medical Disclaimer**: 
This chatbot is designed for informational purposes only. The information provided should not be considered medical advice, diagnosis, or treatment. Always consult with qualified healthcare professionals for medical guidance, accurate diagnosis, and personalized treatment plans.

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📋 To-Do / Future Enhancements

- [ ] Implement advanced NLP for more accurate query matching
- [ ] Add citation links to source materials
- [ ] Create unit tests for knowledge base retrieval
- [ ] Develop a more sophisticated ranking system for results
- [ ] Add language translation support

## 📜 License

[MIT]
