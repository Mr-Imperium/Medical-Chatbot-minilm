import os
import torch
import faiss
import numpy as np
from typing import List

# Explicitly import document loader
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

class MedicalKnowledgeBase:
    def __init__(self, pdf_path: str):
        # Add error handling for PDF path
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        # Force CPU usage for torch
        torch.set_default_device('cpu')
        
        self.pdf_path = pdf_path
        
        # Use a simpler embedding model with explicit device
        self.model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
        
        self.documents = []
        self.embeddings = None
        self.index = None
        
        self.load_and_process_pdf()
    
    def load_and_process_pdf(self):
        try:
            # Load PDF
            loader = PyPDFLoader(self.pdf_path)
            documents = loader.load()
            
            # Split documents
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500, 
                chunk_overlap=100
            )
            self.documents = text_splitter.split_documents(documents)
            
            # Generate embeddings
            texts = [doc.page_content for doc in self.documents]
            self.embeddings = self.model.encode(texts, convert_to_numpy=True)
            
            # Create FAISS index
            d = self.embeddings.shape[1]
            self.index = faiss.IndexFlatL2(d)
            self.index.add(self.embeddings)
        
        except Exception as e:
            print(f"Error processing PDF: {e}")
            raise

    def retrieve_relevant_info(self, query: str, top_k: int = 3) -> List[str]:
        try:
            # Embed query
            query_embedding = self.model.encode([query], convert_to_numpy=True)[0]
            
            # Search in index
            D, I = self.index.search(np.array([query_embedding]), top_k)
            
            # Retrieve top results
            results = [self.documents[i].page_content for i in I[0]]
            return results
        
        except Exception as e:
            print(f"Error retrieving information: {e}")
            return ["Unable to retrieve relevant information."]
