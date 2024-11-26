import torch
import faiss
import numpy as np
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from sentence_transformers import SentenceTransformer

class MedicalKnowledgeBase:
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.documents = []
        self.embeddings = None
        self.index = None
        
        self.load_and_process_pdf()
    
    def load_and_process_pdf(self):
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
        self.embeddings = self.model.encode(texts)
        
        # Create FAISS index
        d = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(d)
        self.index.add(np.array(self.embeddings))
    
    def retrieve_relevant_info(self, query: str, top_k: int = 3) -> List[str]:
        # Embed query
        query_embedding = self.model.encode([query])[0]
        
        # Search in index
        D, I = self.index.search(np.array([query_embedding]), top_k)
        
        # Retrieve top results
        results = [self.documents[i].page_content for i in I[0]]
        return results
