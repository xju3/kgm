from transformers import AutoTokenizer, AutoModel
from llama_index.core.embeddings import BaseEmbedding
import torch
import numpy as np

class CustomBaseEmbedding(BaseEmbedding):
    def __init__(self, model_name: str, device: torch.device = None):
        """
        Initializes the custom embedding model.
        
        Args:
            model_name (str): The name of the Hugging Face model.
            device (torch.device, optional): The device to run the model on. Defaults to MPS if available.
        """
        # Set the device (default to MPS if available)
        self.device = device or torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.model_name = model_name
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
    
    def get_query_embedding(self, query: str) -> np.ndarray:
        """
        Generate embeddings for a single query.
        
        Args:
            query (str): The input query to embed.
        
        Returns:
            np.ndarray: The embeddings as a NumPy array.
        """
        return self._embed(query)
    
    def get_text_embedding(self, text: str) -> np.ndarray:
        """
        Generate embeddings for a single text.
        
        Args:
            text (str): The input text to embed.
        
        Returns:
            np.ndarray: The embeddings as a NumPy array.
        """
        return self._embed(text)
    
    def _embed(self, text: str) -> np.ndarray:
        """
        Tokenizes text and computes embeddings.
        
        Args:
            text (str): The text to embed.
        
        Returns:
            np.ndarray: The embeddings as a NumPy array.
        """
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        # Use mean pooling across the last