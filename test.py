import os
import torch
from dotenv import load_dotenv
load_dotenv('.env')
lms_host = os.getenv("LM_STUDIO_HOST")
hf_embedding_model = os.getenv("HF_EMBEDDING_MODEL")
openai_api_key = os.getenv("OPENAI_API_KEY")
hf_cache_folder  = os.getenv("HF_EMBEDDING_CACHE_FOLDER")

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(hf_embedding_model, lms_host, hf_cache_folder)

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
embedding =  HuggingFaceEmbedding(model_name=hf_embedding_model, 
                                  device=device,  
                                  trust_remote_code=True,)