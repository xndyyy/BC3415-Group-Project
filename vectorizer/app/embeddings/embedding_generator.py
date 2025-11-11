from typing import Union, List
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

def generate_embedding(content: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
    """
    Generate embeddings using a local, free SentenceTransformer model.
    Works entirely offline after initial model download.
    """
    if isinstance(content, str):
        # Encode single string
        embedding = model.encode(content, convert_to_numpy=True).tolist()
        return embedding
    elif isinstance(content, list):
        # Encode list of strings
        embeddings = model.encode(content, convert_to_numpy=True).tolist()
        return embeddings
    else:
        raise ValueError("Content must be either a string or a list of strings")
