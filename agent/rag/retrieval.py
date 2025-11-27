import os
import numpy as np
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from assets.config import config

class Chunk:
    def __init__(self, id: str, content: str, source: str, score: float = 0.0):
        self.id = id
        self.content = content
        self.source = source
        self.score = score

class DocumentRetriever:
    def __init__(self, chunk_size: int = 300, top_k: int = 3):
        self.chunk_size = chunk_size
        self.top_k = top_k
        self.chunks = self._load_and_chunk_docs()
        self.vectorizer = TfidfVectorizer()
        if self.chunks:
            self.tfidf_matrix = self.vectorizer.fit_transform([chunk.content for chunk in self.chunks])
        else:
            self.tfidf_matrix = None


    def _load_and_chunk_docs(self) -> List[Chunk]:
        chunks = []
        docs_dir = config.DOCS_DIR
        
        if not os.path.exists(docs_dir):
            print(f"Warning: docs directory not found at {docs_dir}")
            return chunks
            
        for file in os.listdir(docs_dir):
            if file.endswith('.md'):
                filepath = os.path.join(docs_dir, file)
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                paragraphs = content.split('\n\n')
                for i, para in enumerate(paragraphs):
                    if para.strip():
                        chunk_id = f"{file.replace('.md', '')}::chunk{i}"
                        chunks.append(Chunk(chunk_id, para.strip(), file))
        
        return chunks

    def retrieve(self, query: str) -> List[Chunk]:
        if not self.chunks or self.tfidf_matrix is None:
            return []
        
        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[-self.top_k:][::-1]
        
        results = []
        for idx in top_indices:
            chunk = self.chunks[idx]
            chunk.score = float(similarities[idx])
            results.append(chunk)
        
        return results