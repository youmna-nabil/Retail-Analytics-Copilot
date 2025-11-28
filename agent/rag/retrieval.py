import os
import numpy as np
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from assets.config import config
from assets.settings import settings

class Chunk:
    def __init__(self, id: str, content: str, source: str, score: float = 0.0):
        self.id = id
        self.content = content
        self.source = source
        self.score = score

class DocumentRetriever:
    def __init__(self):
        self.chunk_size = settings.ChunkSize
        self.top_k = settings.TopK

        self.chunks = self._load_and_chunk_docs()
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2),  max_features=5000, stop_words=None)
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
        tfidf_similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        
        # Strategy 1: TF-IDF similarity
        tfidf_scores = {idx: score for idx, score in enumerate(tfidf_similarities)}
        
        # Strategy 2: Keyword overlap scoring
        query_lower = query.lower()
        query_tokens = set(query_lower.split())
        
        keyword_scores = {}
        for idx, chunk in enumerate(self.chunks):
            chunk_lower = chunk.content.lower()
            chunk_tokens = set(chunk_lower.split())
            
            # Calculate Jaccard similarity for keyword overlap
            intersection = query_tokens.intersection(chunk_tokens)
            union = query_tokens.union(chunk_tokens)
            jaccard = len(intersection) / len(union) if union else 0
            
            # Boost score if chunk contains exact phrases from query
            exact_phrase_boost = 0
            for token in query_tokens:
                if len(token) > 3 and token in chunk_lower:
                    exact_phrase_boost += 0.1
            
            keyword_scores[idx] = jaccard + exact_phrase_boost
        
        # Strategy 3: Document type relevance
        doc_type_scores = {}
        for idx, chunk in enumerate(self.chunks):
            score = 0.0
            source_lower = chunk.source.lower()
            
            if any(term in query_lower for term in ['policy', 'return', 'days', 'window']):
                if 'policy' in source_lower:
                    score += 0.3
            
            if any(term in query_lower for term in ['aov', 'kpi', 'definition', 'margin', 'metric']):
                if 'kpi' in source_lower or 'definition' in source_lower:
                    score += 0.3
            
            if any(term in query_lower for term in ['campaign', 'marketing', 'summer', 'winter', 'dates', 'calendar']):
                if 'marketing' in source_lower or 'calendar' in source_lower:
                    score += 0.3
            
            if any(term in query_lower for term in ['category', 'categories', 'product', 'catalog']):
                if 'catalog' in source_lower:
                    score += 0.3
            
            doc_type_scores[idx] = score
        
        combined_scores = {}
        for idx in range(len(self.chunks)):
            combined_scores[idx] = (
                0.5 * tfidf_scores.get(idx, 0) +
                0.3 * keyword_scores.get(idx, 0) +
                0.2 * doc_type_scores.get(idx, 0)
            )
        
        top_indices = sorted(combined_scores.keys(), key=lambda x: combined_scores[x], reverse=True)[:self.top_k]
        
        results = []
        for idx in top_indices:
            chunk = self.chunks[idx]
            chunk.score = float(combined_scores[idx])
            results.append(chunk)
        
        return results