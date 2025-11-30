import os
import re
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
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000, stop_words=None, min_df=1)
        
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
                
                # Split by sections or paragraphs
                if '##' in content:
                    sections = re.split(r'(##[^\n]+)', content)
                    current_section = ""
                    section_idx = 0
                    
                    for part in sections:
                        if part.startswith('##'):
                            if current_section.strip():
                                chunk_id = f"{file.replace('.md', '')}::chunk{section_idx}"
                                chunks.append(Chunk(chunk_id, current_section.strip(), file))
                                section_idx += 1
                            current_section = part + "\n"
                        else:
                            current_section += part
                    
                    # Add last section
                    if current_section.strip():
                        chunk_id = f"{file.replace('.md', '')}::chunk{section_idx}"
                        chunks.append(Chunk(chunk_id, current_section.strip(), file))
                else:
                    # Fallback to paragraph splitting
                    paragraphs = content.split('\n\n')
                    for i, para in enumerate(paragraphs):
                        if para.strip():
                            chunk_id = f"{file.replace('.md', '')}::chunk{i}"
                            chunks.append(Chunk(chunk_id, para.strip(), file))
        
        print(f"Loaded {len(chunks)} chunks from {len(os.listdir(docs_dir))} documents")
        return chunks

    def retrieve(self, query: str) -> List[Chunk]:
        if not self.chunks or self.tfidf_matrix is None:
            return []
        
        query_lower = query.lower()
        
        # Strategy 1: TF-IDF similarity
        query_vec = self.vectorizer.transform([query])
        tfidf_similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        tfidf_scores = {idx: score for idx, score in enumerate(tfidf_similarities)}
        
        # Strategy 2: keyword overlap
        query_tokens = set(query_lower.split())
        keyword_scores = {}
        
        for idx, chunk in enumerate(self.chunks):
            chunk_lower = chunk.content.lower()
            chunk_tokens = set(chunk_lower.split())
            
            # Jaccard similarity
            intersection = query_tokens.intersection(chunk_tokens)
            union = query_tokens.union(chunk_tokens)
            jaccard = len(intersection) / len(union) if union else 0
            
            # Exact phrase matching (high weight)
            exact_match_score = 0
            for token in query_tokens:
                if len(token) > 3 and token in chunk_lower:
                    exact_match_score += 0.15
            
            # Multi-word phrase matching
            if len(query_tokens) > 1:
                query_bigrams = [' '.join(pair) for pair in zip(list(query_tokens)[:-1], list(query_tokens)[1:])]
                for bigram in query_bigrams:
                    if bigram in chunk_lower:
                        exact_match_score += 0.2
            
            keyword_scores[idx] = jaccard + exact_match_score
        
        # Strategy 3: Document type relevance
        doc_type_scores = {}
        
        for idx, chunk in enumerate(self.chunks):
            score = 0.0
            source_lower = chunk.source.lower()
            chunk_lower = chunk.content.lower()
            
            # Return policy questions
            if any(term in query_lower for term in ['return', 'policy', 'days', 'window']):
                if 'policy' in source_lower:
                    score += 0.4
                    # Extra boost if beverages mentioned
                    if 'beverage' in query_lower and 'beverage' in chunk_lower:
                        score += 0.3
            
            # KPI and definition questions
            if any(term in query_lower for term in ['aov', 'kpi', 'definition', 'margin', 'metric', 'average order value', 'gross margin']):
                if 'kpi' in source_lower or 'definition' in source_lower:
                    score += 0.4
            
            # Marketing calendar questions
            if any(term in query_lower for term in ['summer', 'winter', 'campaign', 'marketing', 'dates', 'calendar', '1997']):
                if 'marketing' in source_lower or 'calendar' in source_lower:
                    score += 0.4
                    # Extra boost for specific campaigns
                    if 'summer beverages' in query_lower and 'summer beverages' in chunk_lower:
                        score += 0.3
                    if 'winter classics' in query_lower and 'winter classics' in chunk_lower:
                        score += 0.3
            
            # Category/catalog questions
            if any(term in query_lower for term in ['category', 'categories', 'product', 'catalog']):
                if 'catalog' in source_lower:
                    score += 0.3
            
            doc_type_scores[idx] = score
        
        # Strategy 4: Content-based relevance boosting
        content_boost_scores = {}
        
        for idx, chunk in enumerate(self.chunks):
            score = 0.0
            chunk_lower = chunk.content.lower()
            
            # Boost chunks that contain specific entities mentioned in query
            query_entities = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*|19\d{2}|20\d{2})\b', query)
            for entity in query_entities:
                if entity.lower() in chunk_lower:
                    score += 0.1
            
            # Boost chunks with numeric values for questions asking for numbers
            if any(term in query_lower for term in ['how many', 'what is', 'return', 'days']):
                if re.search(r'\d+', chunk.content):
                    score += 0.1
            
            content_boost_scores[idx] = score
        
        # Combine all strategies with weights
        combined_scores = {}
        for idx in range(len(self.chunks)):
            combined_scores[idx] = (
                0.35 * tfidf_scores.get(idx, 0) +       # TF-IDF similarity
                0.30 * keyword_scores.get(idx, 0) +     # Keyword overlap
                0.25 * doc_type_scores.get(idx, 0) +    # Document type relevance
                0.10 * content_boost_scores.get(idx, 0) # Content boosting
            )
        
        # Get top K results
        top_indices = sorted(
            combined_scores.keys(), 
            key=lambda x: combined_scores[x], 
            reverse=True
        )[:self.top_k]
        
        results = []
        for idx in top_indices:
            chunk = self.chunks[idx]
            chunk.score = float(combined_scores[idx])
            results.append(chunk)
        
        print(f"\n[RETRIEVAL] Query: {query}")
        for chunk in results:
            print(f"  - {chunk.id} (score: {chunk.score:.3f}): {chunk.content[:100]}...")
        
        return results