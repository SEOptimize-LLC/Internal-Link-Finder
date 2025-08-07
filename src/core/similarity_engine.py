"""
Similarity calculation engine using embeddings
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from typing import Dict, List, Tuple  # Added Tuple import
import logging

logger = logging.getLogger(__name__)

class SimilarityEngine:
    """Calculate content similarities using embeddings"""
    
    def __init__(self):
        """Initialize similarity engine"""
        self.similarity_cache = {}
    
    def calculate_similarities(self, embeddings_df: pd.DataFrame, 
                             threshold: float = 0.7) -> pd.DataFrame:
        """
        Calculate pairwise similarities between all pages
        
        Args:
            embeddings_df: DataFrame with URL and embedding columns
            threshold: Minimum similarity threshold
            
        Returns:
            DataFrame with similarity scores
        """
        try:
            # Extract embeddings matrix
            urls = embeddings_df['url'].tolist()
            embeddings = np.vstack(embeddings_df['embedding'].values)
            
            # Calculate cosine similarity
            similarity_matrix = cosine_similarity(embeddings)
            
            # Create results list
            results = []
            for i, url1 in enumerate(urls):
                for j, url2 in enumerate(urls):
                    if i != j and similarity_matrix[i, j] >= threshold:
                        results.append({
                            'source_url': url1,
                            'target_url': url2,
                            'similarity_score': similarity_matrix[i, j]
                        })
            
            # Convert to DataFrame
            similarity_df = pd.DataFrame(results)
            similarity_df = similarity_df.sort_values('similarity_score', ascending=False)
            
            # Cache results
            self.similarity_cache = similarity_matrix
            
            logger.info(f"Calculated {len(similarity_df)} similarity pairs above threshold {threshold}")
            return similarity_df
            
        except Exception as e:
            logger.error(f"Error calculating similarities: {str(e)}")
            return pd.DataFrame()
    
    def get_most_similar(self, url: str, embeddings_df: pd.DataFrame, 
                        top_n: int = 10) -> List[Tuple[str, float]]:
        """Get most similar pages to a given URL"""
        try:
            # Find URL index
            urls = embeddings_df['url'].tolist()
            if url not in urls:
                return []
            
            idx = urls.index(url)
            
            # Get similarities
            if hasattr(self, 'similarity_cache') and self.similarity_cache is not None:
                similarities = self.similarity_cache[idx]
            else:
                # Calculate on demand
                embeddings = np.vstack(embeddings_df['embedding'].values)
                target_embedding = embeddings[idx].reshape(1, -1)
                similarities = cosine_similarity(target_embedding, embeddings)[0]
            
            # Sort and get top N
            similar_indices = np.argsort(similarities)[::-1][1:top_n+1]
            
            results = [(urls[i], similarities[i]) for i in similar_indices]
            return results
            
        except Exception as e:
            logger.error(f"Error getting similar pages: {str(e)}")
            return []
