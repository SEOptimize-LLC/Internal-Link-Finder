"""
Enhanced link opportunity analyzer
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class EnhancedLinkAnalyzer:
    """Analyze and identify internal link opportunities"""
    
    def __init__(self):
        """Initialize link analyzer"""
        self.opportunities_cache = None
    
    def analyze_opportunities(self, processed_data: Dict, 
                            gsc_data: Optional[Dict],
                            analysis_type: str) -> pd.DataFrame:
        """
        Find internal link opportunities
        
        Args:
            processed_data: Processed file data
            gsc_data: GSC metrics (optional)
            analysis_type: Type of analysis to perform
            
        Returns:
            DataFrame with link opportunities
        """
        try:
            logger.info(f"Starting {analysis_type} analysis")
            
            if analysis_type == "Find All Opportunities":
                opportunities = self._find_all_opportunities(processed_data)
            elif analysis_type == "Target Specific URLs":
                opportunities = self._analyze_specific_urls(processed_data)
            elif analysis_type == "Top Performers (GSC)":
                opportunities = self._analyze_top_performers(processed_data, gsc_data)
            else:
                opportunities = pd.DataFrame()
            
            # Add GSC metrics if available
            if gsc_data:
                opportunities = self._add_gsc_metrics(opportunities, gsc_data)
            
            # Sort by opportunity score
            if 'opportunity_score' in opportunities.columns:
                opportunities = opportunities.sort_values('opportunity_score', ascending=False)
            
            self.opportunities_cache = opportunities
            logger.info(f"Found {len(opportunities)} link opportunities")
            
            return opportunities
            
        except Exception as e:
            logger.error(f"Error analyzing opportunities: {str(e)}")
            return pd.DataFrame()
    
    def _find_all_opportunities(self, processed_data: Dict) -> pd.DataFrame:
        """Find all potential link opportunities"""
        
        opportunities = []
        links_df = processed_data['links_df']
        embeddings_lookup = processed_data['embeddings_lookup']
        
        # Get all pages
        all_pages = processed_data['pages']
        
        # Check each page pair
        for source in all_pages:
            for target in all_pages:
                if source != target:
                    # Check if link already exists
                    existing = links_df[
                        (links_df['source'] == source) & 
                        (links_df['destination'] == target)
                    ]
                    
                    if existing.empty:
                        # Calculate similarity if embeddings available
                        similarity = 0.0
                        if source in embeddings_lookup and target in embeddings_lookup:
                            emb1 = embeddings_lookup[source]
                            emb2 = embeddings_lookup[target]
                            if len(emb1) > 0 and len(emb2) > 0:
                                similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
                        
                        if similarity > 0.5:  # Threshold
                            opportunities.append({
                                'target_url': source,
                                'related_url': target,
                                'similarity_score': similarity,
                                'link_exists': 'No',
                                'opportunity_score': similarity
                            })
        
        return pd.DataFrame(opportunities)
    
    def _analyze_specific_urls(self, processed_data: Dict) -> pd.DataFrame:
        """Analyze specific URLs for opportunities"""
        # Simplified implementation
        return self._find_all_opportunities(processed_data).head(100)
    
    def _analyze_top_performers(self, processed_data: Dict, 
                               gsc_data: Optional[Dict]) -> pd.DataFrame:
        """Analyze top performing pages from GSC"""
        
        if not gsc_data:
            return pd.DataFrame()
        
        # Get top pages by clicks
        top_pages = sorted(
            gsc_data['url_clicks'].items(),
            key=lambda x: x[1],
            reverse=True
        )[:20]
        
        opportunities = []
        links_df = processed_data['links_df']
        
        for page, clicks in top_pages:
            # Find related pages without links
            related = self._find_related_pages(page, processed_data)
            
            for related_page, similarity in related:
                existing = links_df[
                    (links_df['source'] == page) & 
                    (links_df['destination'] == related_page)
                ]
                
                if existing.empty:
                    opportunities.append({
                        'target_url': page,
                        'related_url': related_page,
                        'similarity_score': similarity,
                        'target_clicks': clicks,
                        'opportunity_score': similarity * (1 + np.log1p(clicks) / 10)
                    })
        
        return pd.DataFrame(opportunities)
    
    def _find_related_pages(self, url: str, processed_data: Dict, 
                           top_n: int = 10) -> List[Tuple[str, float]]:
        """Find related pages for a URL"""
        
        embeddings_lookup = processed_data['embeddings_lookup']
        
        if url not in embeddings_lookup:
            return []
        
        target_emb = embeddings_lookup[url]
        if len(target_emb) == 0:
            return []
        
        similarities = []
        for other_url, other_emb in embeddings_lookup.items():
            if other_url != url and len(other_emb) > 0:
                sim = np.dot(target_emb, other_emb) / (np.linalg.norm(target_emb) * np.linalg.norm(other_emb))
                similarities.append((other_url, sim))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_n]
    
    def _add_gsc_metrics(self, opportunities: pd.DataFrame, 
                        gsc_data: Dict) -> pd.DataFrame:
        """Add GSC metrics to opportunities"""
        
        if opportunities.empty:
            return opportunities
        
        # Add metrics for target URLs
        opportunities['queries_count'] = opportunities['target_url'].map(
            gsc_data.get('url_queries_count', {})
        ).fillna(0)
        
        opportunities['clicks'] = opportunities['target_url'].map(
            gsc_data.get('url_clicks', {})
        ).fillna(0)
        
        opportunities['impressions'] = opportunities['target_url'].map(
            gsc_data.get('url_impressions', {})
        ).fillna(0)
        
        return opportunities