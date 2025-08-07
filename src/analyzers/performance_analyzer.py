"""
Performance metrics analyzer
"""

import pandas as pd
import numpy as np
from typing import Dict
import logging

logger = logging.getLogger(__name__)

class PerformanceAnalyzer:
    """Analyze performance metrics and enhance with additional data"""
    
    def enhance_with_search_volumes(self, opportunities: pd.DataFrame,
                                   search_volumes: Dict[str, int]) -> pd.DataFrame:
        """
        Add search volume data to opportunities
        
        Args:
            opportunities: DataFrame with opportunities
            search_volumes: Dictionary of search volumes
            
        Returns:
            Enhanced DataFrame
        """
        try:
            if opportunities.empty or not search_volumes:
                return opportunities
            
            # Add search volume column
            opportunities['monthly_search_volume'] = 0
            
            # Map search volumes (simplified - in reality would map based on queries)
            for keyword, volume in search_volumes.items():
                # Match keywords to URLs (simplified logic)
                mask = opportunities['target_url'].str.contains(keyword, case=False, na=False)
                opportunities.loc[mask, 'monthly_search_volume'] += volume
            
            # Recalculate opportunity score with search volume
            if 'opportunity_score' in opportunities.columns:
                volume_factor = np.log1p(opportunities['monthly_search_volume']) / 10
                opportunities['opportunity_score'] = (
                    opportunities['opportunity_score'] * 0.7 + 
                    volume_factor * 0.3
                )
            
            logger.info(f"Enhanced {len(opportunities)} opportunities with search volumes")
            return opportunities
            
        except Exception as e:
            logger.error(f"Error enhancing with search volumes: {str(e)}")
            return opportunities
    
    def calculate_performance_metrics(self, data: pd.DataFrame) -> Dict:
        """Calculate performance metrics summary"""
        
        metrics = {
            'total_opportunities': len(data),
            'avg_similarity': data['similarity_score'].mean() if 'similarity_score' in data.columns else 0,
            'total_search_volume': data['monthly_search_volume'].sum() if 'monthly_search_volume' in data.columns else 0,
            'high_value_opportunities': len(data[data['opportunity_score'] > 0.8]) if 'opportunity_score' in data.columns else 0
        }
        
        return metrics