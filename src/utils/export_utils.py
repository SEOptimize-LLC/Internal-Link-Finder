"""
Export utilities for various formats
"""

import pandas as pd
from datetime import datetime
import json
import logging
from typing import List, Optional
import io

logger = logging.getLogger(__name__)

class ExportUtils:
    """Handle data export in various formats"""
    
    def export_to_excel(self, opportunities: pd.DataFrame,
                       suggestions: Optional[List] = None) -> str:
        """
        Export to Excel with multiple sheets
        
        Args:
            opportunities: DataFrame with opportunities
            suggestions: List of content suggestions
            
        Returns:
            Path to exported file
        """
        try:
            # Create file path
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            file_path = f"link_opportunities_{timestamp}.xlsx"
            
            # Create Excel writer
            with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
                # Write opportunities
                opportunities.to_excel(writer, sheet_name='Opportunities', index=False)
                
                # Write suggestions if available
                if suggestions:
                    suggestions_df = self._format_suggestions(suggestions)
                    suggestions_df.to_excel(writer, sheet_name='Content Suggestions', index=False)
                
                # Add summary sheet
                summary_df = self._create_summary(opportunities)
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            logger.info(f"Exported to Excel: {file_path}")
            return file_path
            
        except Exception as e:
            logger.error(f"Error exporting to Excel: {str(e)}")
            raise
    
    def _format_suggestions(self, suggestions: List) -> pd.DataFrame:
        """Format suggestions for export"""
        
        formatted = []
        for suggestion in suggestions:
            if 'suggestions' in suggestion:
                for item in suggestion['suggestions']:
                    formatted.append({
                        'Target URL': suggestion.get('target_url', ''),
                        'Destination URL': suggestion.get('destination_url', ''),
                        **item
                    })
        
        return pd.DataFrame(formatted)
    
    def _create_summary(self, opportunities: pd.DataFrame) -> pd.DataFrame:
        """Create summary statistics"""
        
        summary = {
            'Metric': [
                'Total Opportunities',
                'Average Similarity Score',
                'Total Search Volume',
                'Pages with Opportunities'
            ],
            'Value': [
                len(opportunities),
                opportunities['similarity_score'].mean() if 'similarity_score' in opportunities.columns else 0,
                opportunities['monthly_search_volume'].sum() if 'monthly_search_volume' in opportunities.columns else 0,
                opportunities['target_url'].nunique() if 'target_url' in opportunities.columns else 0
            ]
        }
        
        return pd.DataFrame(summary)