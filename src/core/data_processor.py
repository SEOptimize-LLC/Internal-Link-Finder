"""
Enhanced Data Processor for handling multiple file uploads and data integration
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from io import StringIO, BytesIO
import json

from config import config, COLUMN_MAPPINGS

logger = logging.getLogger(__name__)

class EnhancedDataProcessor:
    """Enhanced data processor for handling multiple file types"""
    
    def __init__(self):
        """Initialize the data processor"""
        self.config = config
        self.column_mappings = COLUMN_MAPPINGS
        self.processed_data = {}
    
    def process_multiple_files(self, files_dict: Dict) -> Dict[str, Any]:
        """
        Process multiple uploaded files
        
        Args:
            files_dict: Dictionary with keys 'links', 'embeddings', 'gsc' (optional)
        
        Returns:
            Dictionary with processed data
        """
        try:
            logger.info("Starting multi-file processing")
            
            # Process internal links file
            links_df = self._process_links_file(files_dict['links'])
            
            # Process embeddings file
            embeddings_df = self._process_embeddings_file(files_dict['embeddings'])
            
            # Process GSC file if provided
            gsc_df = None
            if 'gsc' in files_dict and files_dict['gsc']:
                gsc_df = self._process_gsc_file(files_dict['gsc'])
            
            # Combine and validate data
            processed_data = self._combine_data(links_df, embeddings_df, gsc_df)
            
            logger.info("Multi-file processing completed successfully")
            return processed_data
            
        except Exception as e:
            logger.error(f"Error processing files: {str(e)}")
            raise
    
    def _process_links_file(self, file) -> pd.DataFrame:
        """Process Screaming Frog internal links export"""
        try:
            # Read CSV
            df = pd.read_csv(file)
            
            # Map columns
            df = self._map_columns(df, 'screaming_frog_links')
            
            # Clean and validate
            df = self._clean_links_data(df)
            
            # Filter internal links only
            df = df[df['type'].str.lower() == 'internal']
            
            logger.info(f"Processed {len(df)} internal links")
            return df
            
        except Exception as e:
            logger.error(f"Error processing links file: {str(e)}")
            raise
    
    def _process_embeddings_file(self, file) -> pd.DataFrame:
        """Process Screaming Frog embeddings export"""
        try:
            # Read CSV
            df = pd.read_csv(file)
            
            # Map columns
            df = self._map_columns(df, 'screaming_frog_embeddings')
            
            # Process embeddings
            df = self._process_embeddings(df)
            
            logger.info(f"Processed embeddings for {len(df)} pages")
            return df
            
        except Exception as e:
            logger.error(f"Error processing embeddings file: {str(e)}")
            raise
    
    def _process_gsc_file(self, file) -> pd.DataFrame:
        """Process Google Search Console export"""
        try:
            # Read CSV
            df = pd.read_csv(file)
            
            # Map columns
            df = self._map_columns(df, 'gsc_export')
            
            # Clean and aggregate
            df = self._clean_gsc_data(df)
            
            logger.info(f"Processed GSC data for {df['page'].nunique()} pages")
            return df
            
        except Exception as e:
            logger.error(f"Error processing GSC file: {str(e)}")
            raise
    
    def _map_columns(self, df: pd.DataFrame, file_type: str) -> pd.DataFrame:
        """Map columns based on file type"""
        mappings = self.column_mappings.get(file_type, {})
        
        for standard_name, possible_names in mappings.items():
            for col_name in possible_names:
                if col_name in df.columns:
                    df = df.rename(columns={col_name: standard_name})
                    break
        
        return df
    
    def _clean_links_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate links data"""
        # Remove null values
        df = df.dropna(subset=['source', 'destination'])
        
        # Standardize URLs
        df['source'] = df['source'].str.strip()
        df['destination'] = df['destination'].str.strip()
        
        # Remove self-links
        df = df[df['source'] != df['destination']]
        
        # Add type if missing
        if 'type' not in df.columns:
            df['type'] = 'internal'
        
        return df
    
    def _process_embeddings(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process embedding vectors"""
        if 'embedding' in df.columns:
            # Parse embedding strings if needed
            if isinstance(df['embedding'].iloc[0], str):
                df['embedding'] = df['embedding'].apply(self._parse_embedding)
        
        return df
    
    def _parse_embedding(self, embedding_str: str) -> np.ndarray:
        """Parse embedding string to numpy array"""
        try:
            # Handle different formats
            if embedding_str.startswith('['):
                return np.array(json.loads(embedding_str))
            else:
                return np.fromstring(embedding_str.strip('[]'), sep=',')
        except:
            return np.array([])
    
    def _clean_gsc_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and aggregate GSC data"""
        # Ensure numeric columns
        numeric_cols = ['clicks', 'impressions', 'position']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remove null values
        df = df.dropna(subset=['page'])
        
        # Standardize URLs
        df['page'] = df['page'].str.strip()
        
        return df
    
    def _combine_data(self, links_df: pd.DataFrame, 
                     embeddings_df: pd.DataFrame,
                     gsc_df: Optional[pd.DataFrame]) -> Dict[str, Any]:
        """Combine all processed data"""
        
        # Create page inventory
        all_pages = set()
        all_pages.update(links_df['source'].unique())
        all_pages.update(links_df['destination'].unique())
        all_pages.update(embeddings_df['url'].unique())
        
        if gsc_df is not None:
            all_pages.update(gsc_df['page'].unique())
        
        # Create links matrix
        links_matrix = self._create_links_matrix(links_df, list(all_pages))
        
        # Create embeddings lookup
        embeddings_lookup = dict(zip(embeddings_df['url'], embeddings_df['embedding']))
        
        # Process GSC metrics if available
        gsc_metrics = None
        if gsc_df is not None:
            gsc_metrics = self._aggregate_gsc_metrics(gsc_df)
        
        return {
            'total_pages': len(all_pages),
            'total_links': len(links_df),
            'pages': list(all_pages),
            'links_df': links_df,
            'links_matrix': links_matrix,
            'embeddings_df': embeddings_df,
            'embeddings_lookup': embeddings_lookup,
            'gsc_df': gsc_df,
            'gsc_metrics': gsc_metrics,
            'processing_time': 0  # Will be updated by caller
        }
    
    def _create_links_matrix(self, links_df: pd.DataFrame, 
                            pages: List[str]) -> pd.DataFrame:
        """Create links adjacency matrix"""
        # Initialize matrix
        n = len(pages)
        matrix = pd.DataFrame(0, index=pages, columns=pages)
        
        # Populate matrix
        for _, row in links_df.iterrows():
            if row['source'] in pages and row['destination'] in pages:
                matrix.loc[row['source'], row['destination']] = 1
        
        return matrix
    
    def _aggregate_gsc_metrics(self, gsc_df: pd.DataFrame) -> Dict[str, Any]:
        """Aggregate GSC metrics by page"""
        metrics = {}
        
        # Group by page
        page_groups = gsc_df.groupby('page')
        
        # Calculate metrics
        metrics['url_queries_count'] = page_groups['query'].nunique().to_dict()
        metrics['url_queries'] = page_groups['query'].apply(list).to_dict()
        metrics['url_clicks'] = page_groups['clicks'].sum().to_dict()
        metrics['url_impressions'] = page_groups['impressions'].sum().to_dict()
        
        # Calculate average position
        if 'position' in gsc_df.columns:
            metrics['url_avg_position'] = page_groups['position'].mean().to_dict()
        
        # Calculate CTR
        if 'ctr' in gsc_df.columns:
            metrics['url_avg_ctr'] = page_groups['ctr'].mean().to_dict()
        
        return metrics
    
    def validate_data(self, processed_data: Dict) -> Tuple[bool, List[str]]:
        """
        Validate processed data
        
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        # Check for minimum pages
        if processed_data['total_pages'] < 2:
            errors.append("At least 2 pages required for analysis")
        
        # Check for links
        if processed_data['total_links'] == 0:
            errors.append("No internal links found in the data")
        
        # Check for embeddings
        if not processed_data['embeddings_lookup']:
            errors.append("No embeddings found in the data")
        
        # Check embedding dimensions
        if processed_data['embeddings_lookup']:
            embedding_dims = [len(e) for e in processed_data['embeddings_lookup'].values() if len(e) > 0]
            if embedding_dims and len(set(embedding_dims)) > 1:
                errors.append("Inconsistent embedding dimensions detected")
        
        return len(errors) == 0, errors
    
    def get_page_metrics(self, url: str, processed_data: Dict) -> Dict:
        """Get all metrics for a specific page"""
        metrics = {
            'url': url,
            'has_embedding': url in processed_data['embeddings_lookup'],
            'outbound_links': 0,
            'inbound_links': 0
        }
        
        # Count links
        if url in processed_data['links_matrix'].index:
            metrics['outbound_links'] = processed_data['links_matrix'].loc[url].sum()
            metrics['inbound_links'] = processed_data['links_matrix'][url].sum()
        
        # Add GSC metrics if available
        if processed_data['gsc_metrics']:
            for metric_name, metric_dict in processed_data['gsc_metrics'].items():
                if url in metric_dict:
                    metrics[metric_name] = metric_dict[url]
        
        return metrics
    
    def export_processed_data(self, processed_data: Dict, 
                            output_path: str) -> None:
        """Export processed data to file"""
        try:
            # Create summary DataFrame
            summary_data = []
            for page in processed_data['pages']:
                metrics = self.get_page_metrics(page, processed_data)
                summary_data.append(metrics)
            
            summary_df = pd.DataFrame(summary_data)
            
            # Export to Excel with multiple sheets
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
                processed_data['links_df'].to_excel(writer, sheet_name='Links', index=False)
                processed_data['embeddings_df'].to_excel(writer, sheet_name='Embeddings', index=False)
                
                if processed_data['gsc_df'] is not None:
                    processed_data['gsc_df'].to_excel(writer, sheet_name='GSC Data', index=False)
            
            logger.info(f"Processed data exported to {output_path}")
            
        except Exception as e:
            logger.error(f"Error exporting data: {str(e)}")
            raise