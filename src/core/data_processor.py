"""
Enhanced Data Processor for handling multiple file uploads and data integration
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from io import StringIO, BytesIO
import json
import traceback

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
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
    
    def _process_links_file(self, file) -> pd.DataFrame:
        """Process Screaming Frog internal links export"""
        try:
            # Read CSV
            df = pd.read_csv(file)
            
            logger.info(f"Links file shape: {df.shape}")
            logger.info(f"Links file columns: {df.columns.tolist()}")
            
            # Map columns
            df = self._map_columns(df, 'screaming_frog_links')
            
            # Clean and validate
            df = self._clean_links_data(df)
            
            # Filter internal links only if type column exists
            if 'type' in df.columns:
                # Handle case where type might be empty
                df['type'] = df['type'].fillna('internal')
                df = df[df['type'].str.lower().str.contains('internal', na=False)]
            
            logger.info(f"Processed {len(df)} internal links")
            return df
            
        except Exception as e:
            logger.error(f"Error processing links file: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
    
    def _process_embeddings_file(self, file) -> pd.DataFrame:
        """Process Screaming Frog embeddings export"""
        try:
            # Read CSV
            df = pd.read_csv(file)
            
            # Log details for debugging
            logger.info(f"Embeddings file shape: {df.shape}")
            logger.info(f"Embeddings file columns: {df.columns.tolist()}")
            logger.info(f"First few column names: {df.columns[:5].tolist()}")
            
            # Print sample of first row to understand data structure
            if len(df) > 0:
                logger.info("First row sample:")
                for col in df.columns[:3]:  # Just first 3 columns for brevity
                    sample_val = str(df[col].iloc[0])[:100]  # First 100 chars
                    logger.info(f"  {col}: {sample_val}...")
            
            # Initialize with original DataFrame
            result_df = df.copy()
            
            # Try to identify URL column
            url_col = None
            url_candidates = ['Address', 'URL', 'Page URL', 'url', 'address', 'page_url', 'Page']
            
            # First try exact match (case-insensitive)
            for col in df.columns:
                if any(col.lower() == candidate.lower() for candidate in url_candidates):
                    url_col = col
                    logger.info(f"Found URL column (exact match): {url_col}")
                    break
            
            # If not found, try contains match
            if not url_col:
                for col in df.columns:
                    if any(candidate.lower() in col.lower() for candidate in ['address', 'url', 'page']):
                        url_col = col
                        logger.info(f"Found URL column (contains match): {url_col}")
                        break
            
            # If still not found, use first column
            if not url_col:
                url_col = df.columns[0]
                logger.warning(f"No URL column found, using first column: {url_col}")
            
            # Try to identify embedding column
            embedding_col = None
            
            # Check for embedding/vector in column names
            for col in df.columns:
                col_lower = col.lower()
                if 'embedding' in col_lower or 'vector' in col_lower:
                    embedding_col = col
                    logger.info(f"Found embedding column by name: {embedding_col}")
                    break
            
            # If not found by name, look for array-like content
            if not embedding_col:
                for col in df.columns:
                    if col == url_col:  # Skip URL column
                        continue
                    
                    # Check first non-null value
                    non_null_vals = df[col].dropna()
                    if len(non_null_vals) > 0:
                        sample = str(non_null_vals.iloc[0])
                        # Check if it looks like an array or list of numbers
                        if (sample.startswith('[') and sample.endswith(']')) or \
                           (sample.startswith('(') and sample.endswith(')')) or \
                           (',' in sample and any(char.isdigit() or char == '.' or char == '-' for char in sample)):
                            embedding_col = col
                            logger.info(f"Found embedding column by content pattern: {embedding_col}")
                            break
            
            # Last resort - find any column with numeric-looking data that's not URL
            if not embedding_col:
                for col in df.columns:
                    if col == url_col:
                        continue
                    non_null_vals = df[col].dropna()
                    if len(non_null_vals) > 0:
                        sample = str(non_null_vals.iloc[0])
                        if any(char.isdigit() for char in sample):
                            embedding_col = col
                            logger.warning(f"Using column '{col}' as embedding (contains numbers)")
                            break
            
            if not embedding_col:
                # List all columns and their first values for debugging
                logger.error("Could not identify embedding column. Column analysis:")
                for col in df.columns:
                    if len(df) > 0:
                        sample = str(df[col].iloc[0])[:50]
                        logger.error(f"  {col}: {sample}")
                raise ValueError(f"Could not find embedding column in: {df.columns.tolist()}")
            
            # Create new DataFrame with standardized column names
            result_df = pd.DataFrame()
            result_df['url'] = df[url_col]
            result_df['embedding'] = df[embedding_col]
            
            # Add other useful columns if they exist
            for orig_col in ['Title', 'title', 'Page Title']:
                if orig_col in df.columns:
                    result_df['title'] = df[orig_col]
                    break
            
            for orig_col in ['Word Count', 'word_count', 'Words']:
                if orig_col in df.columns:
                    result_df['word_count'] = df[orig_col]
                    break
            
            logger.info(f"Created result DataFrame with columns: {result_df.columns.tolist()}")
            
            # Process embeddings
            result_df = self._process_embeddings(result_df)
            
            logger.info(f"Processed embeddings for {len(result_df)} pages")
            return result_df
            
        except Exception as e:
            logger.error(f"Error processing embeddings file: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            if 'df' in locals():
                logger.error(f"Available columns were: {df.columns.tolist()}")
            raise ValueError(f"Failed to process embeddings file: {str(e)}")
    
    def _process_gsc_file(self, file) -> pd.DataFrame:
        """Process Google Search Console export"""
        try:
            # Read CSV
            df = pd.read_csv(file)
            
            logger.info(f"GSC file shape: {df.shape}")
            logger.info(f"GSC file columns: {df.columns.tolist()}")
            
            # Map columns
            df = self._map_columns(df, 'gsc_export')
            
            # Clean and aggregate
            df = self._clean_gsc_data(df)
            
            if 'page' in df.columns:
                logger.info(f"Processed GSC data for {df['page'].nunique()} pages")
            else:
                logger.warning("No 'page' column found in GSC data")
            
            return df
            
        except Exception as e:
            logger.error(f"Error processing GSC file: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
    
    def _map_columns(self, df: pd.DataFrame, file_type: str) -> pd.DataFrame:
        """Map columns based on file type"""
        mappings = self.column_mappings.get(file_type, {})
        
        for standard_name, possible_names in mappings.items():
            column_found = False
            for col_name in possible_names:
                # Case-insensitive matching
                for actual_col in df.columns:
                    if col_name.lower() == actual_col.lower():
                        df = df.rename(columns={actual_col: standard_name})
                        column_found = True
                        logger.info(f"Mapped column '{actual_col}' to '{standard_name}'")
                        break
                if column_found:
                    break
        
        return df
    
    def _clean_links_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate links data"""
        # Check if required columns exist
        if 'source' not in df.columns or 'destination' not in df.columns:
            logger.warning("Source or destination columns not found, attempting to identify...")
            
            # Look for columns that might be source/destination
            for col in df.columns:
                col_lower = col.lower()
                if 'from' in col_lower or 'source' in col_lower:
                    df = df.rename(columns={col: 'source'})
                    logger.info(f"Mapped '{col}' to 'source'")
                elif 'to' in col_lower or 'dest' in col_lower or 'target' in col_lower:
                    df = df.rename(columns={col: 'destination'})
                    logger.info(f"Mapped '{col}' to 'destination'")
        
        # Check again if we have required columns
        if 'source' not in df.columns or 'destination' not in df.columns:
            raise ValueError(f"Could not find source/destination columns. Available: {df.columns.tolist()}")
        
        # Remove null values
        initial_count = len(df)
        df = df.dropna(subset=['source', 'destination'])
        if len(df) < initial_count:
            logger.info(f"Removed {initial_count - len(df)} rows with null source/destination")
        
        # Standardize URLs
        df['source'] = df['source'].astype(str).str.strip()
        df['destination'] = df['destination'].astype(str).str.strip()
        
        # Remove self-links
        self_links = df[df['source'] == df['destination']]
        if len(self_links) > 0:
            df = df[df['source'] != df['destination']]
            logger.info(f"Removed {len(self_links)} self-links")
        
        # Add type if missing
        if 'type' not in df.columns:
            df['type'] = 'internal'
            logger.info("Added 'type' column with default value 'internal'")
        
        return df
    
    def _process_embeddings(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process embedding vectors"""
        if 'embedding' not in df.columns:
            logger.error(f"No 'embedding' column found. Available columns: {df.columns.tolist()}")
            return df
        
        logger.info(f"Processing {len(df)} embeddings")
        
        # Check what type of data we have
        if len(df) > 0:
            sample = df['embedding'].iloc[0]
            logger.info(f"Embedding sample type: {type(sample)}")
            logger.info(f"Embedding sample (first 100 chars): {str(sample)[:100]}")
            
            # Only parse if it's a string
            if isinstance(sample, str) or pd.isna(sample):
                logger.info("Embeddings are strings, parsing...")
                df['embedding'] = df['embedding'].apply(self._parse_embedding)
            elif isinstance(sample, (list, np.ndarray)):
                logger.info("Embeddings are already arrays")
                # Convert lists to numpy arrays
                df['embedding'] = df['embedding'].apply(lambda x: np.array(x) if isinstance(x, list) else x)
            else:
                logger.warning(f"Unknown embedding type: {type(sample)}")
                df['embedding'] = df['embedding'].apply(self._parse_embedding)
        
        # Check embedding dimensions
        valid_embeddings = df['embedding'].apply(lambda x: len(x) > 0 if isinstance(x, np.ndarray) else False)
        logger.info(f"Valid embeddings: {valid_embeddings.sum()} out of {len(df)}")
        
        return df
    
    def _parse_embedding(self, embedding_str) -> np.ndarray:
        """Parse embedding string to numpy array"""
        try:
            if pd.isna(embedding_str) or embedding_str == '' or embedding_str is None:
                return np.array([])
            
            embedding_str = str(embedding_str).strip()
            
            # Handle empty strings
            if not embedding_str:
                return np.array([])
            
            # Handle different formats
            if embedding_str.startswith('[') and embedding_str.endswith(']'):
                # JSON array format
                try:
                    return np.array(json.loads(embedding_str))
                except json.JSONDecodeError:
                    # Try parsing as comma-separated within brackets
                    inner = embedding_str[1:-1]
                    if inner:
                        return np.fromstring(inner, sep=',')
                    return np.array([])
            elif embedding_str.startswith('(') and embedding_str.endswith(')'):
                # Tuple format
                inner = embedding_str[1:-1]
                if inner:
                    return np.fromstring(inner, sep=',')
                return np.array([])
            elif ',' in embedding_str:
                # Comma-separated format
                return np.fromstring(embedding_str, sep=',')
            elif ' ' in embedding_str and any(char.isdigit() for char in embedding_str):
                # Space-separated format
                return np.fromstring(embedding_str, sep=' ')
            else:
                # Single value or unknown format
                try:
                    return np.array([float(embedding_str)])
                except:
                    logger.warning(f"Could not parse embedding: {embedding_str[:50]}")
                    return np.array([])
                    
        except Exception as e:
            logger.warning(f"Failed to parse embedding: {str(e)}")
            return np.array([])
    
    def _clean_gsc_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and aggregate GSC data"""
        # Ensure numeric columns are numeric
        numeric_cols = ['clicks', 'impressions', 'position', 'ctr']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remove null values from page column if it exists
        if 'page' in df.columns:
            initial_count = len(df)
            df = df.dropna(subset=['page'])
            if len(df) < initial_count:
                logger.info(f"Removed {initial_count - len(df)} rows with null page values")
            
            # Standardize URLs
            df['page'] = df['page'].astype(str).str.strip()
        
        # Fill missing values
        fill_values = {
            'clicks': 0,
            'impressions': 0,
            'position': 0,
            'ctr': 0,
            'query': 'Not Available'
        }
        
        for col, val in fill_values.items():
            if col in df.columns:
                df[col] = df[col].fillna(val)
        
        return df
    
    def _combine_data(self, links_df: pd.DataFrame, 
                     embeddings_df: pd.DataFrame,
                     gsc_df: Optional[pd.DataFrame]) -> Dict[str, Any]:
        """Combine all processed data"""
        
        # Create page inventory
        all_pages = set()
        
        if 'source' in links_df.columns:
            all_pages.update(links_df['source'].unique())
        if 'destination' in links_df.columns:
            all_pages.update(links_df['destination'].unique())
        if 'url' in embeddings_df.columns:
            all_pages.update(embeddings_df['url'].unique())
        
        if gsc_df is not None and 'page' in gsc_df.columns:
            all_pages.update(gsc_df['page'].unique())
        
        logger.info(f"Total unique pages found: {len(all_pages)}")
        
        # Create links matrix
        links_matrix = self._create_links_matrix(links_df, list(all_pages))
        
        # Create embeddings lookup
        embeddings_lookup = {}
        if 'url' in embeddings_df.columns and 'embedding' in embeddings_df.columns:
            for _, row in embeddings_df.iterrows():
                url = row['url']
                embedding = row['embedding']
                if isinstance(embedding, np.ndarray) and len(embedding) > 0:
                    embeddings_lookup[url] = embedding
        
        logger.info(f"Created embeddings lookup for {len(embeddings_lookup)} pages")
        
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
        
        # Populate matrix if columns exist
        if 'source' in links_df.columns and 'destination' in links_df.columns:
            for _, row in links_df.iterrows():
                source = row['source']
                dest = row['destination']
                if source in pages and dest in pages:
                    matrix.loc[source, dest] = 1
        
        return matrix
    
    def _aggregate_gsc_metrics(self, gsc_df: pd.DataFrame) -> Dict[str, Any]:
        """Aggregate GSC metrics by page"""
        metrics = {}
        
        if 'page' not in gsc_df.columns:
            logger.warning("No 'page' column in GSC data")
            return metrics
        
        # Group by page
        page_groups = gsc_df.groupby('page')
        
        # Calculate metrics
        if 'query' in gsc_df.columns:
            metrics['url_queries_count'] = page_groups['query'].nunique().to_dict()
            metrics['url_queries'] = page_groups['query'].apply(list).to_dict()
        
        if 'clicks' in gsc_df.columns:
            metrics['url_clicks'] = page_groups['clicks'].sum().to_dict()
        
        if 'impressions' in gsc_df.columns:
            metrics['url_impressions'] = page_groups['impressions'].sum().to_dict()
        
        if 'position' in gsc_df.columns:
            metrics['url_avg_position'] = page_groups['position'].mean().to_dict()
        
        if 'ctr' in gsc_df.columns:
            metrics['url_avg_ctr'] = page_groups['ctr'].mean().to_dict()
        
        logger.info(f"Aggregated GSC metrics for {len(page_groups)} pages")
        
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
            errors.append("No valid embeddings found in the data")
        
        # Check embedding dimensions
        if processed_data['embeddings_lookup']:
            embedding_dims = [len(e) for e in processed_data['embeddings_lookup'].values() if len(e) > 0]
            if embedding_dims:
                unique_dims = set(embedding_dims)
                if len(unique_dims) > 1:
                    logger.warning(f"Multiple embedding dimensions found: {unique_dims}")
                logger.info(f"Embedding dimensions: {unique_dims}")
        
        return len(errors) == 0, errors
    
    def get_page_metrics(self, url: str, processed_data: Dict) -> Dict:
        """Get all metrics for a specific page"""
        metrics = {
            'url': url,
            'has_embedding': url in processed_data.get('embeddings_lookup', {}),
            'outbound_links': 0,
            'inbound_links': 0
        }
        
        # Count links
        links_matrix = processed_data.get('links_matrix')
        if links_matrix is not None and url in links_matrix.index:
            metrics['outbound_links'] = int(links_matrix.loc[url].sum())
            metrics['inbound_links'] = int(links_matrix[url].sum())
        
        # Add GSC metrics if available
        if processed_data.get('gsc_metrics'):
            for metric_name, metric_dict in processed_data['gsc_metrics'].items():
                if url in metric_dict:
                    metrics[metric_name] = metric_dict[url]
        
        return metrics
