"""
Google Search Console data processor for performance metrics integration
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple  # Added Tuple if needed
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class GSCDataProcessor:
    """Process Google Search Console export data"""
    
    def __init__(self):
        """Initialize GSC processor"""
        self.metrics_cache = {}
    
    def process_gsc_export(self, file) -> Dict:
        """
        Process GSC CSV export file
        
        Args:
            file: Uploaded GSC CSV file
            
        Returns:
            Dictionary with processed GSC metrics
        """
        try:
            logger.info("Processing GSC export file")
            
            # Read CSV
            df = pd.read_csv(file)
            
            # Identify and map columns
            df = self._identify_gsc_columns(df)
            
            # Clean and validate data
            df = self._clean_gsc_data(df)
            
            # Calculate URL-level metrics
            url_metrics = self.calculate_url_metrics(df)
            
            # Calculate query-level metrics
            query_metrics = self._calculate_query_metrics(df)
            
            # Store in cache
            self.metrics_cache = {
                'raw_data': df,
                'url_metrics': url_metrics,
                'query_metrics': query_metrics,
                'total_clicks': df['clicks'].sum(),
                'total_impressions': df['impressions'].sum(),
                'unique_queries': df['query'].nunique(),
                'unique_pages': df['page'].nunique()
            }
            
            logger.info(f"Processed GSC data: {self.metrics_cache['unique_pages']} pages, {self.metrics_cache['unique_queries']} queries")
            
            return url_metrics
            
        except Exception as e:
            logger.error(f"Error processing GSC export: {str(e)}")
            raise
    
    def _identify_gsc_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Identify and standardize GSC column names"""
        
        # Common GSC column variations
        column_mappings = {
            'Top pages': 'page',
            'Pages': 'page',
            'Page': 'page',
            'Landing Page': 'page',
            'URL': 'page',
            'Top queries': 'query',
            'Queries': 'query',
            'Query': 'query',
            'Search Query': 'query',
            'Keyword': 'query',
            'Clicks': 'clicks',
            'Impressions': 'impressions',
            'CTR': 'ctr',
            'Click Through Rate': 'ctr',
            'Position': 'position',
            'Average Position': 'position',
            'Avg. position': 'position'
        }
        
        # Rename columns
        for old_name, new_name in column_mappings.items():
            if old_name in df.columns:
                df = df.rename(columns={old_name: new_name})
        
        # Validate required columns
        required_columns = ['page', 'clicks', 'impressions']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            # Try to infer from data structure
            df = self._infer_columns(df)
        
        return df
    
    def _infer_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Try to infer column meanings from data"""
        
        # Look for URL-like columns
        for col in df.columns:
            sample = df[col].astype(str).iloc[0] if len(df) > 0 else ''
            if sample.startswith('http') or sample.startswith('/'):
                df = df.rename(columns={col: 'page'})
                break
        
        # Look for numeric columns that could be clicks/impressions
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) >= 2:
            # Assume larger values are impressions
            col_sums = {col: df[col].sum() for col in numeric_cols}
            sorted_cols = sorted(col_sums.items(), key=lambda x: x[1], reverse=True)
            
            if 'impressions' not in df.columns and len(sorted_cols) > 0:
                df = df.rename(columns={sorted_cols[0][0]: 'impressions'})
            
            if 'clicks' not in df.columns and len(sorted_cols) > 1:
                df = df.rename(columns={sorted_cols[1][0]: 'clicks'})
        
        return df
    
    def _clean_gsc_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize GSC data"""
        
        # Remove rows with missing critical data
        df = df.dropna(subset=['page'])
        
        # Ensure numeric columns are numeric
        numeric_columns = ['clicks', 'impressions', 'position', 'ctr']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Convert CTR percentage if needed
        if 'ctr' in df.columns:
            if df['ctr'].max() > 1:
                df['ctr'] = df['ctr'] / 100
        
        # Clean URLs
        if 'page' in df.columns:
            df['page'] = df['page'].str.strip()
            # Ensure full URLs
            df['page'] = df['page'].apply(self._ensure_full_url)
        
        # Add query column if missing
        if 'query' not in df.columns:
            df['query'] = 'Not Available'
        
        # Fill missing values
        df = df.fillna({
            'clicks': 0,
            'impressions': 0,
            'position': 0,
            'ctr': 0,
            'query': 'Not Available'
        })
        
        return df
    
    def _ensure_full_url(self, url: str) -> str:
        """Ensure URL is complete"""
        if pd.isna(url):
            return ''
        
        url = str(url).strip()
        
        # If it's a relative URL, we'll keep it as is
        # In production, you might want to prepend the domain
        return url
    
    def calculate_url_metrics(self, gsc_data: pd.DataFrame) -> Dict:
        """Calculate per-URL performance metrics"""
        
        metrics = {}
        
        # Group by page
        if 'page' in gsc_data.columns:
            page_groups = gsc_data.groupby('page')
            
            # Number of queries per URL
            metrics['url_queries_count'] = page_groups['query'].nunique().to_dict()
            
            # List of queries per URL
            metrics['url_queries'] = page_groups['query'].apply(
                lambda x: list(x.unique())
            ).to_dict()
            
            # Total clicks per URL
            metrics['url_clicks'] = page_groups['clicks'].sum().to_dict()
            
            # Total impressions per URL
            metrics['url_impressions'] = page_groups['impressions'].sum().to_dict()
            
            # Average position per URL
            if 'position' in gsc_data.columns:
                # Weighted average position by impressions
                def weighted_avg_position(group):
                    if group['impressions'].sum() > 0:
                        return (group['position'] * group['impressions']).sum() / group['impressions'].sum()
                    return 0
                
                metrics['url_avg_position'] = page_groups.apply(weighted_avg_position).to_dict()
            
            # Average CTR per URL
            if 'ctr' in gsc_data.columns:
                # Calculate CTR from clicks and impressions
                url_ctr = {}
                for url in metrics['url_clicks']:
                    if metrics['url_impressions'].get(url, 0) > 0:
                        url_ctr[url] = metrics['url_clicks'][url] / metrics['url_impressions'][url]
                    else:
                        url_ctr[url] = 0
                metrics['url_ctr'] = url_ctr
            
            # Top queries per URL (sorted by clicks)
            metrics['url_top_queries'] = page_groups.apply(
                lambda x: x.nlargest(5, 'clicks')['query'].tolist()
            ).to_dict()
            
            # Query diversity score (number of unique queries / total queries)
            metrics['url_query_diversity'] = page_groups.apply(
                lambda x: x['query'].nunique() / len(x) if len(x) > 0 else 0
            ).to_dict()
        
        return metrics
    
    def _calculate_query_metrics(self, gsc_data: pd.DataFrame) -> Dict:
        """Calculate query-level metrics"""
        
        metrics = {}
        
        if 'query' in gsc_data.columns:
            query_groups = gsc_data.groupby('query')
            
            # Total clicks per query
            metrics['query_clicks'] = query_groups['clicks'].sum().to_dict()
            
            # Total impressions per query
            metrics['query_impressions'] = query_groups['impressions'].sum().to_dict()
            
            # Pages ranking for each query
            metrics['query_pages'] = query_groups['page'].apply(list).to_dict()
            
            # Average position per query
            if 'position' in gsc_data.columns:
                metrics['query_avg_position'] = query_groups['position'].mean().to_dict()
            
            # Top queries by clicks
            top_queries = gsc_data.groupby('query')['clicks'].sum().nlargest(100)
            metrics['top_queries'] = top_queries.to_dict()
        
        return metrics
    
    def get_url_performance(self, url: str) -> Dict:
        """Get performance metrics for a specific URL"""
        
        if not self.metrics_cache or 'url_metrics' not in self.metrics_cache:
            return {}
        
        url_metrics = self.metrics_cache['url_metrics']
        
        performance = {
            'queries_count': url_metrics.get('url_queries_count', {}).get(url, 0),
            'queries': url_metrics.get('url_queries', {}).get(url, []),
            'clicks': url_metrics.get('url_clicks', {}).get(url, 0),
            'impressions': url_metrics.get('url_impressions', {}).get(url, 0),
            'avg_position': url_metrics.get('url_avg_position', {}).get(url, 0),
            'ctr': url_metrics.get('url_ctr', {}).get(url, 0),
            'top_queries': url_metrics.get('url_top_queries', {}).get(url, []),
            'query_diversity': url_metrics.get('url_query_diversity', {}).get(url, 0)
        }
        
        return performance
    
    def get_query_urls(self, query: str) -> List[str]:
        """Get URLs ranking for a specific query"""
        
        if not self.metrics_cache or 'query_metrics' not in self.metrics_cache:
            return []
        
        query_metrics = self.metrics_cache['query_metrics']
        return query_metrics.get('query_pages', {}).get(query, [])
    
    def calculate_opportunity_score(self, url: str) -> float:
        """Calculate opportunity score based on GSC metrics"""
        
        performance = self.get_url_performance(url)
        
        if not performance:
            return 0.0
        
        # Scoring factors
        score = 0.0
        
        # High impressions but low CTR = opportunity
        if performance['impressions'] > 0:
            ctr_opportunity = (1 - performance['ctr']) * min(performance['impressions'] / 1000, 1)
            score += ctr_opportunity * 0.3
        
        # Good position but low clicks = opportunity
        if 0 < performance['avg_position'] <= 10:
            position_opportunity = (11 - performance['avg_position']) / 10
            click_rate = performance['clicks'] / max(performance['impressions'], 1)
            score += position_opportunity * (1 - click_rate) * 0.3
        
        # Query diversity = more opportunities
        score += performance['query_diversity'] * 0.2
        
        # Number of queries = reach
        query_score = min(performance['queries_count'] / 100, 1)
        score += query_score * 0.2
        
        return min(score, 1.0)
    
    def export_gsc_summary(self, output_path: str) -> None:
        """Export GSC summary to file"""
        
        if not self.metrics_cache:
            logger.warning("No GSC data to export")
            return
        
        try:
            # Create summary DataFrame
            url_metrics = self.metrics_cache['url_metrics']
            
            summary_data = []
            for url in url_metrics.get('url_clicks', {}).keys():
                summary_data.append({
                    'URL': url,
                    'Queries': url_metrics['url_queries_count'].get(url, 0),
                    'Clicks': url_metrics['url_clicks'].get(url, 0),
                    'Impressions': url_metrics['url_impressions'].get(url, 0),
                    'CTR': f"{url_metrics['url_ctr'].get(url, 0):.2%}",
                    'Avg Position': f"{url_metrics['url_avg_position'].get(url, 0):.1f}",
                    'Top Queries': ', '.join(url_metrics['url_top_queries'].get(url, [])[:3]),
                    'Opportunity Score': f"{self.calculate_opportunity_score(url):.2f}"
                })
            
            summary_df = pd.DataFrame(summary_data)
            summary_df = summary_df.sort_values('Clicks', ascending=False)
            
            # Export to CSV
            summary_df.to_csv(output_path, index=False)
            logger.info(f"GSC summary exported to {output_path}")
            
        except Exception as e:
            logger.error(f"Error exporting GSC summary: {str(e)}")

            raise
