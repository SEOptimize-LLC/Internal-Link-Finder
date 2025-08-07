"""
DataForSEO API client for fetching monthly search volume data
"""

import requests
import base64
import json
import time
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta
import pandas as pd

from config import config

logger = logging.getLogger(__name__)

class DataForSEOClient:
    """Client for DataForSEO API interactions"""
    
    def __init__(self, login: str, password: str):
        """
        Initialize DataForSEO client
        
        Args:
            login: DataForSEO login
            password: DataForSEO password
        """
        self.login = login
        self.password = password
        self.base_url = config.DATAFORSEO_API_URL
        self.auth_header = self._create_auth_header()
        self.rate_limit_delay = 1.0 / config.DATAFORSEO_RATE_LIMIT
        self.last_request_time = 0
    
    def _create_auth_header(self) -> str:
        """Create base64 encoded auth header"""
        credentials = f"{self.login}:{self.password}"
        encoded = base64.b64encode(credentials.encode()).decode()
        return f"Basic {encoded}"
    
    def _rate_limit(self):
        """Implement rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - time_since_last)
        
        self.last_request_time = time.time()
    
    def get_monthly_search_volume(self, keywords: List[str], 
                                 location: str = "United States",
                                 language: str = "en") -> Dict[str, int]:
        """
        Get monthly search volume for keywords
        
        Args:
            keywords: List of keywords to check
            location: Target location
            language: Target language
            
        Returns:
            Dictionary mapping keywords to monthly search volumes
        """
        try:
            # Prepare request data
            post_data = [{
                "keywords": keywords,
                "location_name": location,
                "language_code": language
            }]
            
            # Make API request
            response = self._make_request(post_data)
            
            # Parse response
            search_volumes = self._parse_search_volume_response(response)
            
            return search_volumes
            
        except Exception as e:
            logger.error(f"Error fetching search volumes: {str(e)}")
            return {}
    
    def get_bulk_search_volumes(self, keywords: List[str],
                               location: str = "United States",
                               batch_size: int = 1000) -> Dict[str, int]:
        """
        Get search volumes for large keyword lists in batches
        
        Args:
            keywords: List of keywords
            location: Target location
            batch_size: Number of keywords per API call
            
        Returns:
            Dictionary mapping keywords to search volumes
        """
        all_volumes = {}
        
        # Process in batches
        for i in range(0, len(keywords), batch_size):
            batch = keywords[i:i + batch_size]
            
            logger.info(f"Fetching search volumes for batch {i // batch_size + 1}")
            
            try:
                volumes = self.get_monthly_search_volume(batch, location)
                all_volumes.update(volumes)
                
            except Exception as e:
                logger.error(f"Error in batch {i // batch_size + 1}: {str(e)}")
                continue
        
        return all_volumes
    
    def _make_request(self, post_data: List[Dict]) -> Dict:
        """Make API request to DataForSEO"""
        
        # Apply rate limiting
        self._rate_limit()
        
        headers = {
            'Authorization': self.auth_header,
            'Content-Type': 'application/json'
        }
        
        try:
            response = requests.post(
                self.base_url,
                headers=headers,
                json=post_data,
                timeout=config.REQUEST_TIMEOUT
            )
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {str(e)}")
            raise
    
    def _parse_search_volume_response(self, response: Dict) -> Dict[str, int]:
        """Parse search volume response from DataForSEO"""
        
        search_volumes = {}
        
        try:
            # Navigate response structure
            if 'tasks' in response and response['tasks']:
                task = response['tasks'][0]
                
                if 'result' in task and task['result']:
                    result = task['result'][0]
                    
                    if 'items' in result:
                        for item in result['items']:
                            keyword = item.get('keyword', '')
                            
                            # Get search volume from the last 30 days
                            if 'keyword_info' in item:
                                keyword_info = item['keyword_info']
                                
                                # Try different possible fields
                                volume = 0
                                
                                if 'search_volume' in keyword_info:
                                    volume = keyword_info['search_volume']
                                elif 'monthly_searches' in keyword_info:
                                    # Get most recent month
                                    monthly = keyword_info['monthly_searches']
                                    if monthly and len(monthly) > 0:
                                        volume = monthly[-1].get('search_volume', 0)
                                
                                search_volumes[keyword] = volume
            
            return search_volumes
            
        except (KeyError, IndexError, TypeError) as e:
            logger.error(f"Error parsing API response: {str(e)}")
            return {}
    
    def get_keyword_data(self, keywords: List[str],
                        location: str = "United States") -> pd.DataFrame:
        """
        Get comprehensive keyword data
        
        Args:
            keywords: List of keywords
            location: Target location
            
        Returns:
            DataFrame with keyword metrics
        """
        try:
            # Prepare request for keyword data endpoint
            post_data = [{
                "keywords": keywords,
                "location_name": location,
                "language_code": "en",
                "include_clickstream_data": True,
                "include_seed_keyword": True,
                "include_serp_info": False
            }]
            
            # Make request
            response = self._make_request(post_data)
            
            # Parse to DataFrame
            df = self._parse_keyword_data_response(response)
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching keyword data: {str(e)}")
            return pd.DataFrame()
    
    def _parse_keyword_data_response(self, response: Dict) -> pd.DataFrame:
        """Parse comprehensive keyword data response"""
        
        data = []
        
        try:
            if 'tasks' in response and response['tasks']:
                task = response['tasks'][0]
                
                if 'result' in task and task['result']:
                    result = task['result'][0]
                    
                    if 'items' in result:
                        for item in result['items']:
                            keyword_data = {
                                'keyword': item.get('keyword', ''),
                                'search_volume': 0,
                                'competition': 0,
                                'cpc': 0,
                                'trend': []
                            }
                            
                            if 'keyword_info' in item:
                                info = item['keyword_info']
                                keyword_data['search_volume'] = info.get('search_volume', 0)
                                keyword_data['competition'] = info.get('competition', 0)
                                keyword_data['cpc'] = info.get('cpc', 0)
                                
                                # Get trend data if available
                                if 'monthly_searches' in info:
                                    keyword_data['trend'] = [
                                        m.get('search_volume', 0) 
                                        for m in info['monthly_searches']
                                    ]
                            
                            data.append(keyword_data)
            
            return pd.DataFrame(data)
            
        except Exception as e:
            logger.error(f"Error parsing keyword data: {str(e)}")
            return pd.DataFrame()
    
    def test_connection(self) -> bool:
        """Test API connection and credentials"""
        try:
            # Test with a simple request
            test_keywords = ["test"]
            response = self.get_monthly_search_volume(test_keywords)
            
            return len(response) > 0
            
        except Exception as e:
            logger.error(f"Connection test failed: {str(e)}")
            return False
    
    def get_remaining_credits(self) -> Optional[Dict]:
        """Get remaining API credits"""
        try:
            # Make request to user info endpoint
            url = "https://api.dataforseo.com/v3/appendix/user_data"
            
            headers = {
                'Authorization': self.auth_header,
                'Content-Type': 'application/json'
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            if 'tasks' in data and data['tasks']:
                task = data['tasks'][0]
                if 'result' in task and task['result']:
                    result = task['result'][0]
                    return {
                        'balance': result.get('money', {}).get('balance', 0),
                        'currency': result.get('money', {}).get('currency', 'USD'),
                        'total_api_calls': result.get('rates', {}).get('requests', {}).get('total', 0)
                    }
            
            return None
            
        except Exception as e:
            logger.error(f"Error fetching credits: {str(e)}")
            return None
    
    def enhance_gsc_queries_with_volume(self, gsc_queries: Dict[str, List[str]],
                                       location: str = "United States") -> Dict[str, Dict]:
        """
        Enhance GSC queries with search volume data
        
        Args:
            gsc_queries: Dictionary mapping URLs to their queries
            location: Target location
            
        Returns:
            Enhanced dictionary with search volumes
        """
        # Collect all unique queries
        all_queries = set()
        for queries in gsc_queries.values():
            all_queries.update(queries)
        
        logger.info(f"Fetching search volumes for {len(all_queries)} unique queries")
        
        # Get search volumes
        search_volumes = self.get_bulk_search_volumes(list(all_queries), location)
        
        # Enhance the data
        enhanced_data = {}
        
        for url, queries in gsc_queries.items():
            enhanced_data[url] = {
                'queries': queries,
                'search_volumes': {q: search_volumes.get(q, 0) for q in queries},
                'total_search_volume': sum(search_volumes.get(q, 0) for q in queries),
                'avg_search_volume': sum(search_volumes.get(q, 0) for q in queries) / len(queries) if queries else 0
            }
        

        return enhanced_data
