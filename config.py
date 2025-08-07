"""
Configuration settings for Enhanced Internal Link Opportunity Finder
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

@dataclass
class AppConfig:
    """Main application configuration"""
    
    # Application settings
    APP_NAME: str = "Enhanced Internal Link Opportunity Finder"
    VERSION: str = "2.0.0"
    DEBUG_MODE: bool = False
    
    # Processing settings
    MAX_WORKERS: int = 4
    BATCH_SIZE: int = 100
    TIMEOUT_SECONDS: int = 30
    MAX_RETRIES: int = 3
    
    # Analysis settings
    DEFAULT_SIMILARITY_THRESHOLD: float = 0.7
    MAX_OPPORTUNITIES_PER_URL: int = 10
    MIN_CONTENT_LENGTH: int = 100
    
    # Content generation settings
    DEFAULT_SNIPPET_LENGTH: int = 4  # sentences
    MIN_SNIPPET_LENGTH: int = 3
    MAX_SNIPPET_LENGTH: int = 6
    ANCHOR_MIN_WORDS: int = 3
    ANCHOR_MAX_WORDS: int = 10
    
    # Quality thresholds
    MIN_STYLE_COHERENCE_SCORE: float = 0.85
    MIN_SEMANTIC_RELEVANCE_SCORE: float = 0.80
    MIN_FLOW_INTEGRATION_SCORE: float = 0.75
    MIN_FORMAT_COMPLIANCE_SCORE: float = 1.0  # 100% required
    
    # DataForSEO settings
    DATAFORSEO_API_URL: str = "https://api.dataforseo.com/v3/keywords_data/clickstream_data/dataforseo_search_volume/live"
    DATAFORSEO_LOCATION: str = "United States"
    DATAFORSEO_LANGUAGE: str = "en"
    DATAFORSEO_DEVICE: str = "desktop"
    DATAFORSEO_BATCH_SIZE: int = 100
    DATAFORSEO_RATE_LIMIT: int = 10  # requests per second
    
    # Scraping settings
    USER_AGENT: str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    REQUEST_TIMEOUT: int = 15
    MAX_REDIRECTS: int = 3
    RETRY_DELAY: int = 2
    
    # Content extraction settings
    BOILERPLATE_TAGS: List[str] = None
    CONTENT_SELECTORS: List[str] = None
    EXCLUDED_CLASSES: List[str] = None
    
    # Style analysis settings
    READABILITY_METRICS: List[str] = None
    LINGUISTIC_FEATURES: List[str] = None
    
    # Export settings
    EXPORT_DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"
    EXCEL_ENGINE: str = "openpyxl"
    CSV_ENCODING: str = "utf-8"
    
    # UI settings
    PAGE_SIZE: int = 25
    MAX_DISPLAY_ROWS: int = 1000
    CHART_HEIGHT: int = 400
    CHART_WIDTH: int = 600
    
    def __post_init__(self):
        """Initialize list fields"""
        if self.BOILERPLATE_TAGS is None:
            self.BOILERPLATE_TAGS = [
                'script', 'style', 'noscript', 'nav', 'footer', 
                'aside', 'header', 'menu', 'menubar'
            ]
        
        if self.CONTENT_SELECTORS is None:
            self.CONTENT_SELECTORS = [
                'article', 'main', '[role="main"]', '.content', 
                '#content', '.post', '.entry-content'
            ]
        
        if self.EXCLUDED_CLASSES is None:
            self.EXCLUDED_CLASSES = [
                'navigation', 'nav', 'menu', 'sidebar', 'footer',
                'header', 'advertisement', 'ad', 'banner', 'popup',
                'modal', 'cookie', 'social', 'share', 'comment'
            ]
        
        if self.READABILITY_METRICS is None:
            self.READABILITY_METRICS = [
                'flesch_reading_ease', 'flesch_kincaid_grade',
                'gunning_fog', 'automated_readability_index'
            ]
        
        if self.LINGUISTIC_FEATURES is None:
            self.LINGUISTIC_FEATURES = [
                'sentence_length', 'word_complexity', 'formality',
                'tone', 'transitions', 'vocabulary_level'
            ]

@dataclass
class StyleProfile:
    """Style profile for content matching"""
    avg_sentence_length: float
    complexity_score: float
    formality_level: str  # formal, semi-formal, informal
    vocabulary_level: str  # basic, intermediate, advanced, expert
    common_transitions: List[str]
    sentence_starters: List[str]
    tone_indicators: List[str]
    technical_density: float
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'avg_sentence_length': self.avg_sentence_length,
            'complexity_score': self.complexity_score,
            'formality_level': self.formality_level,
            'vocabulary_level': self.vocabulary_level,
            'common_transitions': self.common_transitions,
            'sentence_starters': self.sentence_starters,
            'tone_indicators': self.tone_indicators,
            'technical_density': self.technical_density
        }

@dataclass
class QualityScores:
    """Quality scores for content validation"""
    format_compliance: float
    style_coherence: float
    semantic_relevance: float
    flow_integration: float
    
    def meets_thresholds(self, config: AppConfig) -> bool:
        """Check if all scores meet minimum thresholds"""
        return (
            self.format_compliance >= config.MIN_FORMAT_COMPLIANCE_SCORE and
            self.style_coherence >= config.MIN_STYLE_COHERENCE_SCORE and
            self.semantic_relevance >= config.MIN_SEMANTIC_RELEVANCE_SCORE and
            self.flow_integration >= config.MIN_FLOW_INTEGRATION_SCORE
        )
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'format_compliance': self.format_compliance,
            'style_coherence': self.style_coherence,
            'semantic_relevance': self.semantic_relevance,
            'flow_integration': self.flow_integration
        }

@dataclass
class LinkSuggestion:
    """Internal link suggestion structure"""
    anchor_text: str
    placement_hint: str
    content_snippet: str
    quality_scores: QualityScores
    target_url: str
    destination_url: str
    
    def to_table_row(self) -> Dict:
        """Convert to table row format"""
        return {
            'Anchor Text': self.anchor_text,
            'Placement Hint': self.placement_hint,
            'Content Snippet': self.content_snippet
        }
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'anchor_text': self.anchor_text,
            'placement_hint': self.placement_hint,
            'content_snippet': self.content_snippet,
            'quality_scores': self.quality_scores.to_dict(),
            'target_url': self.target_url,
            'destination_url': self.destination_url
        }

# Error messages
ERROR_MESSAGES = {
    'file_not_found': "File not found. Please check the file path.",
    'invalid_format': "Invalid file format. Please upload a CSV file.",
    'missing_columns': "Required columns missing in the uploaded file.",
    'api_error': "API request failed. Please check your credentials.",
    'scraping_error': "Failed to scrape URL content. Please check the URL.",
    'generation_error': "Failed to generate content suggestions.",
    'quality_error': "Generated content did not meet quality thresholds.",
    'export_error': "Failed to export results. Please try again."
}

# Success messages
SUCCESS_MESSAGES = {
    'files_processed': "Files processed successfully!",
    'analysis_complete': "Analysis completed successfully!",
    'content_generated': "Content suggestions generated successfully!",
    'export_complete': "Results exported successfully!",
    'settings_saved': "Settings saved successfully!"
}

# Column mappings for different file types
COLUMN_MAPPINGS = {
    'screaming_frog_links': {
        'source': ['Source', 'Source URL', 'From'],
        'destination': ['Destination', 'Destination URL', 'To'],
        'anchor': ['Anchor Text', 'Anchor', 'Link Text'],
        'type': ['Type', 'Link Type'],
        'status': ['Status Code', 'Status', 'HTTP Status']
    },
    'screaming_frog_embeddings': {
        'url': ['Address', 'URL', 'Page URL'],
        'embedding': ['Embedding', 'Vector', 'Content Embedding'],
        'title': ['Title', 'Page Title'],
        'word_count': ['Word Count', 'Words', 'Content Length']
    },
    'gsc_export': {
        'page': ['Page', 'Landing Page', 'URL'],
        'query': ['Query', 'Search Query', 'Keyword'],
        'clicks': ['Clicks'],
        'impressions': ['Impressions'],
        'ctr': ['CTR', 'Click Through Rate'],
        'position': ['Position', 'Average Position']
    }
}

# Default export templates
EXPORT_TEMPLATES = {
    'educational_walkthrough': {
        'columns': [
            'Target URL', 'Queries Count', 'Clicks', 'Impressions',
            'Monthly Search Volume', 'Related URL 1', 'Status 1',
            'Related URL 2', 'Status 2', 'Related URL 3', 'Status 3',
            'Related URL 4', 'Status 4', 'Related URL 5', 'Status 5',
            'Related URL 6', 'Status 6', 'Related URL 7', 'Status 7',
            'Related URL 8', 'Status 8', 'Related URL 9', 'Status 9',
            'Related URL 10', 'Status 10'
        ]
    },
    'content_suggestions': {
        'columns': [
            'Target URL', 'Destination URL', 'Anchor Text',
            'Placement Hint', 'Content Snippet', 'Quality Score',
            'Implementation Priority'
        ]
    }
}

# Initialize default configuration
config = AppConfig()