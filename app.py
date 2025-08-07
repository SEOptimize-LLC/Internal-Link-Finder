"""
Enhanced Internal Link Opportunity Finder
Main Streamlit Application
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import time
import json
from typing import Dict, List, Tuple, Optional
import logging

# Import custom modules
from src.core.data_processor import EnhancedDataProcessor
from src.core.gsc_processor import GSCDataProcessor
from src.core.dataforseo_client import DataForSEOClient
from src.core.similarity_engine import SimilarityEngine
from src.analyzers.link_analyzer import EnhancedLinkAnalyzer
from src.analyzers.performance_analyzer import PerformanceAnalyzer
from src.content_generation.content_generator import InternalLinkContentGenerator
from src.utils.export_utils import ExportUtils
from src.ui.components import UIComponents
from src.ui.workflows import WorkflowManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Enhanced Internal Link Opportunity Finder",
    page_icon="🔗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        padding: 0rem 1rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
    }
    .success-box {
        padding: 1rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.25rem;
        margin: 1rem 0;
    }
    .warning-box {
        padding: 1rem;
        background-color: #fff3cd;
        border: 1px solid #ffeeba;
        border-radius: 0.25rem;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

class EnhancedLinkFinderApp:
    """Main application class for Enhanced Internal Link Opportunity Finder"""
    
    def __init__(self):
        """Initialize the application"""
        self.init_session_state()
        self.data_processor = EnhancedDataProcessor()
        self.gsc_processor = GSCDataProcessor()
        self.similarity_engine = SimilarityEngine()
        self.link_analyzer = EnhancedLinkAnalyzer()
        self.performance_analyzer = PerformanceAnalyzer()
        self.content_generator = InternalLinkContentGenerator()
        self.export_utils = ExportUtils()
        self.ui_components = UIComponents()
        self.workflow_manager = WorkflowManager()
        
        # Initialize DataForSEO client if credentials available
        self.dataforseo_client = None
        if 'DATAFORSEO_LOGIN' in st.secrets and 'DATAFORSEO_PASSWORD' in st.secrets:
            self.dataforseo_client = DataForSEOClient(
                st.secrets['DATAFORSEO_LOGIN'],
                st.secrets['DATAFORSEO_PASSWORD']
            )
    
    def init_session_state(self):
        """Initialize session state variables"""
        if 'processed_data' not in st.session_state:
            st.session_state.processed_data = None
        if 'gsc_data' not in st.session_state:
            st.session_state.gsc_data = None
        if 'search_volumes' not in st.session_state:
            st.session_state.search_volumes = {}
        if 'opportunities' not in st.session_state:
            st.session_state.opportunities = None
        if 'content_suggestions' not in st.session_state:
            st.session_state.content_suggestions = None
        if 'current_step' not in st.session_state:
            st.session_state.current_step = 1
        if 'processing_status' not in st.session_state:
            st.session_state.processing_status = {}
    
    def render_header(self):
        """Render application header"""
        st.title("🔗 Enhanced Internal Link Opportunity Finder")
        st.markdown("""
        **Advanced internal linking analysis with Google Search Console data and automated content generation**
        
        This tool helps you:
        - 📊 Analyze internal link opportunities using Screaming Frog data
        - 📈 Integrate GSC performance metrics (queries, clicks, impressions)
        - 🔍 Get monthly search volume data from DataForSEO
        - ✍️ Generate style-matched content snippets for natural link insertion
        - 📋 Export comprehensive reports in multiple formats
        """)
        
        # Display workflow progress
        self.ui_components.render_progress_bar(st.session_state.current_step, 5)
    
    def render_sidebar(self):
        """Render sidebar with configuration options"""
        with st.sidebar:
            st.header("⚙️ Configuration")
            
            # API Status
            st.subheader("API Status")
            if self.dataforseo_client:
                st.success("✅ DataForSEO Connected")
            else:
                st.warning("⚠️ DataForSEO Not Configured")
                with st.expander("Setup Instructions"):
                    st.markdown("""
                    Add to `.streamlit/secrets.toml`:
                    ```toml
                    DATAFORSEO_LOGIN = "your_login"
                    DATAFORSEO_PASSWORD = "your_password"
                    ```
                    """)
            
            # Analysis Settings
            st.subheader("Analysis Settings")
            
            # Similarity threshold
            similarity_threshold = st.slider(
                "Similarity Threshold",
                min_value=0.5,
                max_value=1.0,
                value=0.7,
                step=0.05,
                help="Minimum similarity score for related pages"
            )
            
            # Max opportunities per URL
            max_opportunities = st.number_input(
                "Max Opportunities per URL",
                min_value=1,
                max_value=20,
                value=10,
                help="Maximum number of link opportunities to find per target URL"
            )
            
            # Content generation settings
            st.subheader("Content Generation")
            
            snippet_length = st.selectbox(
                "Snippet Length (sentences)",
                options=[3, 4, 5, 6],
                index=1,
                help="Number of sentences for generated content snippets"
            )
            
            style_matching = st.checkbox(
                "Advanced Style Matching",
                value=True,
                help="Use sophisticated linguistic analysis for style matching"
            )
            
            # Save settings
            if st.button("💾 Save Settings"):
                st.session_state.settings = {
                    'similarity_threshold': similarity_threshold,
                    'max_opportunities': max_opportunities,
                    'snippet_length': snippet_length,
                    'style_matching': style_matching
                }
                st.success("Settings saved!")
            
            # Help section
            st.subheader("📚 Help")
            with st.expander("Required Files"):
                st.markdown("""
                **1. Screaming Frog Internal Links Export**
                - Export: Bulk Export > Links > All Inlinks
                
                **2. Screaming Frog Embeddings Export**
                - Export: Content > Embeddings
                
                **3. Google Search Console Report (CSV)**
                - Export: Performance > Export > CSV
                - Include: Pages, Queries, Clicks, Impressions
                """)
    
    def render_file_upload(self):
        """Render file upload section"""
        st.header("📁 Step 1: Upload Data Files")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Internal Links")
            links_file = st.file_uploader(
                "Upload Screaming Frog Links Export",
                type=['csv'],
                key='links_upload',
                help="CSV export from Screaming Frog containing internal links data"
            )
            if links_file:
                st.success(f"✅ {links_file.name}")
        
        with col2:
            st.subheader("Embeddings")
            embeddings_file = st.file_uploader(
                "Upload Screaming Frog Embeddings Export",
                type=['csv'],
                key='embeddings_upload',
                help="CSV export from Screaming Frog containing page embeddings"
            )
            if embeddings_file:
                st.success(f"✅ {embeddings_file.name}")
        
        with col3:
            st.subheader("GSC Data (Optional)")
            gsc_file = st.file_uploader(
                "Upload Google Search Console Report",
                type=['csv'],
                key='gsc_upload',
                help="CSV export from GSC with performance data"
            )
            if gsc_file:
                st.success(f"✅ {gsc_file.name}")
        
        # Process button
        if links_file and embeddings_file:
            if st.button("🚀 Process Files", type="primary", use_container_width=True):
                self.process_uploaded_files(links_file, embeddings_file, gsc_file)
    
    def process_uploaded_files(self, links_file, embeddings_file, gsc_file=None):
        """Process uploaded files"""
        try:
            with st.spinner("Processing files..."):
                # Create progress placeholder
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Step 1: Load and validate files
                status_text.text("Loading files...")
                progress_bar.progress(20)
                
                files_dict = {
                    'links': links_file,
                    'embeddings': embeddings_file
                }
                if gsc_file:
                    files_dict['gsc'] = gsc_file
                
                # Process files
                status_text.text("Processing internal links data...")
                progress_bar.progress(40)
                processed_data = self.data_processor.process_multiple_files(files_dict)
                st.session_state.processed_data = processed_data
                
                # Process GSC data if available
                if gsc_file:
                    status_text.text("Processing GSC performance data...")
                    progress_bar.progress(60)
                    gsc_metrics = self.gsc_processor.process_gsc_export(gsc_file)
                    st.session_state.gsc_data = gsc_metrics
                
                # Calculate similarities
                status_text.text("Calculating content similarities...")
                progress_bar.progress(80)
                
                # Complete
                progress_bar.progress(100)
                status_text.text("Processing complete!")
                time.sleep(1)
                
                # Clear progress indicators
                progress_bar.empty()
                status_text.empty()
                
                # Update session state
                st.session_state.current_step = 2
                st.success("✅ Files processed successfully!")
                
                # Display summary
                self.display_processing_summary()
                
        except Exception as e:
            st.error(f"Error processing files: {str(e)}")
            logger.error(f"File processing error: {str(e)}")
    
    def display_processing_summary(self):
        """Display summary of processed data"""
        if st.session_state.processed_data:
            st.subheader("📊 Processing Summary")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Total Pages",
                    st.session_state.processed_data['total_pages']
                )
            
            with col2:
                st.metric(
                    "Total Links",
                    st.session_state.processed_data['total_links']
                )
            
            with col3:
                if st.session_state.gsc_data:
                    st.metric(
                        "Pages with GSC Data",
                        len(st.session_state.gsc_data['url_queries_count'])
                    )
                else:
                    st.metric("GSC Data", "Not Available")
            
            with col4:
                st.metric(
                    "Processing Time",
                    f"{st.session_state.processed_data.get('processing_time', 0):.2f}s"
                )
    
    def render_analysis_section(self):
        """Render analysis section"""
        if st.session_state.processed_data is None:
            st.info("📤 Please upload and process files first")
            return
        
        st.header("🔍 Step 2: Link Opportunity Analysis")
        
        # Analysis options
        col1, col2 = st.columns(2)
        
        with col1:
            analysis_type = st.selectbox(
                "Analysis Type",
                ["Find All Opportunities", "Target Specific URLs", "Top Performers (GSC)"]
            )
        
        with col2:
            if st.session_state.gsc_data and self.dataforseo_client:
                fetch_volumes = st.checkbox(
                    "Fetch Search Volumes",
                    value=True,
                    help="Get monthly search volumes from DataForSEO (may take time)"
                )
            else:
                fetch_volumes = False
        
        # Run analysis
        if st.button("🔎 Run Analysis", type="primary", use_container_width=True):
            self.run_link_analysis(analysis_type, fetch_volumes)
    
    def run_link_analysis(self, analysis_type, fetch_volumes):
        """Run link opportunity analysis"""
        try:
            with st.spinner("Analyzing link opportunities..."):
                # Get opportunities
                opportunities = self.link_analyzer.analyze_opportunities(
                    st.session_state.processed_data,
                    st.session_state.gsc_data,
                    analysis_type
                )
                
                # Fetch search volumes if requested
                if fetch_volumes and self.dataforseo_client:
                    opportunities = self.fetch_and_add_search_volumes(opportunities)
                
                # Store results
                st.session_state.opportunities = opportunities
                st.session_state.current_step = 3
                
                # Display results
                self.display_analysis_results(opportunities)
                
        except Exception as e:
            st.error(f"Analysis error: {str(e)}")
            logger.error(f"Analysis error: {str(e)}")
    
    def fetch_and_add_search_volumes(self, opportunities):
        """Fetch search volumes for queries"""
        with st.spinner("Fetching search volumes from DataForSEO..."):
            # Extract unique queries
            all_queries = set()
            if st.session_state.gsc_data:
                for url, queries in st.session_state.gsc_data['url_queries'].items():
                    all_queries.update(queries)
            
            if all_queries:
                # Fetch volumes in batches
                search_volumes = self.dataforseo_client.get_bulk_search_volumes(
                    list(all_queries),
                    location="United States",
                    batch_size=100
                )
                st.session_state.search_volumes = search_volumes
                
                # Add to opportunities
                opportunities = self.performance_analyzer.enhance_with_search_volumes(
                    opportunities,
                    search_volumes
                )
        
        return opportunities
    
    def display_analysis_results(self, opportunities):
        """Display analysis results"""
        st.subheader("📊 Link Opportunities Found")
        
        # Summary metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Opportunities", len(opportunities))
        with col2:
            avg_score = opportunities['similarity_score'].mean() if len(opportunities) > 0 else 0
            st.metric("Avg Similarity Score", f"{avg_score:.2%}")
        with col3:
            if 'monthly_search_volume' in opportunities.columns:
                total_volume = opportunities['monthly_search_volume'].sum()
                st.metric("Total Search Volume", f"{total_volume:,}")
        
        # Display opportunities table
        st.dataframe(
            opportunities,
            use_container_width=True,
            hide_index=True
        )
    
    def render_content_generation(self):
        """Render content generation section"""
        if st.session_state.opportunities is None:
            st.info("🔍 Please run analysis first")
            return
        
        st.header("✍️ Step 3: Generate Internal Link Content")
        
        # Select opportunities for content generation
        selected_opps = st.multiselect(
            "Select opportunities for content generation",
            options=st.session_state.opportunities.index,
            format_func=lambda x: f"{st.session_state.opportunities.loc[x, 'target_url'][:50]}... → {st.session_state.opportunities.loc[x, 'related_url'][:50]}..."
        )
        
        if selected_opps:
            if st.button("✨ Generate Content Suggestions", type="primary", use_container_width=True):
                self.generate_content_suggestions(selected_opps)
    
    def generate_content_suggestions(self, selected_indices):
        """Generate content suggestions for selected opportunities"""
        try:
            suggestions = []
            progress_bar = st.progress(0)
            
            for i, idx in enumerate(selected_indices):
                opp = st.session_state.opportunities.loc[idx]
                
                # Update progress
                progress = (i + 1) / len(selected_indices)
                progress_bar.progress(progress)
                
                # Generate suggestions
                suggestion = self.content_generator.generate_link_suggestions(
                    opp['target_url'],
                    opp['related_url']
                )
                suggestions.append(suggestion)
            
            # Store suggestions
            st.session_state.content_suggestions = suggestions
            st.session_state.current_step = 4
            
            # Display suggestions
            self.display_content_suggestions(suggestions)
            
        except Exception as e:
            st.error(f"Content generation error: {str(e)}")
            logger.error(f"Content generation error: {str(e)}")
    
    def display_content_suggestions(self, suggestions):
        """Display generated content suggestions"""
        st.subheader("📝 Generated Content Suggestions")
        
        for i, suggestion in enumerate(suggestions):
            with st.expander(f"Suggestion {i+1}: {suggestion['target_url'][:50]}..."):
                # Display as table
                df = pd.DataFrame(suggestion['suggestions'])
                st.table(df)
                
                # Copy button for each suggestion
                for j, row in df.iterrows():
                    if st.button(f"📋 Copy Snippet {j+1}", key=f"copy_{i}_{j}"):
                        st.code(row['content_snippet'], language='text')
    
    def render_export_section(self):
        """Render export section"""
        if st.session_state.opportunities is None:
            st.info("📊 No results to export yet")
            return
        
        st.header("📥 Step 4: Export Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📊 Export to Excel", use_container_width=True):
                file_path = self.export_utils.export_to_excel(
                    st.session_state.opportunities,
                    st.session_state.content_suggestions
                )
                with open(file_path, 'rb') as f:
                    st.download_button(
                        "⬇️ Download Excel",
                        f,
                        file_name=f"link_opportunities_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
        
        with col2:
            if st.button("📄 Export to CSV", use_container_width=True):
                csv = st.session_state.opportunities.to_csv(index=False)
                st.download_button(
                    "⬇️ Download CSV",
                    csv,
                    file_name=f"link_opportunities_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        with col3:
            if st.button("📋 Export to JSON", use_container_width=True):
                json_data = st.session_state.opportunities.to_json(orient='records')
                st.download_button(
                    "⬇️ Download JSON",
                    json_data,
                    file_name=f"link_opportunities_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
    
    def run(self):
        """Run the main application"""
        # Render header
        self.render_header()
        
        # Render sidebar
        self.render_sidebar()
        
        # Main content area with tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📁 Upload Files",
            "🔍 Analyze",
            "✍️ Generate Content",
            "📥 Export",
            "📊 Dashboard"
        ])
        
        with tab1:
            self.render_file_upload()
        
        with tab2:
            self.render_analysis_section()
        
        with tab3:
            self.render_content_generation()
        
        with tab4:
            self.render_export_section()
        
        with tab5:
            if st.session_state.opportunities is not None:
                self.ui_components.render_dashboard(
                    st.session_state.opportunities,
                    st.session_state.gsc_data,
                    st.session_state.search_volumes
                )
            else:
                st.info("📊 Complete analysis to view dashboard")

def main():
    """Main entry point"""
    app = EnhancedLinkFinderApp()
    app.run()

if __name__ == "__main__":
    main()