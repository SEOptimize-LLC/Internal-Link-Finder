"""
Reusable UI components for Streamlit
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from typing import Dict, Optional

class UIComponents:
    """Reusable UI components"""
    
    def render_progress_bar(self, current_step: int, total_steps: int):
        """Render workflow progress bar"""
        
        progress = current_step / total_steps
        
        # Create progress bar
        progress_bar = st.progress(progress)
        
        # Step indicators
        cols = st.columns(total_steps)
        steps = ['Upload', 'Process', 'Analyze', 'Generate', 'Export']
        
        for i, (col, step) in enumerate(zip(cols, steps)):
            with col:
                if i < current_step:
                    st.success(f"✅ {step}")
                elif i == current_step - 1:
                    st.info(f"🔄 {step}")
                else:
                    st.text(f"⭕ {step}")
    
    def render_dashboard(self, opportunities: pd.DataFrame,
                        gsc_data: Optional[Dict],
                        search_volumes: Optional[Dict]):
        """Render analytics dashboard"""
        
        st.header("📊 Analytics Dashboard")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Opportunities",
                len(opportunities)
            )
        
        with col2:
            avg_score = opportunities['similarity_score'].mean() if 'similarity_score' in opportunities.columns else 0
            st.metric(
                "Avg Similarity",
                f"{avg_score:.2%}"
            )
        
        with col3:
            if 'monthly_search_volume' in opportunities.columns:
                total_volume = opportunities['monthly_search_volume'].sum()
                st.metric(
                    "Total Search Volume",
                    f"{total_volume:,}"
                )
        
        with col4:
            if gsc_data:
                total_clicks = sum(gsc_data.get('url_clicks', {}).values())
                st.metric(
                    "Total Clicks",
                    f"{total_clicks:,}"
                )
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Similarity distribution
            if 'similarity_score' in opportunities.columns:
                fig = px.histogram(
                    opportunities,
                    x='similarity_score',
                    title='Similarity Score Distribution',
                    nbins=20
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Top opportunities
            if len(opportunities) > 0:
                top_opps = opportunities.head(10)
                fig = px.bar(
                    top_opps,
                    x='opportunity_score' if 'opportunity_score' in top_opps.columns else 'similarity_score',
                    y=top_opps.index,
                    orientation='h',
                    title='Top 10 Opportunities'
                )
                st.plotly_chart(fig, use_container_width=True)