#!/usr/bin/env python3
"""
Internal Link Finder - Streamlit Web App

A web-based tool for discovering internal linking opportunities using vector embeddings
from Screaming Frog, enhanced with Google Search Console performance metrics.

Deploy on Streamlit Cloud: https://streamlit.io/cloud
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from io import BytesIO
from openpyxl import Workbook
from openpyxl.styles import PatternFill, Alignment, Font
from openpyxl.utils import get_column_letter

# Import core functions from the main module
from internal_link_finder import (
    clean_link_dataset,
    clean_embeddings_data,
    process_gsc_data,
    clean_url_column,
    DEFAULT_MIN_SIMILARITY
)

# Page configuration
st.set_page_config(
    page_title="Internal Link Finder",
    page_icon="🔗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .success-box {
        padding: 1rem;
        background-color: #d4edda;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .warning-box {
        padding: 1rem;
        background-color: #fff3cd;
        border-radius: 5px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


def find_related_pages_streamlit(df, top_n=5, min_similarity=0.0, progress_callback=None):
    """
    Find top N related pages based on cosine similarity.
    Streamlit-optimized version with progress callback.
    """
    related_pages = {}
    embeddings = np.stack(df['Embeddings'].values)
    urls = df['URL'].values

    # Calculate cosine similarity matrix
    cosine_similarities = cosine_similarity(embeddings)

    # For each URL, find the most similar URLs
    for idx, url in enumerate(urls):
        if progress_callback and idx % 10 == 0:
            progress_callback(idx / len(urls))

        similar_indices = cosine_similarities[idx].argsort()[::-1]

        related_with_scores = []
        for sim_idx in similar_indices:
            if urls[sim_idx] != url:
                score = cosine_similarities[idx][sim_idx]
                if score >= min_similarity:
                    related_with_scores.append((urls[sim_idx], round(score, 4)))
                    if len(related_with_scores) >= top_n:
                        break

        related_pages[url] = related_with_scores

    if progress_callback:
        progress_callback(1.0)

    return related_pages


def create_excel_output(final_df, top_n):
    """Create Excel file with formatting and return as bytes."""
    wb = Workbook()
    ws = wb.active
    ws.title = 'Internal Link Opportunities'

    # Write header
    headers = list(final_df.columns)
    for col_idx, header in enumerate(headers, 1):
        cell = ws.cell(row=1, column=col_idx, value=header)
        cell.font = Font(bold=True)
        cell.alignment = Alignment(horizontal='center', vertical='center')

    # Write data
    for row_idx, row in enumerate(final_df.values, 2):
        for col_idx, value in enumerate(row, 1):
            cell = ws.cell(row=row_idx, column=col_idx, value=value)
            cell.alignment = Alignment(vertical='center')

    # Define fills
    green_fill = PatternFill(start_color="C6EFCE", end_color="C6EFCE", fill_type="solid")
    red_fill = PatternFill(start_color="FFC7CE", end_color="FFC7CE", fill_type="solid")
    header_fill = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")

    # Style header row
    for col_idx in range(1, len(headers) + 1):
        cell = ws.cell(row=1, column=col_idx)
        cell.fill = header_fill
        cell.font = Font(bold=True, color="FFFFFF")

    # Apply conditional formatting to status columns
    base_cols = 5
    link_status_cols = [base_cols + (i * 3) + 3 for i in range(top_n)]

    for col_idx in link_status_cols:
        for row_idx in range(2, len(final_df) + 2):
            cell = ws.cell(row=row_idx, column=col_idx)
            if cell.value == "Exists":
                cell.fill = green_fill
            elif cell.value == "Not Found":
                cell.fill = red_fill

    # Set column widths
    base_widths = [50, 10, 12, 14, 10]
    for col_idx, width in enumerate(base_widths, 1):
        ws.column_dimensions[get_column_letter(col_idx)].width = width

    for i in range(top_n):
        col_base = base_cols + (i * 3) + 1
        ws.column_dimensions[get_column_letter(col_base)].width = 50
        ws.column_dimensions[get_column_letter(col_base + 1)].width = 12
        ws.column_dimensions[get_column_letter(col_base + 2)].width = 18

    ws.freeze_panes = 'B2'

    # Save to bytes
    output = BytesIO()
    wb.save(output)
    output.seek(0)
    return output.getvalue()


def main():
    # Header
    st.markdown('<div class="main-header">🔗 Internal Link Finder</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Discover internal linking opportunities using vector embeddings and GSC metrics</div>',
        unsafe_allow_html=True
    )

    # Sidebar for settings
    with st.sidebar:
        st.header("⚙️ Settings")

        top_n = st.slider(
            "Related pages per URL",
            min_value=1,
            max_value=20,
            value=5,
            help="Number of semantically similar pages to find for each URL"
        )

        min_similarity = st.slider(
            "Minimum similarity threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.05,
            help="Only show pages with similarity above this threshold. 0.7+ recommended for quality matches."
        )

        clean_urls = st.checkbox(
            "Clean URLs",
            value=True,
            help="Remove URLs with tracking parameters (?, =) and non-English characters"
        )

        st.divider()

        st.header("📚 About")
        st.markdown("""
        This tool analyzes your website's internal link structure using:

        1. **Screaming Frog Inlinks** - Existing link structure
        2. **Embeddings Report** - Semantic similarity via OpenAI
        3. **GSC Performance** - Traffic metrics for prioritization

        Based on [Everett Sizemore's methodology](https://moz.com/blog/internal-linking-opportunities-with-vector-embeddings).
        """)

    # Main content area
    st.header("📁 Upload Your Files")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("1. Inlinks Report")
        inlinks_file = st.file_uploader(
            "Screaming Frog All Inlinks CSV",
            type=['csv'],
            key="inlinks",
            help="Bulk Export > Links > All Inlinks"
        )

    with col2:
        st.subheader("2. Embeddings Report")
        embeddings_file = st.file_uploader(
            "Screaming Frog Embeddings CSV",
            type=['csv'],
            key="embeddings",
            help="Bulk Export > Custom > Custom JavaScript (with embeddings extraction)"
        )

    with col3:
        st.subheader("3. GSC Performance")
        gsc_file = st.file_uploader(
            "GSC Organic Performance CSV",
            type=['csv'],
            key="gsc",
            help="Export from Google Search Console with Landing Page, Clicks, Impressions, Position, CTR"
        )

    # Process button
    if st.button("🚀 Find Link Opportunities", type="primary", use_container_width=True):
        if not all([inlinks_file, embeddings_file, gsc_file]):
            st.error("Please upload all three required files.")
            return

        try:
            # Progress tracking
            progress_bar = st.progress(0, text="Starting analysis...")
            status_text = st.empty()

            # Step 1: Process inlinks
            status_text.text("Processing inlinks data...")
            progress_bar.progress(10, text="Processing inlinks...")

            df_links = pd.read_csv(inlinks_file)
            df_cleaned_links = clean_link_dataset(df_links)

            if clean_urls:
                df_cleaned_links = clean_url_column(df_cleaned_links, 'Source', 'Source URLs')
                df_cleaned_links = clean_url_column(df_cleaned_links, 'Destination', 'Destination URLs')

            # Step 2: Process embeddings
            status_text.text("Processing embeddings data...")
            progress_bar.progress(30, text="Processing embeddings...")

            df_embeddings_raw = pd.read_csv(embeddings_file)
            df_embeddings = clean_embeddings_data(df_embeddings_raw)

            if clean_urls:
                df_embeddings = clean_url_column(df_embeddings, 'URL', 'Embeddings URLs')

            # Convert embeddings to arrays
            df_embeddings['Embeddings'] = df_embeddings['Embeddings'].apply(
                lambda x: np.array([float(i) for i in str(x).strip('[]').replace("'", "").split(',')])
            )

            # Step 3: Process GSC data
            status_text.text("Processing GSC data...")
            progress_bar.progress(50, text="Processing GSC data...")

            df_gsc_raw = pd.read_csv(gsc_file)
            df_gsc = process_gsc_data(df_gsc_raw)

            if clean_urls:
                df_gsc = clean_url_column(df_gsc, 'URL', 'GSC URLs')

            # Step 4: Find related pages
            status_text.text("Calculating semantic similarity...")
            progress_bar.progress(60, text="Analyzing relationships...")

            def update_progress(pct):
                progress_bar.progress(int(60 + pct * 30), text=f"Finding related pages... {int(pct*100)}%")

            related_pages = find_related_pages_streamlit(
                df_embeddings,
                top_n=top_n,
                min_similarity=min_similarity,
                progress_callback=update_progress
            )

            # Step 5: Build output DataFrame
            status_text.text("Building results...")
            progress_bar.progress(90, text="Creating output...")

            output_data = []

            for url, related_with_scores in related_pages.items():
                padded_related = related_with_scores + [(None, None)] * (top_n - len(related_with_scores))

                gsc_row = df_gsc[df_gsc['URL'] == url]

                if not gsc_row.empty:
                    clicks = gsc_row['Clicks'].values[0]
                    impressions = gsc_row['Impressions'].values[0]
                    avg_position = gsc_row['Avg. Position'].values[0]
                    ctr = gsc_row['CTR'].values[0]
                else:
                    clicks = 0
                    impressions = 0
                    avg_position = None
                    ctr = "0.00%"

                row = {
                    'URL': url,
                    'Clicks': clicks,
                    'Impressions': impressions,
                    'Avg. Position': avg_position if pd.notna(avg_position) else '',
                    'CTR': ctr
                }

                for i, (related_url, similarity_score) in enumerate(padded_related, 1):
                    row[f'Related URL {i}'] = related_url
                    row[f'Similarity {i}'] = f"{similarity_score:.2%}" if similarity_score is not None else ""
                    if related_url is not None:
                        exists_status = (
                            "Exists"
                            if related_url in df_cleaned_links[
                                df_cleaned_links['Destination'] == url
                            ]['Source'].values
                            else "Not Found"
                        )
                        row[f'URL {i} links to Target?'] = exists_status
                    else:
                        row[f'URL {i} links to Target?'] = ""

                output_data.append(row)

            # Create final DataFrame
            column_order = ['URL', 'Clicks', 'Impressions', 'Avg. Position', 'CTR']
            for i in range(1, top_n + 1):
                column_order.extend([f'Related URL {i}', f'Similarity {i}', f'URL {i} links to Target?'])

            final_df = pd.DataFrame(output_data)
            final_df = final_df[column_order]
            final_df = final_df.sort_values('Clicks', ascending=False).reset_index(drop=True)

            progress_bar.progress(100, text="Complete!")
            status_text.empty()

            # Display results
            st.success("✅ Analysis complete!")

            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)

            total_opportunities = sum(
                (final_df[f'URL {i} links to Target?'] == 'Not Found').sum()
                for i in range(1, top_n + 1)
            )

            with col1:
                st.metric("URLs Analyzed", len(final_df))
            with col2:
                st.metric("URLs with Traffic", len(final_df[final_df['Clicks'] > 0]))
            with col3:
                st.metric("Link Opportunities", total_opportunities)
            with col4:
                existing_links = sum(
                    (final_df[f'URL {i} links to Target?'] == 'Exists').sum()
                    for i in range(1, top_n + 1)
                )
                st.metric("Existing Links", existing_links)

            # Download buttons
            st.header("📥 Download Results")

            col1, col2 = st.columns(2)

            with col1:
                excel_data = create_excel_output(final_df, top_n)
                st.download_button(
                    label="📊 Download Excel (with formatting)",
                    data=excel_data,
                    file_name="internal_link_opportunities.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )

            with col2:
                csv_data = final_df.to_csv(index=False)
                st.download_button(
                    label="📄 Download CSV",
                    data=csv_data,
                    file_name="internal_link_opportunities.csv",
                    mime="text/csv",
                    use_container_width=True
                )

            # Preview data
            st.header("📋 Results Preview")

            # Show only key columns in preview for readability
            preview_cols = ['URL', 'Clicks', 'Impressions', 'Related URL 1', 'Similarity 1', 'URL 1 links to Target?']
            st.dataframe(
                final_df[preview_cols].head(50),
                use_container_width=True,
                hide_index=True
            )

            with st.expander("View Full Results"):
                st.dataframe(final_df, use_container_width=True, hide_index=True)

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.exception(e)

    # Instructions section
    with st.expander("📖 How to Get the Required Files"):
        st.markdown("""
        ### 1. Screaming Frog Inlinks Report
        1. Open Screaming Frog and crawl your website
        2. Go to **Bulk Export > Links > All Inlinks**
        3. Save the CSV file

        ### 2. Screaming Frog Embeddings Report
        1. In Screaming Frog, go to **Configuration > API Access > AI**
        2. Add your OpenAI API key
        3. Go to **Configuration > Custom > Custom JavaScript**
        4. Click **Add from Library** > Select **(ChatGPT) Extract embeddings from page content**
        5. Run your crawl
        6. Go to **Bulk Export > Custom > Custom JavaScript**
        7. Save the CSV file

        ### 3. GSC Organic Performance Report
        1. Go to [Google Search Console](https://search.google.com/search-console)
        2. Navigate to **Performance > Search Results**
        3. Set your date range
        4. Click on **Pages** tab
        5. Export the data as CSV

        The tool accepts various column name formats for GSC data (Landing Page, URL, Clicks, etc.)
        """)


if __name__ == "__main__":
    main()
