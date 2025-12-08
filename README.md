# Internal Link Finder with GSC Integration

A Python tool for discovering internal linking opportunities using vector embeddings from Screaming Frog, enhanced with Google Search Console performance metrics.

Based on [Everett Sizemore's methodology](https://moz.com/blog/internal-linking-opportunities-with-vector-embeddings) for finding internal linking opportunities with vector embeddings.

## Overview

This tool analyzes your website's internal link structure and identifies opportunities to add contextually relevant internal links. It uses:

1. **Screaming Frog Inlinks Report** - To understand existing internal links
2. **Screaming Frog Embeddings Report** - Vector embeddings for semantic similarity analysis
3. **GSC Organic Performance Report** - To prioritize opportunities by traffic impact

## Features

- Identifies the top N most semantically similar pages for each URL (configurable)
- **Cosine similarity scores** for each related page
- **Minimum similarity threshold filtering** for quality matches
- Checks if those related pages already link to the target URL
- Includes GSC performance metrics (Clicks, Impressions, Position, CTR) in output
- **URL cleaning** - Removes tracking parameters (`?`, `=`) and non-English characters
- Color-coded Excel output for easy identification of opportunities
- Flexible column name handling for GSC reports
- Works in **Streamlit Cloud**, Google Colab, and locally

## Deployment Options

### Option 1: Streamlit Cloud (Recommended)

Deploy your own instance in minutes:

1. Fork this repository to your GitHub account
2. Go to [Streamlit Cloud](https://streamlit.io/cloud)
3. Click "New app"
4. Select your forked repository
5. Set the main file path to `app.py`
6. Click "Deploy"

Your app will be live at `https://your-app-name.streamlit.app`

### Option 2: Run Streamlit Locally

```bash
# Clone the repository
git clone https://github.com/your-username/Internal-Link-Finder.git
cd Internal-Link-Finder

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
```

Open your browser to `http://localhost:8501`

### Option 3: Google Colab

1. Open Google Colab: [colab.research.google.com](https://colab.research.google.com)
2. Upload `internal_link_finder.py` or copy the code into a new notebook
3. Run the script
4. Upload your files when prompted
5. Download the output files

### Option 4: Command Line

```bash
# Install dependencies
pip install -r requirements.txt

# Run with default settings
python internal_link_finder.py

# Run with custom settings
python internal_link_finder.py --threshold 0.7 --top-n 10

# Disable URL cleaning
python internal_link_finder.py --no-clean
```

## Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--threshold`, `-t` | Minimum similarity threshold (0.0-1.0) | 0.0 |
| `--top-n`, `-n` | Number of related pages per URL | 5 |
| `--no-clean` | Disable URL cleaning | False |

## Requirements

### Software
- Python 3.8+
- Screaming Frog SEO Spider (with OpenAI API configured)

### Python Dependencies
```bash
pip install -r requirements.txt
```

## Input Files Required

### 1. Screaming Frog Inlinks Report
Export from Screaming Frog: **Bulk Export > Links > All Inlinks**

Required columns:
- Source
- Destination
- Type
- Status Code
- Link Position

### 2. Screaming Frog Embeddings Report
Export from Screaming Frog: **Bulk Export > Custom > Custom JavaScript**

Before crawling, set up embeddings extraction:
1. Go to **Configuration > API Access > AI** and add your OpenAI API key
2. Go to **Configuration > Custom > Custom JavaScript**
3. Click **Add from Library** > Select **(ChatGPT) Extract embeddings from page content**
4. Run your crawl

Required columns:
- Address/URL
- Embeddings column (contains vector data)

### 3. GSC Organic Performance Report
Export from Google Search Console or your preferred SEO tool.

Required columns (flexible naming supported):
| Data Type | Accepted Column Names |
|-----------|----------------------|
| URL | `Landing Page`, `Landing Pages`, `URL`, `URLs`, `Page`, `Pages`, `Address` |
| Clicks | `Clicks`, `Click`, `Total Clicks` |
| Impressions | `Impressions`, `Impression`, `Total Impressions`, `Impr` |
| Position | `Average Position`, `Avg. Position`, `Avg Position`, `Avg. Pos`, `Position`, `Pos` |
| CTR | `CTR`, `URL CTR`, `Average CTR`, `Avg. CTR`, `Click Through Rate` |

## Output

The script generates two output files:

### 1. `internal_link_opportunities.xlsx`
Excel file with color formatting:
- **Green cells**: Link already exists
- **Red cells**: Linking opportunity (link not found)

### 2. `internal_link_opportunities.csv`
Plain CSV for compatibility with other tools.

### Output Columns

| Column | Description |
|--------|-------------|
| URL | Target page URL |
| Clicks | GSC clicks for this page |
| Impressions | GSC impressions for this page |
| Avg. Position | Average ranking position |
| CTR | Click-through rate |
| Related URL 1-N | Top N semantically similar pages |
| Similarity 1-N | Cosine similarity score (higher = more relevant) |
| URL 1-N links to Target? | Whether the related page links to the target |

## How It Works

1. **Clean Inlinks Data**: Filters for relevant hyperlinks (status 200, content area)
2. **Process Embeddings**: Validates and parses vector embeddings
3. **Merge GSC Data**: Adds performance metrics with flexible column matching
4. **Clean URLs** (optional): Removes tracking parameters and non-English characters
5. **Calculate Similarity**: Uses cosine similarity to find related pages
6. **Apply Threshold** (optional): Filters out low-similarity matches
7. **Check Existing Links**: Verifies if related pages already link to targets
8. **Generate Report**: Creates prioritized list sorted by clicks

## URL Cleaning

When enabled (default), URL cleaning:
- Removes URLs containing `?` or `=` (tracking parameters)
- Removes URLs with non-ASCII characters (non-English)
- Removes fragment identifiers (`#`)
- Filters common tracking patterns (`utm_`, `fbclid`, `gclid`, etc.)

Disable with `--no-clean` flag or uncheck in Streamlit UI.

## Similarity Threshold

The similarity threshold filters out weak matches:
- **0.0** (default): Show all matches
- **0.5-0.6**: Moderate relevance
- **0.7-0.8**: Strong relevance (recommended)
- **0.9+**: Very high relevance

Higher thresholds = fewer but more relevant suggestions.

## Customization

### URL Filtering Patterns
Edit the `invalid_patterns` list in `clean_link_dataset()` to customize which URLs to exclude:

```python
invalid_patterns = [
    'category/', 'tag/', 'sitemap', 'search', '/home/', 'index'
]
```

## Troubleshooting

### "Could not find column" Error
Ensure your GSC export has the required columns. The tool accepts various column name formats - check the table above.

### Empty Embeddings
If embeddings are missing:
1. Verify OpenAI API key is configured in Screaming Frog
2. Check that JavaScript rendering is enabled
3. Re-crawl pages with missing embeddings

### Memory Issues
For large sites (10,000+ URLs), consider:
- Running in Google Colab (free GPU/memory)
- Running locally with more RAM
- Processing in batches
- Using Streamlit Cloud (has memory limits for free tier)

### Streamlit Cloud Deployment Issues
- Ensure `requirements.txt` is in the root directory
- Check that `app.py` is the main file
- Review logs in Streamlit Cloud dashboard

## Credits

- **Methodology**: [Everett Sizemore](https://moz.com/blog/internal-linking-opportunities-with-vector-embeddings)
- **Original Script**: [Britney Muller](https://maven.com/britney-muller/generative-ai-fundamentals)
- **Vector Embeddings Approach**: [Gus Pelogia](https://www.guspelogia.com/map-related-pages-embeddings)
- **iPullRank Article**: [Mike King](https://ipullrank.com/vector-embeddings-is-all-you-need)

## License

MIT License - See LICENSE file for details.
