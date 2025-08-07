# 🔗 Enhanced Internal Link Opportunity Finder

An advanced Streamlit application that identifies internal linking opportunities using Screaming Frog exports, Google Search Console data, and DataForSEO API, with automated content generation following a sophisticated linguistic framework.

## ✨ Features

### Core Functionality
- **📊 Multi-Source Data Integration**
  - Screaming Frog internal links analysis
  - Screaming Frog content embeddings for similarity matching
  - Google Search Console performance metrics
  - DataForSEO monthly search volumes

- **🤖 Intelligent Content Generation**
  - Advanced style matching using NLP
  - Automated snippet generation (3-6 sentences)
  - Context-aware anchor text selection
  - Quality assurance with scoring thresholds

- **📈 Advanced Analytics**
  - Similarity-based opportunity detection
  - Performance-weighted scoring
  - Search volume integration
  - Interactive dashboards

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- Streamlit account (for deployment)
- DataForSEO account (optional, for search volumes)
- Screaming Frog SEO Spider

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/enhanced-internal-link-finder.git
cd enhanced-internal-link-finder
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Download NLP models**
```bash
python -m spacy download en_core_web_sm
python -m nltk.downloader punkt stopwords averaged_perceptron_tagger
```

4. **Configure API credentials**

Create `.streamlit/secrets.toml`:
```toml
DATAFORSEO_LOGIN = "your_login"
DATAFORSEO_PASSWORD = "your_password"
```

5. **Run the application**
```bash
streamlit run app.py
```

## 📁 Data Requirements

### 1. Screaming Frog Internal Links Export
Export from Screaming Frog:
- Navigate to: **Bulk Export → Links → All Inlinks**
- Format: CSV
- Required columns: Source, Destination, Anchor Text, Type

### 2. Screaming Frog Embeddings Export
Export from Screaming Frog:
- Navigate to: **Content → Embeddings**
- Format: CSV
- Required columns: URL, Embedding, Title, Word Count

### 3. Google Search Console Export (Optional)
Export from GSC:
- Navigate to: **Performance → Export → CSV**
- Date range: Last 28-90 days
- Required data: Pages, Queries, Clicks, Impressions

## 🎯 Usage Guide

### Step 1: Upload Files
1. Upload your Screaming Frog exports (Links and Embeddings)
2. Optionally upload GSC performance data
3. Click "Process Files" to validate and load data

### Step 2: Analyze Opportunities
1. Choose analysis type:
   - **Find All Opportunities**: Comprehensive analysis
   - **Target Specific URLs**: Focus on particular pages
   - **Top Performers**: Prioritize high-traffic pages
2. Enable search volume fetching (requires DataForSEO)
3. Click "Run Analysis"

### Step 3: Generate Content
1. Select opportunities for content generation
2. Click "Generate Content Suggestions"
3. Review generated snippets with:
   - Anchor text (3-10 words)
   - Placement hints
   - Style-matched content (3-6 sentences)

### Step 4: Export Results
Choose your preferred format:
- **Excel**: Multi-sheet workbook with all data
- **CSV**: Simple tabular format
- **JSON**: Structured data for APIs

## 📊 Output Format

The tool generates suggestions in a standardized table format:

| Anchor Text | Placement Hint | Content Snippet |
|-------------|----------------|-----------------|
| [3-10 word phrase] | After '[text]' in [section] | [3-6 sentences with natural integration] |

### Quality Metrics
Each suggestion is validated against:
- **Format Compliance**: 100% required
- **Style Coherence**: ≥85% match with target content
- **Semantic Relevance**: ≥80% topical alignment
- **Flow Integration**: ≥75% natural placement

## 🔧 Configuration

### Modify Settings in `config.py`

```python
# Analysis thresholds
DEFAULT_SIMILARITY_THRESHOLD = 0.7
MAX_OPPORTUNITIES_PER_URL = 10

# Content generation
DEFAULT_SNIPPET_LENGTH = 4  # sentences
MIN_SNIPPET_LENGTH = 3
MAX_SNIPPET_LENGTH = 6

# Quality thresholds
MIN_STYLE_COHERENCE_SCORE = 0.85
MIN_SEMANTIC_RELEVANCE_SCORE = 0.80
```

### DataForSEO Settings
```python
DATAFORSEO_LOCATION = "United States"
DATAFORSEO_LANGUAGE = "en"
DATAFORSEO_BATCH_SIZE = 100
```

## 📈 Performance Optimization

### For Large Datasets
- Process files in batches (100-500 URLs at a time)
- Use similarity threshold to reduce computation
- Enable caching in Streamlit settings

### API Rate Limiting
- DataForSEO: 10 requests/second (configurable)
- Automatic retry with exponential backoff
- Batch processing for bulk operations

## 🐛 Troubleshooting

### Common Issues and Solutions

| Issue | Solution |
|-------|----------|
| "DataForSEO Not Connected" | Check credentials in `.streamlit/secrets.toml` |
| "No embeddings found" | Ensure Screaming Frog export includes embedding data |
| "Failed to scrape content" | Verify URLs are accessible; some sites block scraping |
| Memory errors | Process smaller batches; increase Streamlit limits |
| Slow processing | Reduce similarity threshold; limit opportunities per URL |

### Debug Mode
Enable debug logging in `config.py`:
```python
DEBUG_MODE = True
```

Check logs:
```bash
tail -f logs/app.log
```

## 🚀 Deployment

### Deploy to Streamlit Cloud

1. **Push to GitHub**
```bash
git add .
git commit -m "Initial deployment"
git push origin main
```

2. **Connect to Streamlit Cloud**
- Visit [share.streamlit.io](https://share.streamlit.io)
- Connect your GitHub repository
- Select branch and main file (`app.py`)

3. **Configure Secrets**
In Streamlit Cloud dashboard, add:
```toml
DATAFORSEO_LOGIN = "your_login"
DATAFORSEO_PASSWORD = "your_password"
```

4. **Deploy**
Click "Deploy" and wait for app to build

### Environment Variables
For local development, use `.env`:
```bash
DATAFORSEO_LOGIN=your_login
DATAFORSEO_PASSWORD=your_password
```

## 📚 Advanced Features

### Custom Style Profiles
Create custom style profiles for different content types:

```python
blog_style = StyleProfile(
    avg_sentence_length=15,
    complexity_score=8,
    formality_level='informal',
    vocabulary_level='intermediate'
)
```

### Quality Threshold Customization
Adjust quality requirements per project:

```python
strict_quality = QualityScores(
    format_compliance=1.0,
    style_coherence=0.90,  # Increased
    semantic_relevance=0.85,  # Increased
    flow_integration=0.80  # Increased
)
```

### Batch Processing API
Process multiple URL pairs programmatically:

```python
from src.content_generation.content_generator import InternalLinkContentGenerator

generator = InternalLinkContentGenerator()
for target, dest in url_pairs:
    suggestions = generator.generate_link_suggestions(target, dest)
```

## 📊 Metrics & Monitoring

### Success Metrics
Track performance with built-in analytics:
- Total opportunities identified
- Average similarity scores
- Content generation success rate
- Quality score distributions

### Export Analytics
Generated reports include:
- URL-level performance metrics
- Query and click data from GSC
- Monthly search volumes
- Opportunity scores

## 🤝 Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

### Development Setup
```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Format code
black src/

# Lint
flake8 src/
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Built with [Streamlit](https://streamlit.io)
- NLP powered by [spaCy](https://spacy.io) and [NLTK](https://www.nltk.org)
- Embeddings via [Sentence Transformers](https://www.sbert.net)
- SEO data from [DataForSEO](https://dataforseo.com)

## 📧 Support

For issues or questions:
- Create an issue on GitHub
- Check the [Wiki](https://github.com/yourusername/enhanced-internal-link-finder/wiki)
- Review closed issues for solutions

## 🗺️ Roadmap

### Version 2.1 (Planned)
- [ ] WordPress/CMS integration
- [ ] Bulk URL processing from sitemap
- [ ] Historical performance tracking
- [ ] A/B testing for suggestions

### Version 2.2 (Future)
- [ ] AI-powered content rewriting
- [ ] Competitor analysis integration
- [ ] Automated implementation via API
- [ ] Multi-language support

---

**Built with ❤️ by SEOptimize LLC for SEO professionals and content strategists**
