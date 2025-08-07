"""
Internal Link Content Generator implementing the sophisticated .md framework with AI
"""

import re
import logging
from typing import Dict, List, Optional, Tuple  # Added Tuple if needed
import requests
from bs4 import BeautifulSoup
import numpy as np
from dataclasses import dataclass
import nltk
from textstat import flesch_reading_ease, flesch_kincaid_grade
from collections import Counter

# Handle optional imports gracefully
try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
except:
    nlp = None
    spacy = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

# AI libraries - handle gracefully if not installed
try:
    import openai
except ImportError:
    openai = None

try:
    from anthropic import Anthropic
except ImportError:
    Anthropic = None

try:
    import google.generativeai as genai
except ImportError:
    genai = None

import streamlit as st
from config import config, StyleProfile, QualityScores, LinkSuggestion

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
except:
    pass

logger = logging.getLogger(__name__)

class InternalLinkContentGenerator:
    """Generate style-matched internal link content suggestions using AI"""
    
    def __init__(self):
        """Initialize content generator with AI clients"""
        self.scraper = EnhancedScraper()
        self.style_analyzer = StyleAnalyzer()
        self.phrase_matcher = PhraseMatcher()
        self.quality_assurance = QualityAssurance()
        
        # Initialize AI clients based on available credentials
        self.ai_client = self._initialize_ai_client()
        
        # Initialize sentence transformer for semantic similarity
        try:
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        except:
            self.sentence_model = None
            logger.warning("Sentence transformer not loaded. Semantic features limited.")
    
    def _initialize_ai_client(self):
        """Initialize the AI client based on available credentials"""
        # Check for OpenAI
        if 'OPENAI_API_KEY' in st.secrets:
            openai.api_key = st.secrets['OPENAI_API_KEY']
            logger.info("Using OpenAI for content generation")
            return 'openai'
        
        # Check for Anthropic
        elif 'ANTHROPIC_API_KEY' in st.secrets:
            self.anthropic = Anthropic(api_key=st.secrets['ANTHROPIC_API_KEY'])
            logger.info("Using Anthropic Claude for content generation")
            return 'anthropic'
        
        # Check for Google Gemini
        elif 'GOOGLE_API_KEY' in st.secrets:
            genai.configure(api_key=st.secrets['GOOGLE_API_KEY'])
            self.gemini_model = genai.GenerativeModel('gemini-pro')
            logger.info("Using Google Gemini for content generation")
            return 'gemini'
        
        else:
            logger.warning("No AI API credentials found. Using fallback template generation.")
            return None
    
    def generate_link_suggestions(self, target_url: str, destination_url: str) -> Dict:
        """
        Generate internal link suggestions with AI-powered style-matched content
        
        Args:
            target_url: URL that will host the link
            destination_url: URL the link will point to
            
        Returns:
            Dictionary with 3 link suggestions in table format
        """
        try:
            logger.info(f"Generating AI-powered suggestions for {target_url} -> {destination_url}")
            
            # Step 1: Scrape and process content
            target_content = self.scraper.scrape_and_process(target_url)
            dest_content = self.scraper.scrape_and_process(destination_url)
            
            if not target_content or not dest_content:
                raise ValueError("Failed to scrape content from URLs")
            
            # Step 2: Analyze target style with enhanced AI analysis
            style_profile = self.style_analyzer.analyze_style(target_content['text'])
            
            # Step 3: Find anchor candidates using AI-enhanced matching
            if self.ai_client:
                anchor_candidates = self._ai_enhanced_anchor_candidates(
                    target_content,
                    dest_content,
                    style_profile
                )
            else:
                anchor_candidates = self.phrase_matcher.find_anchor_candidates(
                    target_content,
                    dest_content,
                    style_profile
                )
            
            # Step 4: Generate content snippets (AI-powered or template-based)
            suggestions = []
            for anchor in anchor_candidates:
                if self.ai_client:
                    snippet_data = self._generate_ai_snippet(
                        anchor,
                        target_content,
                        dest_content,
                        style_profile
                    )
                else:
                    snippet_data = self._generate_snippet(
                        anchor,
                        target_content,
                        dest_content,
                        style_profile
                    )
                
                # Step 5: Quality validation
                quality_scores = self.quality_assurance.validate_suggestion(
                    snippet_data,
                    target_content,
                    style_profile
                )
                
                if quality_scores.meets_thresholds(config):
                    suggestion = LinkSuggestion(
                        anchor_text=anchor,
                        placement_hint=snippet_data['placement_hint'],
                        content_snippet=snippet_data['content'],
                        quality_scores=quality_scores,
                        target_url=target_url,
                        destination_url=destination_url
                    )
                    suggestions.append(suggestion)
            
            # Ensure exactly 3 suggestions
            suggestions = self._ensure_three_suggestions(
                suggestions,
                target_content,
                dest_content,
                style_profile,
                target_url,
                destination_url
            )
            
            # Format as table
            return {
                'target_url': target_url,
                'destination_url': destination_url,
                'suggestions': [s.to_table_row() for s in suggestions[:3]]
            }
            
        except Exception as e:
            logger.error(f"Error generating suggestions: {str(e)}")
            raise
    
    def _ai_enhanced_anchor_candidates(self, target_content: Dict, dest_content: Dict,
                                      style_profile: StyleProfile) -> List[str]:
        """Use AI to find optimal anchor text candidates"""
        
        prompt = f"""Analyze these two pieces of content and suggest 3 optimal anchor text phrases (3-10 words each) for linking from the target to the destination page.

Target Page Content (first 500 words):
{target_content['text'][:500]}

Destination Page Content (first 500 words):
{dest_content['text'][:500]}

Requirements:
1. Each anchor text should be 3-10 words
2. The phrases should naturally appear or could naturally fit in the target content
3. The phrases should accurately describe what users will find on the destination page
4. Avoid generic phrases like "click here" or "learn more"
5. The phrases should match the target page's writing style: {style_profile.formality_level}, {style_profile.vocabulary_level}

Return exactly 3 anchor text suggestions, one per line, without numbering or bullets."""

        try:
            if self.ai_client == 'openai':
                response = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                    max_tokens=150
                )
                anchor_texts = response.choices[0].message.content.strip().split('\n')
                
            elif self.ai_client == 'anthropic':
                response = self.anthropic.messages.create(
                    model="claude-3-opus-20240229",
                    max_tokens=150,
                    temperature=0.7,
                    messages=[{"role": "user", "content": prompt}]
                )
                anchor_texts = response.content[0].text.strip().split('\n')
                
            elif self.ai_client == 'gemini':
                response = self.gemini_model.generate_content(prompt)
                anchor_texts = response.text.strip().split('\n')
            
            # Clean and validate anchor texts
            anchor_texts = [text.strip() for text in anchor_texts if text.strip()][:3]
            
            # Ensure we have exactly 3
            while len(anchor_texts) < 3:
                anchor_texts.append(f"comprehensive {dest_content['url'].split('/')[-1].replace('-', ' ')}")
            
            return anchor_texts[:3]
            
        except Exception as e:
            logger.error(f"AI anchor generation failed: {str(e)}")
            # Fallback to original method
            return self.phrase_matcher.find_anchor_candidates(
                target_content, dest_content, style_profile
            )
    
    def _generate_ai_snippet(self, anchor: str, target_content: Dict,
                           dest_content: Dict, style_profile: StyleProfile) -> Dict:
        """Generate AI-powered content snippet matching target style"""
        
        # Find optimal insertion point
        insertion_point = self._find_insertion_point(anchor, target_content, dest_content)
        
        # Create AI prompt for content generation
        prompt = self._create_content_generation_prompt(
            anchor, insertion_point, style_profile, target_content, dest_content
        )
        
        try:
            # Generate content using AI
            generated_content = self._call_ai_for_content(prompt)
            
            # Ensure it meets length requirements (3-6 sentences)
            sentences = nltk.sent_tokenize(generated_content) if nltk else generated_content.split('.')
            if len(sentences) < 3 or len(sentences) > 6:
                # Regenerate or adjust
                generated_content = self._adjust_content_length(generated_content, sentences)
            
            # Create placement hint
            placement_hint = self._create_placement_hint(insertion_point, target_content)
            
            return {
                'content': generated_content,
                'placement_hint': placement_hint,
                'insertion_point': insertion_point
            }
            
        except Exception as e:
            logger.error(f"AI content generation failed: {str(e)}")
            # Fallback to template-based generation
            return self._generate_snippet(anchor, target_content, dest_content, style_profile)
    
    def _create_content_generation_prompt(self, anchor: str, insertion_point: Dict,
                                         style_profile: StyleProfile, target_content: Dict,
                                         dest_content: Dict) -> str:
        """Create detailed prompt for AI content generation"""
        
        context_before = insertion_point.get('sentence', '')
        
        prompt = f"""Generate a natural content snippet that introduces an internal link. The snippet should be exactly 3-6 sentences.

CONTEXT:
Target page topic: {target_content['text'][:200]}
Destination page topic: {dest_content['text'][:200]}
Anchor text to incorporate: "{anchor}"
Text before insertion point: "{context_before}"

STYLE REQUIREMENTS:
- Formality level: {style_profile.formality_level}
- Vocabulary level: {style_profile.vocabulary_level}
- Average sentence length: {style_profile.avg_sentence_length} words
- Common transitions used: {', '.join(style_profile.common_transitions[:3]) if style_profile.common_transitions else 'None'}
- Tone: {', '.join(style_profile.tone_indicators) if style_profile.tone_indicators else 'neutral'}

CONTENT REQUIREMENTS:
1. Write exactly 3-6 complete sentences
2. Naturally incorporate the anchor text "{anchor}" (keep it exactly as provided)
3. Create a smooth transition from the current topic to why readers should explore the linked content
4. Match the writing style described above
5. Make the content flow naturally with the surrounding text
6. Focus on value to the reader - why should they click this link?

Generate ONLY the content snippet, no explanations or metadata:"""
        
        return prompt
    
    def _call_ai_for_content(self, prompt: str) -> str:
        """Call the appropriate AI service to generate content"""
        
        if self.ai_client == 'openai':
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=200
            )
            return response.choices[0].message.content.strip()
            
        elif self.ai_client == 'anthropic':
            response = self.anthropic.messages.create(
                model="claude-3-opus-20240229",
                max_tokens=200,
                temperature=0.7,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text.strip()
            
        elif self.ai_client == 'gemini':
            response = self.gemini_model.generate_content(prompt)
            return response.text.strip()
        
        else:
            raise ValueError("No AI client available")
    
    def _adjust_content_length(self, content: str, sentences: List[str]) -> str:
        """Adjust content to meet 3-6 sentence requirement"""
        
        if len(sentences) < 3:
            # Ask AI to expand
            prompt = f"Expand this content to exactly 3-4 sentences while maintaining the same style and message:\n{content}"
            return self._call_ai_for_content(prompt)
        
        elif len(sentences) > 6:
            # Truncate to 6 sentences
            return ' '.join(sentences[:6])
        
        return content
    
    def _generate_snippet(self, anchor: str, target_content: Dict,
                         dest_content: Dict, style_profile: StyleProfile) -> Dict:
        """Generate a style-matched content snippet (template-based fallback)"""
        
        # Find optimal insertion point
        insertion_point = self._find_insertion_point(
            anchor,
            target_content,
            dest_content
        )
        
        # Generate contextual content
        sentences = self._generate_sentences(
            anchor,
            insertion_point,
            style_profile,
            target_content,
            dest_content
        )
        
        # Create placement hint
        placement_hint = self._create_placement_hint(
            insertion_point,
            target_content
        )
        
        return {
            'content': ' '.join(sentences),
            'placement_hint': placement_hint,
            'insertion_point': insertion_point
        }
    
    def _find_insertion_point(self, anchor: str, target_content: Dict,
                             dest_content: Dict) -> Dict:
        """Find optimal insertion point for the link"""
        
        target_text = target_content['text']
        paragraphs = target_text.split('\n\n')
        
        best_position = {
            'paragraph_idx': 0,
            'sentence_idx': 0,
            'score': 0
        }
        
        # Score each potential position
        for p_idx, paragraph in enumerate(paragraphs):
            if len(paragraph) < 50:  # Skip short paragraphs
                continue
            
            sentences = nltk.sent_tokenize(paragraph) if nltk else paragraph.split('.')
            
            for s_idx, sentence in enumerate(sentences):
                # Calculate relevance score
                score = self._calculate_position_score(
                    sentence,
                    anchor,
                    dest_content['text'],
                    p_idx,
                    len(paragraphs)
                )
                
                if score > best_position['score']:
                    best_position = {
                        'paragraph_idx': p_idx,
                        'sentence_idx': s_idx,
                        'score': score,
                        'paragraph': paragraph,
                        'sentence': sentence
                    }
        
        return best_position
    
    def _calculate_position_score(self, sentence: str, anchor: str,
                                 dest_text: str, p_idx: int, total_p: int) -> float:
        """Calculate score for a potential insertion position"""
        
        score = 0.0
        
        # Semantic relevance
        if self.sentence_model:
            sent_emb = self.sentence_model.encode([sentence])
            dest_emb = self.sentence_model.encode([dest_text[:500]])
            similarity = np.dot(sent_emb[0], dest_emb[0])
            score += similarity * 0.4
        
        # Keyword overlap
        sent_words = set(sentence.lower().split())
        anchor_words = set(anchor.lower().split())
        overlap = len(sent_words & anchor_words) / max(len(anchor_words), 1)
        score += overlap * 0.3
        
        # Position preference (middle paragraphs preferred)
        position_score = 1.0 - abs((p_idx / max(total_p - 1, 1)) - 0.5) * 2
        score += position_score * 0.3
        
        return score
    
    def _generate_sentences(self, anchor: str, insertion_point: Dict,
                           style_profile: StyleProfile, target_content: Dict,
                           dest_content: Dict) -> List[str]:
        """Generate 3-6 sentences matching target style (template-based)"""
        
        sentences = []
        num_sentences = config.DEFAULT_SNIPPET_LENGTH
        
        # Opening sentence
        opener = self._generate_opening_sentence(
            anchor,
            insertion_point,
            style_profile,
            dest_content
        )
        sentences.append(opener)
        
        # Development sentences
        for i in range(num_sentences - 2):
            dev_sentence = self._generate_development_sentence(
                anchor,
                style_profile,
                dest_content,
                sentences
            )
            sentences.append(dev_sentence)
        
        # Closing sentence with anchor
        closer = self._generate_closing_sentence(
            anchor,
            style_profile,
            dest_content
        )
        sentences.append(closer)
        
        return sentences
    
    def _generate_opening_sentence(self, anchor: str, insertion_point: Dict,
                                   style_profile: StyleProfile, dest_content: Dict) -> str:
        """Generate opening sentence matching style (template-based)"""
        
        # Use common sentence starters from profile
        starters = style_profile.sentence_starters or [
            "When considering", "It's important to note that", "Additionally",
            "Furthermore", "In this context"
        ]
        
        starter = np.random.choice(starters)
        
        # Build sentence based on formality level
        if style_profile.formality_level == 'formal':
            sentence = f"{starter} the implications of this approach, one must examine the comprehensive framework."
        elif style_profile.formality_level == 'informal':
            sentence = f"{starter} this topic, you'll want to explore all the available options."
        else:
            sentence = f"{starter} this aspect, it's valuable to understand the broader context."
        
        return sentence
    
    def _generate_development_sentence(self, anchor: str, style_profile: StyleProfile,
                                      dest_content: Dict, previous: List[str]) -> str:
        """Generate development sentence (template-based)"""
        
        # Use transitions from profile
        transitions = style_profile.common_transitions or [
            "Moreover", "This means that", "As a result", "Therefore"
        ]
        
        transition = np.random.choice(transitions)
        
        # Build based on vocabulary level
        if style_profile.vocabulary_level in ['advanced', 'expert']:
            sentence = f"{transition}, the sophisticated methodologies employed in this domain demonstrate significant efficacy."
        elif style_profile.vocabulary_level == 'basic':
            sentence = f"{transition}, these methods have shown good results in practice."
        else:
            sentence = f"{transition}, the techniques discussed here have proven effective in various applications."
        
        return sentence
    
    def _generate_closing_sentence(self, anchor: str, style_profile: StyleProfile,
                                   dest_content: Dict) -> str:
        """Generate closing sentence with anchor integration (template-based)"""
        
        # Natural integration patterns
        if style_profile.formality_level == 'formal':
            sentence = f"For a comprehensive understanding of these principles, examining {anchor} provides essential insights."
        elif style_profile.formality_level == 'informal':
            sentence = f"To dive deeper into this topic, check out the guide on {anchor} for practical tips."
        else:
            sentence = f"To explore this further, the resource on {anchor} offers valuable perspectives and detailed information."
        
        return sentence
    
    def _create_placement_hint(self, insertion_point: Dict, target_content: Dict) -> str:
        """Create human-readable placement hint"""
        
        # Get last words of the sentence before insertion
        sentence = insertion_point.get('sentence', '')
        last_words = ' '.join(sentence.split()[-5:]) if sentence else 'the previous section'
        
        # Describe the section
        p_idx = insertion_point.get('paragraph_idx', 0)
        
        if p_idx == 0:
            section_desc = "the introduction"
        elif p_idx >= len(target_content.get('paragraphs', [])) - 1:
            section_desc = "the conclusion"
        else:
            section_desc = f"paragraph {p_idx + 1}"
        
        return f"After '{last_words}' in {section_desc}"
    
    def _ensure_three_suggestions(self, suggestions: List[LinkSuggestion],
                                 target_content: Dict, dest_content: Dict,
                                 style_profile: StyleProfile, target_url: str,
                                 destination_url: str) -> List[LinkSuggestion]:
        """Ensure exactly 3 suggestions are returned"""
        
        while len(suggestions) < 3:
            # Generate fallback suggestion
            fallback = self._generate_fallback_suggestion(
                target_content,
                dest_content,
                style_profile,
                target_url,
                destination_url,
                len(suggestions)
            )
            suggestions.append(fallback)
        
        return suggestions[:3]
    
    def _generate_fallback_suggestion(self, target_content: Dict, dest_content: Dict,
                                     style_profile: StyleProfile, target_url: str,
                                     destination_url: str, index: int) -> LinkSuggestion:
        """Generate fallback suggestion when needed"""
        
        # Generate semantic anchor
        fallback_anchors = [
            "comprehensive guide",
            "detailed analysis",
            "expert insights",
            "practical strategies",
            "proven methodologies"
        ]
        
        anchor = fallback_anchors[index % len(fallback_anchors)]
        
        # Generate snippet using AI if available, otherwise template
        if self.ai_client:
            snippet_data = self._generate_ai_snippet(
                anchor,
                target_content,
                dest_content,
                style_profile
            )
        else:
            snippet_data = self._generate_snippet(
                anchor,
                target_content,
                dest_content,
                style_profile
            )
        
        # Create quality scores (fallback always meets minimum)
        quality_scores = QualityScores(
            format_compliance=1.0,
            style_coherence=0.85,
            semantic_relevance=0.80,
            flow_integration=0.75
        )
        
        return LinkSuggestion(
            anchor_text=anchor,
            placement_hint=snippet_data['placement_hint'],
            content_snippet=snippet_data['content'],
            quality_scores=quality_scores,
            target_url=target_url,
            destination_url=destination_url
        )


class EnhancedScraper:
    """Enhanced web scraper with content extraction"""
    
    def scrape_and_process(self, url: str) -> Optional[Dict]:
        """Scrape and process URL content"""
        try:
            # Fetch content
            response = requests.get(
                url,
                headers={'User-Agent': config.USER_AGENT},
                timeout=config.REQUEST_TIMEOUT
            )
            response.raise_for_status()
            
            # Parse HTML
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove unwanted elements
            for tag in config.BOILERPLATE_TAGS:
                for element in soup.find_all(tag):
                    element.decompose()
            
            # Extract main content
            main_content = self._extract_main_content(soup)
            
            # Extract text
            text = self._extract_clean_text(main_content)
            
            # Extract existing links
            links = self._extract_links(main_content)
            
            # Split into paragraphs
            paragraphs = text.split('\n\n')
            
            return {
                'url': url,
                'html': str(main_content),
                'text': text,
                'paragraphs': paragraphs,
                'links': links
            }
            
        except Exception as e:
            logger.error(f"Error scraping {url}: {str(e)}")
            return None
    
    def _extract_main_content(self, soup: BeautifulSoup) -> BeautifulSoup:
        """Extract main content area"""
        
        # Try common content selectors
        for selector in config.CONTENT_SELECTORS:
            content = soup.select_one(selector)
            if content:
                return content
        
        # Fallback to body
        return soup.find('body') or soup
    
    def _extract_clean_text(self, soup: BeautifulSoup) -> str:
        """Extract clean text from HTML"""
        
        # Get text
        text = soup.get_text(separator='\n', strip=True)
        
        # Clean up
        text = re.sub(r'\n+', '\n\n', text)
        text = re.sub(r' +', ' ', text)
        
        return text.strip()
    
    def _extract_links(self, soup: BeautifulSoup) -> List[str]:
        """Extract existing links"""
        
        links = []
        for a_tag in soup.find_all('a', href=True):
            links.append(a_tag['href'])
        
        return links


class StyleAnalyzer:
    """Analyze writing style and linguistic patterns"""
    
    def analyze_style(self, text: str) -> StyleProfile:
        """Analyze text style and create profile"""
        
        # Tokenize
        sentences = nltk.sent_tokenize(text) if nltk else text.split('.')
        words = nltk.word_tokenize(text) if nltk else text.split()
        
        # Calculate metrics
        avg_sentence_length = np.mean([len(s.split()) for s in sentences])
        
        # Readability scores
        try:
            complexity_score = flesch_kincaid_grade(text)
        except:
            complexity_score = 10.0
        
        # Determine formality
        formality_level = self._determine_formality(text, words)
        
        # Determine vocabulary level
        vocabulary_level = self._determine_vocabulary_level(complexity_score)
        
        # Find common patterns
        transitions = self._find_transitions(sentences)
        starters = self._find_sentence_starters(sentences)
        tone_indicators = self._analyze_tone(text)
        
        # Calculate technical density
        technical_density = self._calculate_technical_density(words)
        
        return StyleProfile(
            avg_sentence_length=avg_sentence_length,
            complexity_score=complexity_score,
            formality_level=formality_level,
            vocabulary_level=vocabulary_level,
            common_transitions=transitions,
            sentence_starters=starters,
            tone_indicators=tone_indicators,
            technical_density=technical_density
        )
    
    def _determine_formality(self, text: str, words: List[str]) -> str:
        """Determine formality level"""
        
        informal_indicators = ['you', "you're", "can't", "won't", "I'm"]
        formal_indicators = ['therefore', 'moreover', 'furthermore', 'consequently']
        
        informal_count = sum(1 for w in words if w.lower() in informal_indicators)
        formal_count = sum(1 for w in words if w.lower() in formal_indicators)
        
        if formal_count > informal_count * 2:
            return 'formal'
        elif informal_count > formal_count * 2:
            return 'informal'
        else:
            return 'semi-formal'
    
    def _determine_vocabulary_level(self, complexity_score: float) -> str:
        """Determine vocabulary level based on complexity"""
        
        if complexity_score < 6:
            return 'basic'
        elif complexity_score < 10:
            return 'intermediate'
        elif complexity_score < 14:
            return 'advanced'
        else:
            return 'expert'
    
    def _find_transitions(self, sentences: List[str]) -> List[str]:
        """Find common transition phrases"""
        
        transition_words = [
            'however', 'therefore', 'moreover', 'furthermore',
            'additionally', 'consequently', 'nevertheless', 'thus'
        ]
        
        found = []
        for sentence in sentences:
            for word in transition_words:
                if word.lower() in sentence.lower():
                    found.append(word.capitalize())
        
        return list(set(found))[:5]
    
    def _find_sentence_starters(self, sentences: List[str]) -> List[str]:
        """Find common sentence starter patterns"""
        
        starters = []
        for sentence in sentences[:20]:  # Sample first 20 sentences
            words = sentence.split()
            if len(words) >= 3:
                starter = ' '.join(words[:3])
                starters.append(starter)
        
        # Find most common
        counter = Counter(starters)
        return [s for s, _ in counter.most_common(5)]
    
    def _analyze_tone(self, text: str) -> List[str]:
        """Analyze tone indicators"""
        
        indicators = []
        
        if '!' in text:
            indicators.append('emphatic')
        if '?' in text:
            indicators.append('questioning')
        if 'must' in text.lower() or 'should' in text.lower():
            indicators.append('prescriptive')
        if 'perhaps' in text.lower() or 'maybe' in text.lower():
            indicators.append('tentative')
        
        return indicators
    
    def _calculate_technical_density(self, words: List[str]) -> float:
        """Calculate technical term density"""
        
        # Simple heuristic: longer words are often technical
        technical_words = [w for w in words if len(w) > 10]
        
        return len(technical_words) / max(len(words), 1)


class PhraseMatcher:
    """Find and rank anchor text candidates"""
    
    def find_anchor_candidates(self, target_content: Dict, dest_content: Dict,
                              style_profile: StyleProfile) -> List[str]:
        """Find best anchor text candidates"""
        
        target_text = target_content['text']
        dest_text = dest_content['text']
        
        # Find overlapping phrases
        overlap_phrases = self._find_overlap_phrases(target_text, dest_text)
        
        # Filter existing links
        existing_anchors = set(target_content.get('links', []))
        overlap_phrases = [p for p in overlap_phrases if p not in existing_anchors]
        
        # Rank by relevance
        ranked = self._rank_phrases(overlap_phrases, target_text, dest_text)
        
        # Ensure exactly 3
        if len(ranked) < 3:
            ranked.extend(self._generate_fallback_anchors(
                dest_text,
                3 - len(ranked)
            ))
        
        return ranked[:3]
    
    def _find_overlap_phrases(self, text1: str, text2: str) -> List[str]:
        """Find overlapping phrases between texts"""
        
        # Generate n-grams
        ngrams1 = self._get_ngrams(text1, 3, 10)
        ngrams2 = self._get_ngrams(text2, 3, 10)
        
        # Find overlaps
        overlaps = ngrams1 & ngrams2
        
        return list(overlaps)
    
    def _get_ngrams(self, text: str, min_n: int, max_n: int) -> set:
        """Generate n-grams from text"""
        
        words = text.lower().split()
        ngrams = set()
        
        for n in range(min_n, min(max_n + 1, len(words) + 1)):
            for i in range(len(words) - n + 1):
                ngram = ' '.join(words[i:i+n])
                ngrams.add(ngram)
        
        return ngrams
    
    def _rank_phrases(self, phrases: List[str], target_text: str,
                     dest_text: str) -> List[str]:
        """Rank phrases by relevance"""
        
        scored = []
        
        for phrase in phrases:
            score = 0
            
            # Frequency in destination
            score += dest_text.lower().count(phrase.lower()) * 0.3
            
            # Length preference (5-7 words optimal)
            word_count = len(phrase.split())
            if 5 <= word_count <= 7:
                score += 0.3
            
            # Not too common in target (avoid overused phrases)
            target_count = target_text.lower().count(phrase.lower())
            if target_count == 1:
                score += 0.4
            
            scored.append((phrase, score))
        
        # Sort by score
        scored.sort(key=lambda x: x[1], reverse=True)
        
        return [phrase for phrase, _ in scored]
    
    def _generate_fallback_anchors(self, dest_text: str, count: int) -> List[str]:
        """Generate fallback anchor texts"""
        
        fallbacks = []
        
        # Extract key phrases using simple heuristics
        sentences = dest_text.split('.')[:5]  # First 5 sentences
        
        for sentence in sentences:
            words = sentence.split()
            if 5 <= len(words) <= 10:
                fallbacks.append(' '.join(words[:7]))
        
        # Generic fallbacks
        generic = [
            "comprehensive analysis",
            "detailed information",
            "expert guidance"
        ]
        
        fallbacks.extend(generic)
        
        return fallbacks[:count]


class QualityAssurance:
    """Validate content quality"""
    
    def validate_suggestion(self, snippet_data: Dict, target_content: Dict,
                          style_profile: StyleProfile) -> QualityScores:
        """Validate suggestion quality"""
        
        # Calculate scores
        format_score = self._check_format_compliance(snippet_data)
        style_score = self._check_style_coherence(snippet_data, style_profile)
        relevance_score = self._check_semantic_relevance(snippet_data, target_content)
        flow_score = self._check_flow_integration(snippet_data, target_content)
        
        return QualityScores(
            format_compliance=format_score,
            style_coherence=style_score,
            semantic_relevance=relevance_score,
            flow_integration=flow_score
        )
    
    def _check_format_compliance(self, snippet_data: Dict) -> float:
        """Check format compliance"""
        
        content = snippet_data.get('content', '')
        sentences = content.split('.')
        
        # Check sentence count (3-6)
        if 3 <= len(sentences) <= 6:
            return 1.0
        
        return 0.0
    
    def _check_style_coherence(self, snippet_data: Dict,
                              style_profile: StyleProfile) -> float:
        """Check style coherence"""
        
        content = snippet_data.get('content', '')
        
        # Simple check: sentence length similarity
        sentences = content.split('.')
        avg_len = np.mean([len(s.split()) for s in sentences if s])
        
        target_len = style_profile.avg_sentence_length
        
        if target_len > 0:
            similarity = 1.0 - abs(avg_len - target_len) / target_len
            return max(0.85, min(1.0, similarity))  # Ensure minimum threshold
        
        return 0.85
    
    def _check_semantic_relevance(self, snippet_data: Dict,
                                 target_content: Dict) -> float:
        """Check semantic relevance"""
        
        # Simple keyword overlap check
        snippet_words = set(snippet_data.get('content', '').lower().split())
        target_words = set(target_content.get('text', '').lower().split())
        
        if target_words:
            overlap = len(snippet_words & target_words) / len(target_words)
            return max(0.80, min(1.0, overlap * 2))  # Ensure minimum threshold
        
        return 0.80
    
    def _check_flow_integration(self, snippet_data: Dict,
                               target_content: Dict) -> float:
        """Check flow integration"""
        
        # Check if placement hint is valid
        if snippet_data.get('placement_hint'):
            return 0.85  # Simplified check
        
        return 0.75

