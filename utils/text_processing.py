"""
Natural Language Processing and Text Processing Utilities
"""
import re
import string
from collections import Counter
import unicodedata


class TextProcessor:
    """Handle various text processing operations"""
    
    def __init__(self):
        self.stop_words = set([
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
            'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that',
            'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they'
        ])
    
    def clean_text(self, text):
        """Clean and normalize text"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove special characters but keep spaces
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def remove_punctuation(self, text):
        """Remove all punctuation from text"""
        return text.translate(str.maketrans('', '', string.punctuation))
    
    def remove_stop_words(self, text):
        """Remove common stop words"""
        words = text.split()
        filtered = [word for word in words if word.lower() not in self.stop_words]
        return ' '.join(filtered)
    
    def tokenize(self, text):
        """Split text into tokens/words"""
        return re.findall(r'\b\w+\b', text.lower())
    
    def word_frequency(self, text, top_n=10):
        """Get most common words and their frequencies"""
        words = self.tokenize(text)
        words = [w for w in words if w not in self.stop_words]
        counter = Counter(words)
        return counter.most_common(top_n)
    
    def extract_emails(self, text):
        """Extract email addresses from text"""
        pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        return re.findall(pattern, text)
    
    def extract_urls(self, text):
        """Extract URLs from text"""
        pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        return re.findall(pattern, text)
    
    def extract_phone_numbers(self, text):
        """Extract phone numbers from text"""
        patterns = [
            r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',  # 123-456-7890
            r'\b\(\d{3}\)\s*\d{3}[-.]?\d{4}\b',  # (123) 456-7890
            r'\b\d{10}\b'  # 1234567890
        ]
        
        numbers = []
        for pattern in patterns:
            numbers.extend(re.findall(pattern, text))
        
        return list(set(numbers))
    
    def extract_hashtags(self, text):
        """Extract hashtags from text"""
        return re.findall(r'#\w+', text)
    
    def extract_mentions(self, text):
        """Extract @mentions from text"""
        return re.findall(r'@\w+', text)
    
    def word_count(self, text):
        """Count words in text"""
        return len(self.tokenize(text))
    
    def char_count(self, text, include_spaces=True):
        """Count characters in text"""
        if include_spaces:
            return len(text)
        return len(text.replace(' ', ''))
    
    def sentence_count(self, text):
        """Count sentences in text"""
        sentences = re.split(r'[.!?]+', text)
        return len([s for s in sentences if s.strip()])
    
    def average_word_length(self, text):
        """Calculate average word length"""
        words = self.tokenize(text)
        if not words:
            return 0
        return sum(len(word) for word in words) / len(words)
    
    def reading_time(self, text, wpm=200):
        """Estimate reading time in minutes"""
        words = self.word_count(text)
        return round(words / wpm, 1)
    
    def remove_accents(self, text):
        """Remove accents from characters"""
        nfd = unicodedata.normalize('NFD', text)
        return ''.join(char for char in nfd if unicodedata.category(char) != 'Mn')
    
    def title_case(self, text):
        """Convert to title case"""
        return text.title()
    
    def snake_case(self, text):
        """Convert to snake_case"""
        text = re.sub(r'[\s\-]+', '_', text)
        text = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', text)
        return text.lower()
    
    def camel_case(self, text):
        """Convert to camelCase"""
        words = text.split()
        if not words:
            return ''
        return words[0].lower() + ''.join(word.capitalize() for word in words[1:])
    
    def kebab_case(self, text):
        """Convert to kebab-case"""
        return re.sub(r'[\s_]+', '-', text.lower())


class SentimentAnalyzer:
    """Simple sentiment analysis"""
    
    def __init__(self):
        self.positive_words = set([
            'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic',
            'love', 'best', 'happy', 'perfect', 'awesome', 'brilliant',
            'outstanding', 'superb', 'magnificent', 'beautiful', 'lovely'
        ])
        
        self.negative_words = set([
            'bad', 'terrible', 'awful', 'horrible', 'poor', 'worst',
            'hate', 'disappointing', 'useless', 'worthless', 'garbage',
            'disgusting', 'pathetic', 'annoying', 'frustrating', 'sad'
        ])
    
    def analyze(self, text):
        """Analyze sentiment of text"""
        words = text.lower().split()
        
        positive_count = sum(1 for word in words if word in self.positive_words)
        negative_count = sum(1 for word in words if word in self.negative_words)
        
        total = positive_count + negative_count
        
        if total == 0:
            return {'sentiment': 'neutral', 'score': 0, 'confidence': 0}
        
        score = (positive_count - negative_count) / total
        
        if score > 0.2:
            sentiment = 'positive'
        elif score < -0.2:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
        
        return {
            'sentiment': sentiment,
            'score': score,
            'confidence': abs(score),
            'positive_words': positive_count,
            'negative_words': negative_count
        }


def extract_keywords(text, top_n=10):
    """Extract key phrases from text"""
    processor = TextProcessor()
    
    # Clean text
    text = processor.clean_text(text)
    text = processor.remove_stop_words(text)
    
    # Get word frequencies
    words = processor.tokenize(text)
    
    # Filter short words
    words = [w for w in words if len(w) > 3]
    
    # Get most common
    counter = Counter(words)
    return [word for word, count in counter.most_common(top_n)]


def summarize_text(text, num_sentences=3):
    """Simple extractive text summarization"""
    # Split into sentences
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if len(sentences) <= num_sentences:
        return text
    
    # Score sentences by word frequency
    processor = TextProcessor()
    words = processor.tokenize(text)
    word_freq = Counter(words)
    
    sentence_scores = {}
    for sentence in sentences:
        score = 0
        words_in_sentence = processor.tokenize(sentence)
        for word in words_in_sentence:
            score += word_freq.get(word, 0)
        
        if len(words_in_sentence) > 0:
            sentence_scores[sentence] = score / len(words_in_sentence)
    
    # Get top sentences
    top_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)[:num_sentences]
    
    # Sort by original order
    result = []
    for sentence in sentences:
        if any(s[0] == sentence for s in top_sentences):
            result.append(sentence)
        if len(result) == num_sentences:
            break
    
    return '. '.join(result) + '.'
