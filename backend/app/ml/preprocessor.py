import re
from typing import List, Tuple

# Comprehensive English stop words set (no runtime download required)
ENGLISH_STOP_WORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", 
    "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", 
    "but", "by", "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't", 
    "doing", "don't", "down", "during", "each", "few", "for", "from", "further", "had", "hadn't", 
    "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here", 
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", 
    "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", "let's", 
    "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off", "on", 
    "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", 
    "same", "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", 
    "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", 
    "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", 
    "those", "through", "to", "too", "under", "until", "up", "very", "was", "wasn't", "we", 
    "we'd", "we'll", "we're", "we've", "were", "weren't", "what", "what's", "when", "when's", 
    "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", 
    "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours", 
    "yourself", "yourselves"
}

# Simple Devnagari detection regex for Hindi
DEVANAGARI_REGEX = re.compile(r'[\u0900-\u097F]')

# Common Hinglish markers
HINGLISH_MARKERS = {"kya", "hai", "nahi", "hota", "yeh", "woh", "bhi", "aur", "mein", "gaya", "karna", "sab"}

def detect_language(text: str) -> str:
    """
    Detects language: 'hi' (Hindi / Devanagari), 'hi-Latn' (Hinglish), or 'en' (English).
    """
    if not text:
        return "en"
    
    # Check for Devanagari script
    if DEVANAGARI_REGEX.search(text):
        return "hi"
    
    # Check for Hinglish markers in lowercased Latin tokens
    words = set(re.findall(r'\b[a-z]{3,}\b', text.lower()))
    hinglish_matches = words.intersection(HINGLISH_MARKERS)
    if len(hinglish_matches) >= 2:
        return "hi-Latn"
    
    return "en"

def clean_news_text(text: str, remove_stopwords: bool = False) -> str:
    """
    Cleans raw news text:
    - Normalizes URLs and emails
    - Strips special characters while preserving sentence terminators
    - Normalizes multiple spaces
    - Optionally removes stopwords
    """
    if not text:
        return ""
    
    # Remove URLs
    cleaned = re.sub(r'https?://\S+|www\.\S+', '', text)
    # Remove HTML tags if any
    cleaned = re.sub(r'<.*?>', '', cleaned)
    # Remove email addresses
    cleaned = re.sub(r'\S+@\S+', '', cleaned)
    # Remove hashtags and mentions
    cleaned = re.sub(r'[@#]\S+', '', cleaned)
    # Keep standard alphanumeric, punctuation useful for sentence segmentation
    cleaned = re.sub(r'[^a-zA-Z0-9\s.,!?\'"-]', ' ', cleaned)
    # Normalize whitespaces
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    
    if remove_stopwords:
        tokens = cleaned.lower().split()
        tokens = [t for t in tokens if t not in ENGLISH_STOP_WORDS]
        return ' '.join(tokens)
    
    return cleaned

def extract_sentences(text: str) -> List[str]:
    """
    Splits text into meaningful sentences based on punctuation and linebreaks.
    """
    if not text:
        return []
    
    # Split on periods, exclamation marks, question marks, and newlines
    raw_sentences = re.split(r'(?<=[.!?])\s+|\n+', text)
    valid_sentences = []
    
    for s in raw_sentences:
        clean_s = s.strip()
        # Sentence must have at least 4 words to form a meaningful claim
        if len(clean_s.split()) >= 4:
            valid_sentences.append(clean_s)
            
    return valid_sentences
