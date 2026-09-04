import re
from typing import List, Dict, Any, Tuple

TITLES = {"president", "prime minister", "governor", "minister", "director", "chief", "judge", "dr", "dr.", "prof", "prof."}
ORG_KEYWORDS = {"nasa", "who", "un", "isro", "cdc", "fbi", "cia", "eu", "iaea", "imf", "biden", "trump", "modi", "supreme court", "government", "parliament", "congress", "senate", "esa", "rbi"}
COMMON_LOCS = {"mars", "moon", "earth", "antarctica", "washington", "delhi", "tokyo", "paris", "london", "vatican", "fukushima", "florida", "california", "beijing", "moscow", "geneva", "mumbai"}

def extract_numbers_and_statistics(text: str) -> List[str]:
    """
    Extracts statistical expressions, percentages, amounts, and numeric data.
    e.g. '200%', '200 percent', '$500 million', '1,200', '3.2 percent', '25 basis points'
    """
    numbers = []
    
    # Percentages
    pct_matches = re.findall(r'\b\d+(?:\.\d+)?\s*(?:%|percent\b)', text, re.IGNORECASE)
    numbers.extend(pct_matches)
    
    # Currencies & Values ($500M, ₹2000, 500 rupee)
    curr_matches = re.findall(r'(?:[\$€£₹]\s*\d+(?:,\d+)*(?:\.\d+)?(?:\s*(?:billion|million|trillion))?|\b\d+(?:,\d+)*\s*(?:rupees?|dollars?|euros?|pounds?)\b)', text, re.IGNORECASE)
    numbers.extend(curr_matches)
    
    # Large formatted numbers (1,200, 150,000)
    num_matches = re.findall(r'\b\d{1,3}(?:,\d{3})+\b', text)
    numbers.extend(num_matches)
    
    # Basis points or ratios
    bp_matches = re.findall(r'\b\d+\s+basis\s+points\b', text, re.IGNORECASE)
    numbers.extend(bp_matches)
    
    return list(dict.fromkeys(numbers))[:5]

def extract_entities_and_dates(text: str) -> Dict[str, List[str]]:
    """
    Extracts named entities (organizations, people, locations), numbers, and temporal expressions.
    """
    entities = []
    dates = []
    locations = []
    numbers = extract_numbers_and_statistics(text)
    
    # Extract Dates: 4-digit years (1900-2099), dates like 'Jan 15', '2024-05-12', 'yesterday', 'tomorrow'
    year_matches = re.findall(r'\b(19\d{2}|20\d{2})\b', text)
    month_day_matches = re.findall(r'\b(January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2}(?:st|nd|rd|th)?(?:,?\s+\d{4})?\b', text, re.IGNORECASE)
    relative_dates = re.findall(r'\b(today|yesterday|tomorrow|next week|last month|recent years|next year|next month)\b', text, re.IGNORECASE)
    
    for y in year_matches:
        if y not in dates:
            dates.append(y)
    for m in month_day_matches:
        if m not in dates:
            dates.append(m)
    for r in relative_dates:
        if r.lower() not in dates:
            dates.append(r.lower())

    # Extract Organizations and Prominent Figures
    text_lower = text.lower()
    for org in ORG_KEYWORDS:
        if re.search(r'\b' + re.escape(org) + r'\b', text_lower):
            entities.append(org.upper() if len(org) <= 4 else org.title())

    # Extract Capitalized Multi-word Nouns (proper names, entities)
    proper_nouns = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
    for pn in proper_nouns:
        pn_clean = pn.strip()
        if len(pn_clean) > 2 and pn_clean.lower() not in {"the", "this", "that", "breaking", "urgent", "shocking", "confirmed", "official", "municipal"}:
            if pn_clean not in entities:
                entities.append(pn_clean)

    # Locations
    for loc in COMMON_LOCS:
        if re.search(r'\b' + re.escape(loc) + r'\b', text_lower):
            locations.append(loc.title())

    return {
        "entities": entities[:6],
        "dates": dates[:4],
        "locations": locations[:4],
        "numbers": numbers
    }

def normalize_claim_text(text: str) -> str:
    """
    Cleans sensational prefixes and filler words to produce a standardized queryable claim,
    while leaving the original claim intact.
    """
    sensational_prefixes = [
        r"^shocking:?\s*",
        r"^urgent\s+alert:?\s*",
        r"^breaking\s*(news)?:?\s*",
        r"^confirmed:?\s*",
        r"^exposed:?\s*",
        r"^leaked:?\s*",
        r"^watch:?\s*",
        r"^must\s+watch:?\s*",
        r"^they\s+don't\s+want\s+you\s+to\s+know:?\s*",
        r"^viral\s+reports\s+(claim|state|indicate)\s+(that)?\s*",
        r"^furthermore,?\s*"
    ]
    
    normalized = text.strip()
    for pattern in sensational_prefixes:
        normalized = re.sub(pattern, "", normalized, flags=re.IGNORECASE).strip()
        
    return normalized
