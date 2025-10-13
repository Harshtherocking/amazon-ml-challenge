import re 
import html
import pandas as pd


def remove_emojis(text):
    """Remove emojis from text"""
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
        u"\U0001FA00-\U0001FA6F"  # Chess Symbols
        u"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
        u"\U00002600-\U000026FF"  # Miscellaneous Symbols
        u"\U00002700-\U000027BF"
        "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

def remove_html_tags(text):
    """Remove HTML tags but keep the text content"""
    text = html.unescape(text)
    clean = re.compile('<.*?>')
    text = re.sub(clean, '', text)
    return text

def parse_catalog_content(catalog_content):
    """Parse the catalog content and extract structured information"""
    if pd.isna(catalog_content):
        return {"item": "", "points": [], "value": "", "unit": ""}

    # Clean the text
    text = str(catalog_content).strip()
    text = remove_html_tags(text)
    text = remove_emojis(text)

    # Initialize extracted data
    item_name = ""
    bullet_points = []
    value = ""
    unit = ""

    # Split by newlines
    lines = text.split('\n')

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Extract Item Name
        if line.startswith("Item Name:"):
            item_name = line.replace("Item Name:", "").strip()

        # Extract Bullet Points
        elif line.startswith("Bullet Point"):
            bullet_text = re.sub(r'^Bullet Point\s*\d*:?\s*', '', line, flags=re.IGNORECASE).strip()
            if bullet_text:
                bullet_points.append(bullet_text)

        # Extract Product Description (treat as a bullet point)
        elif line.startswith("Product Description:"):
            desc_text = line.replace("Product Description:", "").strip()
            if desc_text:
                bullet_points.append(desc_text)

        # Extract Value
        elif line.startswith("Value:"):
            value = line.replace("Value:", "").strip()

        # Extract Unit
        elif line.startswith("Unit:"):
            unit = line.replace("Unit:", "").strip()

    return {
        "item": item_name,
        "points": bullet_points,
        "value": value,
        "unit": unit
    }