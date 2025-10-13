import re 
import html
import pandas as pd
import numpy as np
import pickle

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

from sklearn.preprocessing import MinMaxScaler, RobustScaler










def transform(prices, scaler=None, rbscaler=None):
    prices = np.array(prices).reshape(-1, 1)

    # 1
    log_prices = np.log1p(prices)

    if scaler is None or rbscaler is None:
      #2
        rbscale = RobustScaler()
        rb_prices = rbscale.fit_transform(log_prices)  # Changed: save result
        #3
        scaler = MinMaxScaler()
        normalized_prices = scaler.fit_transform(rb_prices)  # Changed: use rb_prices

        with open('price_scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)
        with open('rb_scaler.pkl', 'wb') as f:
            pickle.dump(rbscale, f)
    else:
        with open(scaler, 'rb') as f:
            scaler = pickle.load(f)
        with open(rbscaler, 'rb') as f:
            rbscaler = pickle.load(f)
        rb_prices = rbscaler.transform(log_prices)
        normalized_prices = scaler.transform(rb_prices)

    return normalized_prices.flatten()

def inverse_transform(normalized_prices, scaler, rbscaler):
    normalized_prices = np.array(normalized_prices).reshape(-1, 1)

    with open(scaler, 'rb') as f:
        scaler = pickle.load(f)
    with open(rbscaler, 'rb') as f:
        rbscaler = pickle.load(f)
    #3
    rb_prices = scaler.inverse_transform(normalized_prices)  # Changed: inverse MinMaxScaler first
    #2
    log_prices = rbscaler.inverse_transform(rb_prices)  # Changed: then inverse RobustScaler
    #1
    original_prices = np.expm1(log_prices)

    return original_prices.flatten()