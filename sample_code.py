import os
import random
import pandas as pd
import torch
from PIL import Image
import requests
from model import RegressionHead, IMAGE_PROCESSOR, IMAGE_ENCODER, TEXT_ENCODER, TEXT_TOKENIZER
from train import get_text_vision_emb
from text_transform import parse_catalog_content, transform, inverse_transform
from src.utils import download_image
from io import BytesIO

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

state_dict = torch.load("reg_head_epoch_0")
model = RegressionHead(IMAGE_ENCODER.config.hidden_size, TEXT_ENCODER.config.hidden_size).load_state_dict(state_dict)
# model = RegressionHead(IMAGE_ENCODER.config.hidden_size, TEXT_ENCODER.config.hidden_size)

model.to(device)
IMAGE_ENCODER.to(device)
TEXT_ENCODER.to(device)

def predictor(sample_id, catalog_content, image_link):
    '''
    Call your model/approach here
    
    Parameters:
    - sample_id: Unique identifier for the sample
    - catalog_content: Text containing product title and description
    - image_link: URL to product image
    
    Returns:
    - price: Predicted price as a float
    '''
    # catalog content preprocessing 
    parsed_data = parse_catalog_content(catalog_content=catalog_content)
    item = parsed_data["item"].lower()
    points = [p.lower() for p in parsed_data["points"]]
    value = parsed_data["value"].lower()
    unit = parsed_data["unit"].lower()

    # Build the formatted string with [SEP] separator
    formatted = f"item:{item}[SEP]points:"

    # Add all bullet points separated by [SEP]
    if points:
        formatted += "[SEP]".join(points)

    formatted += f"[SEP]value:{value}[SEP]unit:{unit}"
    # =============================
    # image_link download 
    try:
        response = requests.get(image_link, timeout=10)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert('RGB')
    except Exception as e:
        print(f"Warning: Failed to download image sample {image_link} from {image_link}: {e}")
        image = torch.zeros((224, 224, 3))
    # ==============================
    # text emb + image emb 
    text_emb, image_emb = get_text_vision_emb(formatted, image)
    text_emb.to(device)
    image_emb.to(device)
    # ==============================
    # model (text emb, image emb)
    output = model(text_emb, image_emb)
    # ==============================
    # inverse transform
    inv = inverse_transform(output.detach().numpy(), scaler=r"dataset/price_scaler.pkl", rbscaler=r"dataset/rb_scaler.pkl")
    print(inv[0])
    # ==============================
    return inv[0]

if __name__ == "__main__":
    DATASET_FOLDER = r'/Users/ayush/dev/train fas fas/amazon-ml-challenge/dataset/'
    
    # Read test data
    test = pd.read_csv(os.path.join(DATASET_FOLDER, 'sample_test.csv'))
    
    # Apply predictor function to each row
    test['price'] = test.apply(
        lambda row: predictor(row['sample_id'], row['catalog_content'], row['image_link']), 
        axis=1
    )
    
    # Select only required columns for output
    output_df = test[['sample_id', 'price']]
    
    # Save predictions
    output_filename = os.path.join(DATASET_FOLDER, 'sample_test_out.csv')
    output_df.to_csv(output_filename, index=False)
    
    print(f"Predictions saved to {output_filename}")
    print(f"Total predictions: {len(output_df)}")
    print(f"Sample predictions:\n{output_df.head()}")
