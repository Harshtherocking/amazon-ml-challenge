import torch
import os
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
from torchvision import transforms
import requests
from io import BytesIO

class AmazonDataset(Dataset):
    def __init__(self, csv_path, images_path):
        self.df = pd.read_csv(csv_path)
        self.images_path = images_path
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = row['catalog_content']
        image_link = row['image_link']
        value = row['price']

        image_name = image_link.split('/')[-1]
        image_path = os.path.join(self.images_path, image_name)

        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            # print(f"Warning: Failed to load image sample {row['sample_id']} from {image_path}: {e}")
            try:
                response = requests.get(image_link, timeout=10)
                response.raise_for_status()
                image = Image.open(BytesIO(response.content)).convert('RGB')
            except Exception as e:
                print(f"Warning: Failed to download image sample {row['sample_id']} from {image_link}: {e}")
                image = torch.zeros((224, 224, 3))

        sample = {
            'text': text,
            'image': image,
            'target': torch.tensor(value, dtype=torch.float32)
        }
        return sample



def collate_fn(batch):
    texts = [item['text'] for item in batch]
    images = ([item['image'] for item in batch])
    targets = torch.stack([item['target'] for item in batch])
    targets = torch.reshape(targets, shape= (-1, 1))
    
    return {
        'texts': texts,
        'images': images,
        'targets': targets
    }


# def test_dataset():
#     current_dir = os.getcwd()
#     csv_path = os.path.join(current_dir, 'dataset', 'train_transformed_sep.csv')
#     # image_path = os.path.join(current_dir, 'images')

#     dataloader = DataLoader(
#         AmazonDataset(csv_path=csv_path),
#         batch_size=4,
#         shuffle=False,
#         num_workers=0,
#         collate_fn=collate_fn
#     )
    
#     for batch in dataloader:
        
#         print(batch.get("targets"))
#         exit()
# if __name__ == '__main__':
#     test_dataset()
