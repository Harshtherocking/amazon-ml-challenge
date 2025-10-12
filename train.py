import torch 
from torch import nn
from torch import optim
from model import LinearRegressionHead, IMAGE_PROCESSOR, IMAGE_ENCODER, TEXT_ENCODER, TEXT_TOKENIZER
from PIL import Image
import requests
import os
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
import time
import io

from dataset import AmazonDataset, collate_fn
from torch.utils.data import DataLoader



def get_text_vision_emb(texts, images):
    text_inputs = TEXT_TOKENIZER(texts,  padding = True, truncation = True, return_tensors = "pt")
    image_inputs = IMAGE_PROCESSOR(images, return_tensors="pt")

    text_out = TEXT_ENCODER(**text_inputs)
    image_out = IMAGE_ENCODER(**image_inputs)

    text_emb = text_out.last_hidden_state[:, 0, :]
    image_emb = image_out.last_hidden_state[:, 0, :]
    return text_emb, image_emb



def train_batch(dataloader, model, device=None, epochs=3, lr=1e-4, log_dir='runs'):
    device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    criterion = nn.MSELoss()

    writer = SummaryWriter(log_dir=os.path.join(log_dir, time.strftime('%Y%m%d-%H%M%S')))

    global_step = 0
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_samples = 0

        print("Epoch Training started")
        for batch in dataloader:
            texts = batch.get('texts')
            images = batch.get('images')
            targets = batch.get('targets')
            
            text_embs, image_embs = get_text_vision_emb(texts, images)


            text_batch = text_embs.to(device)
            image_batch = image_embs.to(device)


            preds = model(text_batch, image_batch)
            loss = criterion(preds, targets)
            print(loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_n = text_batch.shape[0]
            epoch_loss += loss.item() * batch_n
            n_samples += batch_n
            writer.add_scalar('train/loss', loss.item(), global_step)
            global_step += 1

        print("Epoch Training completed")
        epoch_loss = epoch_loss / max(1, n_samples)
        writer.add_scalar('train/epoch_loss', epoch_loss, epoch)
        print(f'Epoch {epoch+1}/{epochs} - loss: {epoch_loss:.4f}')

        scheduler.step()
        writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], epoch)

    writer.close()


if __name__ == "__main__" : 
    csv_path = '/content/drive/MyDrive/amazon/amazon-ml-challenge/dataset/train_transformed_sep.csv'
    image_path = '/content/drive/MyDrive/amazon/images/train_images'

    dataloader = DataLoader(
        AmazonDataset(csv_path=csv_path),
        batch_size=4,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )
    print("DataLoader loaded")

    model = LinearRegressionHead(IMAGE_ENCODER.config.hidden_size, TEXT_ENCODER.config.hidden_size)

    train_batch(dataloader,model, epochs= 1)
