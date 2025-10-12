from transformers import ViTImageProcessor, ViTModel
import torch
from torch import nn
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BertModel, BertTokenizer
from sentence_transformers import SentenceTransformer

IMAGE_PROCESSOR = ViTImageProcessor.from_pretrained('google/vit-large-patch16-224-in21k')
IMAGE_ENCODER = ViTModel.from_pretrained('google/vit-large-patch16-224-in21k', device_map="auto" )
# !pip install transformers torch accelerate flash-attn

TEXT_ENCODER = BertModel.from_pretrained('bert-base-uncased', device_map = "auto")
TEXT_TOKENIZER = BertTokenizer.from_pretrained("bert-base-uncased")


# class Regression_head (nn.Module) : 
#     def __init__(self, image_dim : int, text_dim : int, hid_dim : int, *args, **kwargs) -> None:
#         super().__init__(*args, **kwargs)
#         self.image_dim = image_dim
#         self.text_dim = text_dim
#         self.hid_dim = hid_dim

#         self.Wq = nn.Linear(in_features = text_dim, out_feature = hid_dim) 
#         self.Wk = nn.Linear(in_features = image_dim, out_features= hid_dim)
#         self.Wv = nn.Linear(in_features= image_dim, out_features= hid_dim)

#     def cross_attention (self, image_emd, text_emd)  :

#         q = self.Wq(text_emd)
#         # 
#         k = self.Wk(image_emd)
#         v = self.Wv(image_emd)

#         score = torch.softmax(q.T * k, dim = 1)
        
#         pass


class LinearRegressionHead(nn.Module) : 
    def __init__(self, image_dim : int, text_dim : int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.image_dim = image_dim
        self.text_dim = text_dim
        # self.hid_dim = hid_dim

        self.W1 = nn.Linear(in_features= text_dim + image_dim , out_features= 1024)
        self.W2 = nn.Linear(1024, 512)
        self.W3 = nn.Linear(512, 256)
        self.W4 = nn.Linear(256, 1)

        self.sigmoid = nn.Sigmoid()

    
    def forward(self, text_emd, image_emd) : 
        comb_emd = torch.concat((text_emd,image_emd), dim =1)
        x = self.W1(comb_emd)
        x = self.W2(x)
        x = self.W3(x)
        x = self.W4(x)
        
        return self.sigmoid(x)

