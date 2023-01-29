import torch 
from PIL import Image 

import torch.nn as nn
from torch.utils.data import Dataset 


SPECIAL_TOKENS = ["[bos]", "[eos]",] 
SPECIAL_TOKENS_DICT = {'bos_token': "[bos]", 'eos_token': "[eos]"}


def tokenize(obj, tokenizer):
    if isinstance(obj, str):
        # return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))
        return tokenizer.encode(obj)
    if isinstance(obj, dict):
        return dict((n, tokenize(o)) for n, o in obj.items())
    return list(tokenize(o) for o in obj) 


class COCODataset(Dataset): 
    def __init__(self, data, image_preprocess, tokenizer, max_length=30): 
        self.data = data 
        self.image_preprocess = image_preprocess 
        self.tokenizer = tokenizer 
        self.max_length = max_length 
        self.bos, self.eos = self.tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS) 
    
    def __len__(self): 
        return len(self.data) 

    def __getitem__(self, index):
        data_pair = self.data[index] 
        image = Image.open(data_pair['image']) 
        image = self.image_preprocess(image, return_tensors='pt')["pixel_values"]
        image = image.squeeze(0)

        text = [self.bos] + tokenize(data_pair['text'], self.tokenizer) + [self.eos]
        # print(self.tokenizer.decode(text))
        text = torch.tensor(text, dtype=torch.int64) 
        tokens, mask = self.pad_tokens(text)
        return image, tokens, mask 

    def pad_tokens(self, tokens): 
        padding = self.max_length - tokens.shape[0]
        if padding > 0:
            tokens = torch.cat((tokens, torch.zeros(padding, dtype=torch.int64) - 1))
        elif padding < 0:
            tokens = tokens[:self.max_length]
        mask = tokens.ge(0)  # mask is zero where we out of sequence
        tokens[~mask] = 0
        mask = mask.float() 
        return tokens, mask 
