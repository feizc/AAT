import torch
import torch.nn as nn
from transformers import ViTModel, GPT2LMHeadModel 
from typing import Tuple


class MLP(nn.Module):
    def __init__(self, sizes: Tuple[int, ...], bias=True, act=nn.Tanh):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
            if i < len(sizes) - 2:
                layers.append(act())
        self.model = nn.Sequential(*layers) 
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class TransformerImageCaption(nn.Module): 
    def __init__(self, vision_config, language_config):
        super().__init__()
        self.vision_encoder = ViTModel(vision_config, add_pooling_layer=False, use_mask_token=False)
        self.language_decoder = GPT2LMHeadModel(language_config) 
    
    def forward(self, image, text, mask): 
        encoder_outputs = self.vision_encoder(pixel_values=image).last_hidden_state 
        decoder_outputs = self.language_decoder(input_ids=text, 
                                                attention_mask=mask, 
                                                encoder_hidden_states=encoder_outputs)
        return decoder_outputs 
        
