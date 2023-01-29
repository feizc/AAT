"""
    compare the FLOPs and params for image captioning models
"""

import torch 
from torch.utils.data import DataLoader
from transformers import ViTFeatureExtractor, GPT2Tokenizer 
from thop import profile 
from model import VisionEncoderConfig, LanguageDecoderConfig, TransformerImageCaption


def main(): 
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    vision_config = VisionEncoderConfig()
    language_config = LanguageDecoderConfig() 
    language_config.add_cross_attention = True
    model = TransformerImageCaption(vision_config=vision_config, language_config=language_config)

    image = torch.randn(1, 3, 224, 224)
    token = torch.ones(1, 11).long()
    mask = torch.ones(1, 11).long() 
    
    flops, params = profile(model, inputs=(image, token, mask)) 
    print('FLOPs: ', flops)
    print('Params: ', params) 

    flops, params = profile(model.vision_encoder, inputs=(image,)) 
    print('FLOPs: ', flops)
    print('Params: ', params) 

if __name__ == "__main__":
    main() 