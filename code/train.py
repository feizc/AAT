import torch 
from torch.utils.data import DataLoader
from transformers import ViTFeatureExtractor, GPT2Tokenizer 

from split import preprocess_coco 
from dataset import COCODataset 
from model import VisionEncoderConfig, LanguageDecoderConfig, TransformerImageCaption


def main(): 
    if torch.cuda.is_available():
        # This enables tf32 on Ampere GPUs which is only 8% slower than
        # float16 and almost as accurate as float32
        # This was a default in pytorch until 1.12
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    vit_path = './ckpt/vit' 
    gpt2_path = './ckpt/gpt2'
    dataset_path = '/Users/feizhengcong/Desktop/COCO'
    annotaion_path = '/Users/feizhengcong/Desktop/COCO/dataset_coco.json' 
    batch_size = 1

    image_process = ViTFeatureExtractor.from_pretrained(vit_path)
    tokenizer = GPT2Tokenizer.from_pretrained(gpt2_path)
    train_data = preprocess_coco(dataset_path, annotaion_path, 'train') 
    train_dataset = COCODataset(data=train_data, image_preprocess=image_process, tokenizer=tokenizer)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    vision_config = VisionEncoderConfig()
    language_config = LanguageDecoderConfig() 
    language_config.add_cross_attention = True
    model = TransformerImageCaption(vision_config=vision_config, language_config=language_config)

    for item in train_dataloader: 
        image, tokens, mask = item
        print(image.size())
        print(mask) 
        print(tokens)
        output = model(image, tokens, mask).logits
        print(output.size())
        break 

if __name__ == "__main__":
    main() 