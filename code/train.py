import os 
import torch 
from torch import optim 
import argparse 
from torch.utils.data import DataLoader 
from torch.nn import CrossEntropyLoss
from tqdm import tqdm 
from transformers import ViTFeatureExtractor, GPT2Tokenizer 

from split import preprocess_coco 
from dataset import COCODataset 
from model import VisionEncoderConfig, LanguageDecoderConfig, TransformerImageCaption, AttenAlignTransformer 

SPECIAL_TOKENS = ["[bos]", "[eos]",] 
SPECIAL_TOKENS_DICT = {'bos_token': "[bos]", 'eos_token': "[eos]"}


def shift_cross_entropy_loss(logits, labels): 
    shift_logits = logits[..., :-1, :].contiguous() 
    shift_labels = labels[..., 1:].contiguous()
    # Flatten the tokens
    loss_fct = CrossEntropyLoss(ignore_index=0)
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    return loss 


def main(): 
    parser = argparse.ArgumentParser() 
    parser.add_argument("--vit_path", type=str, default='./ckpt/vit') 
    parser.add_argument("--gpt2_path", type=str, default='./ckpt/gpt2') 
    parser.add_argument("--dataset_path", type=str, default='/Users/feizhengcong/Desktop/COCO') 
    parser.add_argument("--annotaion_path", type=str, default='/Users/feizhengcong/Desktop/COCO/dataset_coco.json' ) 
    parser.add_argument("--save_path", type=str, default='./ckpt') 

    parser.add_argument("--use_atten_align", type=bool, default=True) 
    parser.add_argument("--batch_size", type=int, default=4) 
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate.")
    parser.add_argument("--beta1", type=float, default=0.98, help="Adam beta 1.")
    parser.add_argument("--beta2", type=float, default=0.999, help="Adam beta 2.")
    parser.add_argument("--eps", type=float, default=1e-6, help="Adam epsilon.")
    parser.add_argument("--workers", type=int, default=1, help="Number of dataloader workers per GPU.")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs.") 
    parser.add_argument("--debug", type=bool, default=True) 
    args = parser.parse_args()


    if torch.cuda.is_available():
        # This enables tf32 on Ampere GPUs which is only 8% slower than
        # float16 and almost as accurate as float32
        # This was a default in pytorch until 1.12
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    device = "cuda:0" if torch.cuda.is_available() else "cpu"  

    image_process = ViTFeatureExtractor.from_pretrained(args.vit_path)
    tokenizer = GPT2Tokenizer.from_pretrained(args.gpt2_path) 
    tokenizer.add_special_tokens(SPECIAL_TOKENS_DICT) 

    train_data = preprocess_coco(args.dataset_path, args.annotaion_path, 'train') 
    val_data = preprocess_coco(args.dataset_path, args.annotaion_path, 'val') 
    train_dataset = COCODataset(data=train_data, image_preprocess=image_process, tokenizer=tokenizer)
    val_dataset = COCODataset(data=val_data, image_preprocess=image_process, tokenizer=tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=args.workers)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    vision_config = VisionEncoderConfig()
    language_config = LanguageDecoderConfig() 
    language_config.add_cross_attention = True
    language_config.vocab_size = len(tokenizer) 
    if args.use_atten_align == True: 
        model = AttenAlignTransformer(vision_config=vision_config, language_config=language_config)
    else:
        model = TransformerImageCaption(vision_config=vision_config, language_config=language_config)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
    )

    for epoch in range(args.epochs):
        model.train()
        loss_cum = .0 
        progress = tqdm(total=len(train_loader), desc='training') 
        for i, batch in enumerate(train_loader):
            image, tokens, mask = batch 
            image = image.to(device)
            tokens = tokens.to(device) 
            mask = mask.to(device) 

            output = model(image, tokens, mask).logits
            loss = shift_cross_entropy_loss(output, tokens) 
            loss.backward() 
            optimizer.step() 
            optimizer.zero_grad()
            loss_cum += loss.item() 
            progress.set_postfix({"loss": loss_cum / (i + 1)})
            progress.update()
            if args.debug == True:
                break 
        
        model.eval() 
        with torch.no_grad(): 
            progress = tqdm(total=len(val_loader), desc='evaluation') 
            loss_cum = .0 
            for i, batch in enumerate(val_loader):
                image, tokens, mask = batch 
                image = image.to(device)
                tokens = tokens.to(device) 
                mask = mask.to(device) 

                output = model(image, tokens, mask).logits
                loss = shift_cross_entropy_loss(output, tokens) 
                loss_cum += loss.item() 
                progress.set_postfix({"loss": loss_cum / (i + 1)})
                progress.update()
                if args.debug == True:
                    break 
        
        torch.save(model.state_dict(), os.path.join(args.save_path, str(epoch) + '.pt'))

        if args.debug == True:
            break 
        

if __name__ == "__main__":
    main() 