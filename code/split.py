"""
preprocess the MS COCO/CC-3M dataset
The split of MS COCO dataset can be referred to as: https://github.com/karpathy/neuraltalk2/issues/192 
"""

import argparse 
import json 
import os 

def static_coco(annotation_path): 

    with open(annotation_path, 'r') as f: 
        data_dict = json.load(f) 
    
    print(data_dict['images'][0])
    print('total images: ', len(data_dict['images']))

    num_train = 0 
    num_val = 0 
    num_test = 0 

    sentence_length_sum = 0 
    sentence_num = 0 
    for data in data_dict['images']: 
        if data['split'] == 'train': 
            num_train += 1 
            for sentence in data['sentences']: 
                sentence_length_sum += len(sentence['tokens']) 
                sentence_num += 1 
        elif data['split'] == 'val':
            num_val += 1 
        elif data['split'] == 'test':
            num_test += 1 

    print('train: ', num_train, ' val: ', num_val, ' test: ', num_test) 
    print('average length: ', float(sentence_length_sum / sentence_num))

def preprocess_coco(dataset_path, annotation_path, split='train'): 
    with open(annotation_path, 'r') as f: 
        data_dict = json.load(f) 
    
    data_list = [] 
    for data in data_dict['images']: 
        if data['split'] != split: 
            continue 
        image_path = os.path.join(dataset_path, data['filepath']) 
        image_path = os.path.join(image_path, data['filename']) 
        
        for sentence in data['sentences']: 
            data_list.append(
                {
                    'image': image_path, 
                    'text': sentence['raw'],
                }
            )
    print('image-text pair: ', len(data_list))
    return data_list 


def main(): 
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_type", type=str, default='COCO', help="COCO or CC-3M") 
    parser.add_argument("--dataset_path", type=str, default='/Users/feizhengcong/Desktop/COCO', help="path for dataset") 
    parser.add_argument("--annotation_path", type=str, default='/Users/feizhengcong/Desktop/COCO/dataset_coco.json', help="path for annotation file")
    args = parser.parse_args() 

    if args.dataset_type == 'COCO':
        static_coco(args.annotation_path) 
        preprocess_coco(args.dataset_path, args.annotation_path, 'train')
    else:
        pass 

if __name__ == "__main__":
    main()

