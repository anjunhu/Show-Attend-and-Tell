import os
import re
import argparse, json
from collections import Counter


def generate_json_data(split_path, data_path, max_captions_per_image, min_word_count):
    # use the coco word dict
    with open('./data/coco/word_dict_coco.json', 'r') as f:
        word_dict = json.load(f)

    # just get the coco max lengths
    with open('./data/coco/train_captions.json', 'r') as f:
        train_captions = json.load(f)
    max_length = len(train_captions[0])

    train_img_paths = []
    train_caption_tokens = []
    validation_img_paths = []
    validation_caption_tokens = []
    
    for split in ['train', 'test']:
        annotations = json.load(open(f'./data/iot/coco_format_{split}_from_csv.json', 'r'))
        word_count = Counter()
        for img in annotations['images']:
            # print(img)
            caption_count = 0
            for sentence in img['sentences']:
                if caption_count < max_captions_per_image:
                    caption_count += 1
                else:
                    break

                if 'train' in split:
                    train_img_paths.append(os.path.join(data_path, img['filepath'], img['filename']))
                    train_caption_tokens.append(sentence['tokens'])
                elif 'test' in split:
                    validation_img_paths.append(os.path.join(data_path, img['filepath'], img['filename']))
                    validation_caption_tokens.append(sentence['tokens'])
                max_length = max(max_length, len(sentence['tokens']))
                word_count.update(sentence['tokens'])
    # print(train_img_paths, train_caption_tokens)
    # print(validation_img_paths, validation_caption_tokens)

    train_captions = process_caption_tokens(train_caption_tokens, word_dict, max_length)
    validation_captions = process_caption_tokens(validation_caption_tokens, word_dict, max_length)

    with open('./data/cocodict-iot/train_img_paths.json', 'w') as f:
        json.dump(train_img_paths, f)
    with open('./data/cocodict-iot/val_img_paths.json', 'w') as f:
        json.dump(validation_img_paths, f)
    with open('./data/cocodict-iot/train_captions.json', 'w') as f:
        json.dump(train_captions, f)
    with open('./data/cocodict-iot/val_captions.json', 'w') as f:
        json.dump(validation_captions, f)


def process_caption_tokens(caption_tokens, word_dict, max_length):
    captions = []
    missing_words = []
    for tokens in caption_tokens:
        token_idxs = []   
        for token in tokens:  
            token = token.lower()
            token = re.sub(r'[^a-zA-Z]', '', token)
            if token in word_dict: 
                token_idxs.append(word_dict[token])  
            else:
                missing_words.append(token)
                token_idxs.append(word_dict['<unk>'])  


        captions.append(
            [word_dict['<start>']] + token_idxs + [word_dict['<eos>']] +
            [word_dict['<pad>']] * (max_length - len(tokens)))
    missing_words = set(missing_words)
    print(missing_words)
    return captions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate json files')
    parser.add_argument('--split-path', type=str, default='data/coco/dataset.json')
    parser.add_argument('--data-path', type=str, default='/home/anhu/.computer_vision_dataset_cache/')
    parser.add_argument('--max-captions', type=int, default=5,
                        help='maximum number of captions per image')
    parser.add_argument('--min-word-count', type=int, default=5,
                        help='minimum number of occurences of a word to be included in word dictionary')
    args = parser.parse_args()

    generate_json_data(args.split_path, args.data_path, args.max_captions, args.min_word_count)
