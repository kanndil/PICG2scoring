import torch
import yaml
from torch.utils.data import Dataset
from PIL import Image
import json
import llama.utils
from llama import Tokenizer
import copy
import torchvision.transforms as transforms
import pandas as pd
import random
import cv2
import numpy as np

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}

## create data
#transform_train = transforms.Compose([
#    transforms.RandomResizedCrop(size=(224, 224), scale=(0.9, 1.0), ratio=(0.75, 1.3333), interpolation=BICUBIC,
#                                 antialias=None),  # 3 is bicubic
#    transforms.ToTensor(),
#    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])])




# Define the transformation pipeline
transform_train = transforms.Compose([
    # Random rotation: rotate the image by a random angle within the given range (-degrees, +degrees)
    transforms.RandomRotation(degrees=30, interpolation=InterpolationMode.BICUBIC),

    
    # Horizontal flip: randomly flip the image along the vertical axis
    transforms.RandomHorizontalFlip(p=0.5),  # p is the probability of flipping
    
    # Vertical flip: randomly flip the image along the horizontal axis
    transforms.RandomVerticalFlip(p=0.5),  # p is the probability of flipping
    
    # Random Resized Crop: resize and crop the image to a random size
    transforms.RandomResizedCrop(size=(224, 224), scale=(0.9, 1.0), ratio=(0.75, 1.3333), interpolation=InterpolationMode.BICUBIC),
    
    # Random translation (shifting) of the image in both horizontal and vertical directions
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # translate=(x_fraction, y_fraction), e.g., 0.1 means a 10% shift
    
    # Convert the image to a tensor
    transforms.ToTensor(),
    
    # Normalize the image with the given mean and std values for RGB channels
    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
])

class FinetuneDataset(Dataset):
    def __init__(self, config_path, transform, max_words=30, tokenizer_path=None):
        print(f"read dataset config from {config_path}")
        with open(config_path, 'r') as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
        print("DATASET CONFIG:")
        print(self.config)
        ann = []
        for meta_path in self.config['META']:
            meta_l = json.load(open(meta_path))
            print(f"{meta_path}: len {len(meta_l)}")
            ann += meta_l
        self.ann = ann
        print(f"total length: {len(self)}")
        self.transform = transform
        self.max_words = max_words
        self.tokenizer = Tokenizer(model_path=tokenizer_path)

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        data_item = self.ann[index]
        if 'image' in data_item.keys():
            filename = data_item['image']
            question = data_item['conversations'][0]['value']
            answer = data_item['conversations'][1]['value']
     
            image = cv2.imread(filename)
            image = Image.fromarray(image)
            image = self.transform(image)
            format_instruction = question
            format_input = None
        else:
            image = torch.zeros(3, 224, 224)
            format_instruction = data_item['instruction'],
            format_input = data_item['input']
            answer = data_item['output']
        input1 = llama.utils.format_prompt(format_instruction, format_input)
        input2 = input1 + answer
        input1 = torch.tensor(self.tokenizer.encode(input1, bos=True, eos=False), dtype=torch.int64)
        input2 = torch.tensor(self.tokenizer.encode(input2, bos=True, eos=True), dtype=torch.int64)
        padding = self.max_words - input2.shape[0]
        if padding > 0:
            input2 = torch.cat((input2, torch.zeros(padding, dtype=torch.int64) - 1))
        elif padding < 0:
            input2 = input2[:self.max_words]
        labels = copy.deepcopy(input2)
        labels[:len(input1)] = -1
        input2_mask = input2.ge(0)
        label_mask = labels.ge(0)
        input2[~input2_mask] = 0
        labels[~label_mask] = 0
        input2_mask = input2_mask.float()
        label_mask = label_mask.float()
        return input2, labels, input2_mask, image

class ProstateDataset(FinetuneDataset):
    '''
    Loading the prosatate data
    there are three MRI modalities
    T2W, DWI, ADC
    '''
    def __init__(self, config_path, transform, max_words=300, tokenizer_path=None):
        super().__init__(config_path, transform, max_words, tokenizer_path)
        tf = torch.nn.Upsample(size=(84, 84, 98))
        # tf = torch.nn.Upsample(size=(84, 98, 42))
        # numpy -> tensor
        # resize (resize + center crop)
        # set channel_num -> 1
        # normalize
        avg, std = 0.48, 0.27
        self.transform = lambda x: (tf(torch.from_numpy(x).float()[None, None, ...]) / 255. - avg) / std


    def __getitem__(self, index):
        data_item = self.ann[index]
        img = np.load(data_item['img'])
        # t2w = np.load(data_item['T2W'])
        # dwi = np.load(data_item['DWI'])
        # adc = np.load(data_item['ADC'])
        # format these images
        # --> resize them
        # t2w = self.transform(t2w).squeeze(0)
        # dwi = self.transform(dwi).squeeze(0)
        # adc = self.transform(adc).squeeze(0)    
        img = self.transform(img).squeeze(0)

        format_instruction = data_item['instruction']
        format_input = data_item['input']
        answer = data_item['output']
        input1 = llama.utils.format_prompt(format_instruction, format_input)
        input2 = input1 + answer
        input1 = torch.tensor(self.tokenizer.encode(input1, bos=True, eos=False), dtype=torch.int64)
        input2 = torch.tensor(self.tokenizer.encode(input2, bos=True, eos=True), dtype=torch.int64)
        padding = self.max_words - input2.shape[0]
        if padding > 0:
            input2 = torch.cat((input2, torch.zeros(padding, dtype=torch.int64) - 1))
        elif padding < 0:
            input2 = input2[:self.max_words]
        labels = copy.deepcopy(input2)
        labels[:len(input1)] = -1
        input2_mask = input2.ge(0)
        label_mask = labels.ge(0)
        input2[~input2_mask] = 0
        labels[~label_mask] = 0
        input2_mask = input2_mask.float()
        label_mask = label_mask.float()
        return input2, labels, input2_mask, img

class PretrainDataset(Dataset):
    def __init__(self, config_path, transform, max_words=30, tokenizer_path=None):
        print(f"read dataset config from {config_path}")
        with open(config_path, 'r') as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
        print("DATASET CONFIG:")
        print(self.config)
        images, captions = [], []
        for meta_path in self.config['META']:
            images_this_meta, captions_this_meta = [], []
            for chunk in pd.read_csv(meta_path, sep='\t', lineterminator='\n', chunksize=10 ** 6):
                images_this_meta.extend(chunk['url'].tolist())
                captions_this_meta.extend(chunk['caption'].tolist())
            print(f"{meta_path}: len {len(images_this_meta)}")
            images.extend(images_this_meta)
            captions.extend(captions_this_meta)

        self.data_list = []
        for x, y in zip(images, captions):
            self.data_list.append({'url': x, 'caption': y})
        print(f"total length: {len(self)}")
        self.transform = transform
        self.max_words = max_words
        self.tokenizer = Tokenizer(model_path=tokenizer_path)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        sample = self.data_list[index]
        image_path, caption = sample['url'], sample['caption']
        if isinstance(caption, list):
            caption = random.choice(caption)
        caption = str(caption)

        image = cv2.imread(image_path)
        image = Image.fromarray(image)
        image = self.transform(image)

        format_instruction = "Generate caption of this image"
        input1 = llama.utils.format_prompt(format_instruction, None)
        input2 = input1 + caption

        input1 = torch.tensor(self.tokenizer.encode(input1, bos=True, eos=False), dtype=torch.int64)
        input2 = torch.tensor(self.tokenizer.encode(input2, bos=True, eos=True), dtype=torch.int64)
        padding = self.max_words - input2.shape[0]
        if padding > 0:
            input2 = torch.cat((input2, torch.zeros(padding, dtype=torch.int64) - 1))
        elif padding < 0:
            input2 = input2[:self.max_words]
        labels = copy.deepcopy(input2)
        labels[:len(input1)] = -1
        input2_mask = input2.ge(0)
        label_mask = labels.ge(0)
        input2[~input2_mask] = 0
        labels[~label_mask] = 0
        input2_mask = input2_mask.float()
        label_mask = label_mask.float()
        return input2, labels, input2_mask, image
