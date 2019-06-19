#coding: utf-8
from __future__ import division
import os
import math
import time
import random
import torch
import numpy as np
import torchvision.transforms as transforms
import cv2
from tqdm import tqdm
from models.crnn import DenseNet
from utils.utils import CTCLabelConverter

class CRNN_OCR(object):
    def __init__(self, model_path, alphabet, image_size, device=None):
        self._alphabet = alphabet
        self._image_size = image_size
        self._device = device
        self._transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,))])

        alphabet = ''
        with open(self._alphabet, mode='rb') as f:
            for line in f.readlines():
                alphabet += line.decode('utf-8')[0]
        self._alphabet = alphabet
        self._converter = CTCLabelConverter(self._alphabet, ignore_case=False)

        num_classes = len(self._alphabet) + 1
        num_channels = 1
        self._model = DenseNetRNN(num_channels=num_channels, num_classes=num_classes,
                                  rnn=True, num_hidden=256,
                                  growth_rate=12, block_config=(3,6,9),#(3,6,12,16),
                                  compression=0.5, num_init_features=64, bn_size=4, drop_rate=0, small_inputs=True, efficient=False)
        for param in self._model.parameters():
            param.requires_grad = False

        state_dict = torch.load(model_path, map_location=device)
        self._model.load_state_dict(state_dict['state_dict'])
        self.mode = self._model.to(self._device)
        self._model.eval()

    def predict(self, image):
        assert image.ndim == 2
        width, height = image.shape[1], image.shape[0]
        scale = self._image_size[0] / height
        width = int(round(width*scale))
        width = min(width, self._image_size[1])
        image = cv2.resize(np.array(image),(width, self._image_size[0]),interpolation=cv2.INTER_CUBIC)
        # random padding
        left = random.randint(0, self._image_size[1]-width)
        right = self._image_size[1] - width - left
        image = cv2.copyMakeBorder(image, 0, 0, left, right, cv2.BORDER_REPLICATE)

        image = self._transform(image)
        image = torch.unsqueeze(image, 0).to(self._device)   # B, C, H, W
        with torch.set_grad_enabled(False):
            _, pred = self._model(image)
        score, pred_index = pred.max(2)

        score = score.contiguous().view(-1)
        pred_index = pred_index.contiguous().view(-1)

        score = score.data.cpu()
        pred_index = pred_index.data.cpu()

        text, valid_index = self._converter.decode(pred_index.data, torch.IntTensor([pred_index.size(0)]), raw=False)
        return text

if __name__ == '__main__':
    test_root = '/home/oliverjwliu/datasets/gangao/train_datasets/'
    model_path = './checkpoints/densenet_epoch_14_step_75_acc_97.49%_loss_0.449914.pth.tar'
    chars_file = './chars/char_20868.txt'
    label_file = './experiments/val_labels.txt'
    model = CRNN_OCR(model_path, chars_file, image_size=(32, 650), device='cuda')
    with open(label_file, 'rb') as file_reader:
        lines = file_reader.readlines()

    TP = 0
    count = 0
    total_time = 0.0
    for line in tqdm(lines):
        line = line.decode('utf-8').strip('\r\n').split(' ')
        image_path = line[0]
        label = ' '.join(line[1:])
        image = cv2.imread(os.path.join(test_root, image_path), cv2.IMREAD_GRAYSCALE)
        start = time.time()
        text = model.predict(image)
        end = time.time()
        total_time += (end - start)
        count += 1
        if text == label:
            TP += 1
        else:
            print('\n'+'='*60)
            print(text)
            print(label)
    print('accuracy: {}, {}/{}'.format(TP/count, TP, count))
    print('avg time: {}s, {}s/{}'.format(total_time/count, total_time, count))
