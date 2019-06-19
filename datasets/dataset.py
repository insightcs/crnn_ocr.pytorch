# encoding: utf-8
from __future__ import division
import os
import cv2
import torch
import lmdb
import math
import random
import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms

def randnoise(img):
    """
    给图片随机加不同的模糊
    """
    randnum=random.randint(1,3)
    if randnum==1:
        img1 = cv2.GaussianBlur(img, (9,9), 0)
    elif randnum==2:
        randblur=random.randint(1,7)            
        img1=cv2.blur(img, (1, randblur))
    elif randnum==3:
        randblur=random.randint(1,7)
        img1=cv2.blur(img, (randblur,1))
    return img1

def rotate_image(image, angle, scale=1.):
    if angle == 0:
        return image
    w = image.shape[1]
    h = image.shape[0]
    randangle = random.randint(-angle, angle)
    rangle = np.deg2rad(randangle)  # angle in radians
    nw = (abs(np.sin(rangle)*h) + abs(np.cos(rangle)*w))*scale
    nh = (abs(np.cos(rangle)*h) + abs(np.sin(rangle)*w))*scale
    rot_mat = cv2.getRotationMatrix2D((nw*0.5, nh*0.5), randangle, scale)
    rot_move = np.dot(rot_mat, np.array([(nw-w)*0.5, (nh-h)*0.5,0]))
    rot_mat[0,2] += rot_move[0]
    rot_mat[1,2] += rot_move[1]
    return cv2.warpAffine(image, rot_mat, (int(math.ceil(nw)), int(math.ceil(nh))), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REPLICATE)


class LmdbDataset(data.Dataset):

    def __init__(self, root, image_size, transform=None, target_transform=None):
        self.env = lmdb.open(
            root,
            max_readers=1,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False)

        if not self.env:
            print('cannot creat lmdb from %s' % (root))
            sys.exit(0)

        with self.env.begin(write=False) as txn:
            nSamples = int(txn.get('num-samples'))
            self.nSamples = nSamples

        self._image_size = image_size
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        index += 1
        with self.env.begin(write=False) as txn:
            img_key = 'image-%09d' % index
            imgbuf = txn.get(img_key)

            try:
                imageBuf = np.fromstring(imgbuf, dtype=np.uint8)
                image = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
                # data augmention
                randnum=random.randint(0,1)
                if randnum == 0:
                    image = randnoise(image.copy())

                randnum = random.randint(0,1)
                if randnum == 0:
                    image = rotate_image(image.copy(), 2)
                
                #image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

                width, height = image.shape[1], image.shape[0]
                scale = height * 1.0 / self._image_size[0]
                new_width=int(round(float(width/scale)/16)*16)
                width = min(new_width, self._image_size[1])

                image = cv2.resize(np.array(image),(width, self._image_size[0]),interpolation=cv2.INTER_CUBIC).astype(np.float32)

                image = image / 255.0 - 0.5
                padding_image = np.zeros((self._image_size[0], self._image_size[1]), dtype=np.float32)
                randnum = random.randint(0,1)
                if randnum == 0:
                    padding_image[:image.shape[0], self._image_size[1]-image.shape[1]:] = image
                else:
                    padding_image[:image.shape[0], :image.shape[1]] = image  
                #image = Image.fromarray(padding_image)
            except Exception as e:
                print('invalid image for {}, {}'.format(index, e))
                return self[index + 1]

            if self.transform is not None:
                image = self.transform(image)

            label_key = 'label-%09d' % index
            label = str(txn.get(label_key))

            if self.target_transform is not None:
                label = self.target_transform(label)

        return (image, label)

class Dataset(data.Dataset):
    """Digits dataset."""
    def __init__(self, mode, data_root, transform=None):
        if not (mode == 'train' or mode == 'val'):
            raise(RuntimeError("MODE should be either train or dev, passed as string"))

        self.mode = mode
        self.transform = transform
        self.img_root = data_root#os.path.join(data_root, 'images')
        self.img_names = []
        self.targets = []

        label_path = os.path.join(data_root, '{}_labels.txt'.format(self.mode))
        with open(label_path, 'rb') as f:
            lines = f.readlines()

        for _ in range(5):
            random.shuffle(lines)

        for line in lines:
            line = line.decode('utf-8').strip('\r\n').split(' ')
            self.img_names.append(line[0])
            self.targets.append(' '.join(line[1:]))   #处理字符集中含有空格的情况

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        try:
            img_name = os.path.join(self.img_root, self.img_names[idx])
            image_buff = np.fromfile(img_name, dtype=np.uint8)
            image = cv2.imdecode(image_buff, cv2.IMREAD_GRAYSCALE)
            image_shape = image.shape
            target = self.targets[idx]
        except Exception as e:
            print('invalid image for {}, {}'.format(self.img_names[idx], e))
            return self[idx+1]

        # data augmention for real images
        if self.mode == 'train' and self.img_names[idx].startswith('real'):
            randnum=random.randint(0,1)
            if randnum == 0:
                image = randnoise(image)

            randnum = random.randint(0,1)
            if randnum == 0:
                image = rotate_image(image, 2)

        if self.transform is not None:
            image = self.transform(image)
        return image, target, image_shape

class Preprocessing(object):
    def __init__(self, image_size, mode, interpolation=cv2.INTER_CUBIC):
        self.image_size = image_size
        self.mode = mode
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=(0.5,), std=(0.5,))

    def __call__(self, image):
        width, height = image.shape[1], image.shape[0]
        scale = self.image_size[0] / height
        width = int(round(width*scale))
        width = min(width, self.image_size[1])
        image = cv2.resize(np.array(image),(width, self.image_size[0]),interpolation=self.interpolation)

        # random padding
        if self.mode == 'train':
            left = random.randint(0, self.image_size[1]-width)
        else:
            left = 0
        right = self.image_size[1] - width - left
        image = cv2.copyMakeBorder(image, 0, 0, left, right, cv2.BORDER_CONSTANT, value=(np.mean(image),))
        image = self.toTensor(image)
        image = self.normalize(image)
        return image

class DatasetCollater(object):
    """Digits Collater."""
    def __init__(self, image_size, mode, keep_ratio=False):
        self.image_size = image_size
        self.mode = mode
        self.keep_ratio = keep_ratio

    def __call__(self, batch):
        images, targets, images_shape = zip(*batch)
        ratio = []
        for shape in images_shape:
            h, w = shape[0:2]
            ratio.append(w/h)
        max_ratio = np.max(ratio)
        image_h = self.image_size[0]
        if self.keep_ratio:
            image_w = int(np.round(max_ratio * self.image_size[0]))
        else:
            image_w = self.image_size[1]

        transform = Preprocessing((image_h, image_w), self.mode)
        images = [transform(image) for image in images]
        images = torch.cat([t.unsqueeze(0) for t in images], 0)
        return images, targets
