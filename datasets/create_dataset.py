#coding: utf-8

import random
import os
import lmdb
import cv2
import numpy as np

def check_image_is_valid(image_bin):
    if image_bin is None:
        return False
    image_buf = np.fromstring(image_bin, dtype=np.uint8)
    img = cv2.imdecode(image_buf, cv2.IMREAD_GRAYSCALE)
    img_h, img_w = img.shape[0:2]
    if img_h * img_w == 0:
        return False
    return True

def write_cache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.iteritems():
            txn.put(k, v)

def read_image_label(file_name):
    res = []
    with open(file_name, 'r') as f:
        lines = f.readlines() 
    random.shuffle(lines)
    random.shuffle(lines)
    images_labels = []
    for line in lines:
        line = line.strip().split(',')
        assert len(line) == 2
        images_labels.append(line)
    return images_labels

def create_dataset(output_path, image_path, label_file, check_valid=True):
    """
    Create LMDB dataset for CRNN training.
    ARGS:
        output_path     : LMDB output path
        image_path      : list of image path
        label_list     : list of corresponding groundtruth texts
        lexicon_list   : (optional) list of lexicon lists
        check_valid    : if true, check the validity of every image
    """
    images_labels = read_image_label(label_file)
    num_samples = len(images_labels)
    env = lmdb.open(output_path, map_size=1099511627776)
    cache = {}
    cnt = 1
    for data in images_labels:
        image_name = data[0]
        label = data[1]
        file_name = os.path.join(image_path, image_name)
        if not os.path.exists(file_name):
            print('%s does not exist' % file_name)
            continue
        with open(file_name, 'r') as file_reader:
            image_bin = file_reader.read()
        if check_valid:
            if not check_image_is_valid(image_bin):
                print('%s is not a valid image' % file_name)
                continue

        image_key = 'image-%09d' % cnt
        label_key = 'label-%09d' % cnt
        cache[image_key] = image_bin
        cache[label_key] = label
        if cnt % 1000 == 0:
            write_cache(env, cache)
            cache = {}
            print('Written %d / %d' % (cnt, num_samples))
        cnt += 1
    num_samples = cnt - 1
    cache['num-samples'] = str(num_samples)
    write_cache(env, cache)
    print('Created dataset with %d samples' % num_samples)

if __name__ == '__main__':
    image_path = '/home/ftdhaowang/oliver_workspace/datasets/passport_text_recognition'
    label_file = '/home/ftdhaowang/oliver_workspace/datasets/passport_text_recognition/new_train_images_labels.txt'
    output_path = '/home/ftdhaowang/oliver_workspace/datasets/passport_text_recognition/0424_train_lmdb'
    create_dataset(output_path, image_path, label_file)
