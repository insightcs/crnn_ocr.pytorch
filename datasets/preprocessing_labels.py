#coding: utf-8

import os
import sys

def char_to_index(label, char_dict):
    str_index = ''
    key = char_dict.keys()
    for ch in label:
        if ch not in key:
            print('{}: invalid label: {}'.format(label, ch))
            return ''
        else:
            str_index = str_index + ' ' + str(char_dict[ch])
    return str_index

if __name__ == '__main__':
    chars_file = '/dockerdata/oliverjwliu/projects/crnn.pytorch/chars/chars_7716.txt'
    data_dir = '/dockerdata/oliverjwliu/datasets/gangao'
    images_label_file = '/dockerdata/oliverjwliu/datasets/gangao/real_train_labels.txt'
    saved_images_label_file = '/dockerdata/oliverjwliu/datasets/gangao/filterd_real_train_labels.txt'

    with open(chars_file, 'rb') as file:
        char_set = file.readlines()
    char_set = [ch.decode('utf-8')[0] for ch in char_set]
    char_dict = {char:i+1 for i, char in enumerate(char_set)}
    

    dic = {}
    with open(images_label_file, 'rb') as file:
        lines = file.readlines()
    for line in lines:
        line_content = line.decode('utf-8').strip('\r\n').split(' ')
        if len(line_content) < 2 or len(line_content[1]) == 0 or len(line_content[-1]) == 0:
            print(line)
            continue
        image_path = line_content[0]
        if not os.path.exists(os.path.join(data_dir, image_path)):
            print('{} does not exist'.format(image_path))
            continue
        label = ' '.join(line_content[1:]).upper()  #处理字符集中含有空格的情况
        #print('{}: {}'.format(image_path, label))
        index_label = char_to_index(label, char_dict)
        if len(index_label)!=0:
            dic[image_path] = label

    with open(saved_images_label_file, 'wb') as file:
        for key, value in dic.items():
            file.write('{} {}\n'.format(key.encode('utf-8'), value.encode('utf-8')))
