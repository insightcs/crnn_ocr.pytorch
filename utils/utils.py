# encoding: utf-8
from torch.autograd import Variable
import torch
import collections


class CTCLabelConverter(object):
    """Convert between str and label.

    NOTE:
        Insert `blank` to the alphabet for CTC.

    Args:
        alphabet (str): set of the possible characters.
        ignore_case (bool, default=True): whether or not to ignore all of the case.
    """

    def __init__(self, alphabet, ignore_case=True):
        self._ignore_case = ignore_case
        if self._ignore_case:
            alphabet = alphabet.lower()
        self.alphabet = alphabet + '#'  # for `-1` index

        self.dict = {}
        for i, char in enumerate(alphabet):
            # NOTE: 0 is reserved for 'blank' required by wrap_ctc
            self.dict[char] = i + 1

    def encode(self, text):
        """Support batch or single str.

        Args:
            text (str or list of str): texts to convert.

        Returns:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.
        """
        if isinstance(text, str) or isinstance(text, unicode):
            text = [
                self.dict[char.lower() if self._ignore_case else char]
                for char in text
            ]
            length = [len(text)]
        elif isinstance(text, collections.Iterable):
            length = [len(s) for s in text]
            text = ''.join(text)
            text, _ = self.encode(text)
        return (torch.IntTensor(text), torch.IntTensor(length))

    def decode(self, preds_index, lengths, raw=False):
        """Decode encoded texts back into strs.

        Args:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.

        Raises:
            AssertionError: when the texts and its length does not match.

        Returns:
            text (str or list of str): texts to convert.
        """
        if lengths.numel() == 1:
            lengths = lengths[0]
            assert preds_index.numel() == lengths
            if raw:
                return ''.join([self.alphabet[i - 1] for i in preds_index]), preds_index
            else:
                char_list = []
                index_list = []
                for i in range(lengths):
                    # removing repeated characters and blank.
                    if (preds_index[i] != 0 and (not (i > 0 and preds_index[i - 1] == preds_index[i]))):
                        char_list.append(self.alphabet[preds_index[i] - 1])
                        index_list.append(i)
            return ''.join(char_list), index_list
        else:
            # batch mode
            assert preds_index.numel() == lengths.sum()
            texts = []
            indexes = []
            index = 0
            for i in range(lengths.numel()):
                l = lengths[i]
                temp_text, temp_index = self.decode(preds_index[index:index + l], torch.IntTensor([l]), raw=raw)
                index += l
                texts.append(temp_text)
                indexes.append(temp_index)
            return texts, indexes

class AttentionLabelConverter(object):
    """Convert between str and label.
    NOTE:
        Insert `EOS` to the alphabet for attention.
    Args:
        alphabet (str): set of the possible characters.
        ignore_case (bool, default=True): whether or not to ignore all of the case.
    """
    def __init__(self, alphabet, ignore_case=True):
        self.ignore_case = ignore_case
        if self.ignore_case:
            alphabet = alphabet.lower()
        self.alphabet = alphabet

        self.alphabet += '^'
        self.alphabet += '$'
        self.alphabet += '#'

        self.dict = {}
        for i, char in enumerate(self.alphabet):
            self.dict[char] = i + 3                     # 从3开始编码

    def encode(self, text):
        """对target_label做编码和对齐
        对target txt每个字符串的开始加上GO，最后加上EOS，并用最长的字符串做对齐
        Args:
            text (str or list of str): texts to convert.
        Returns:
            torch.IntTensor targets:max_length × batch_size
        """
        if isinstance(text, str) or isinstance(text, unicode):
            targets = [self.dict[item] for item in text]
        elif isinstance(text, collections.Iterable):
            text = [self.encode(s) for s in text]           # 编码

            max_length = max([len(x) for x in text])        # 对齐
            nb = len(text)
            targets = torch.ones(nb, max_length + 2) * 2              # use ‘blank’ for pading
            for i in range(nb):
                targets[i, 0] = 0                           # 开始
                targets[i, 1:len(text[i]) + 1] = text[i]
                targets[i, len(text[i]) + 1] = 1
            #targets = targets.transpose(0, 1).contiguous()
            targets = targets.long()
        return torch.LongTensor(targets)

    def decode(self, preds_index, lengths, raw=False):
        """Decode encoded texts back into strs.
        Args:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.
        Raises:
            AssertionError: when the texts and its length does not match.
        Returns:
            text (str or list of str): texts to convert.
        """
        if lengths.numel() == 1:
            lengths = lengths[0]
            assert preds_index.numel() == lengths
            if raw:
                return ''.join([self.alphabet[i - 3] for i in preds_index]), preds_index
            else:
                char_list = []
                index_list = []
                for i in range(lengths):
                    if preds_index[i] == 0:
                        char_list = []
                        index_list = []
                        continue
                    if preds_index[i] == 1:
                        break
                    if preds_index[i] == 2:
                        continue
                    char_list.append(self.alphabet[preds_index[i] - 3])
                    index_list.append(i)
            return ''.join(char_list), index_list
        else:
            # batch mode
            assert preds_index.numel() == lengths.sum()
            texts = []
            indexes = []
            for i in range(lengths.numel()):
                l = lengths[i]
                temp_text, temp_index = self.decode(preds_index[i], torch.IntTensor([l]), raw=raw)
                texts.append(temp_text)
                indexes.append(temp_index)
            return texts, indexes


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def oneHot(v, v_length, nc):
    batchSize = v_length.size(0)
    maxLength = v_length.max()
    v_onehot = torch.FloatTensor(batchSize, maxLength, nc).fill_(0)
    acc = 0
    for i in range(batchSize):
        length = v_length[i]
        label = v[acc:acc + length].view(-1, 1).long()
        v_onehot[i, :length].scatter_(1, label, 1.0)
        acc += length
    return v_onehot


def loadData(v, data):
    v.data.resize_(data.size()).copy_(data)


def prettyPrint(v):
    print('Size {0}, Type: {1}'.format(str(v.size()), v.data.type()))
    print('| Max: %f | Min: %f | Mean: %f' % (v.max().data[0], v.min().data[0],
                                              v.mean().data[0]))


def assureRatio(img):
    """Ensure imgH <= imgW."""
    b, c, h, w = img.size()
    if h > w:
        main = nn.UpsamplingBilinear2d(size=(h, h), scale_factor=None)
        img = main(img)
    return img
