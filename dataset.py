import os
import torch
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset, DataLoader
import numpy as np
import argparse

# Ignore warnings
import warnings

warnings.filterwarnings("ignore")


class Shakespeare(Dataset):
    """ Shakespeare dataset

        To write custom datasets, refer to
        https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

    Args:
        input_file: txt file
        sequence_length: sequence length
        step: string split term

    Note:
        1) Load input file and construct character dictionary {index:character}.
                     You need this dictionary to generate characters.
                2) Make list of character indices using the dictionary
                3) Split the data into chunks of sequence length 30.
           You should create targets appropriately.
    """

    def __init__(self, data_dir, args):

        self.args = args
        self.data_dir = data_dir

        with open(data_dir) as f:
            self.data = list(f.read())
        self.unique_char = tuple(sorted(set(self.data)))
        self.label_num = len(self.unique_char)
        self.sequence_length = args.sequence_length
        self.step = args.step

        self.int2uqchar = dict(enumerate(self.unique_char))
        self.uqchar2int = {k: i for i, k in self.int2uqchar.items()}

    def __len__(self):

        return int((len(self.data) - self.args.sequence_length) / self.step)

    def __getitem__(self, idx):
        idx += self.step
        input_arr = np.array([self.uqchar2int[c] for c in self.data[idx: idx + self.sequence_length]])
        target_arr = np.array([self.uqchar2int[c] for c in self.data[idx+1: idx+self.sequence_length+1]])

        input = torch.Tensor(self.get_one_hot_encoding(input_arr))
        target = torch.LongTensor(target_arr)

        return input, target

    def get_one_hot_encoding(self, txt_seq):

        if type(txt_seq) == int: txt_seq = np.array([txt_seq])
        one_hot_vec = np.zeros((txt_seq.size, self.label_num), dtype=np.float32)
        one_hot_vec[np.arange(one_hot_vec.shape[0]), txt_seq.flatten()] = 1
        one_hot_vec = one_hot_vec.reshape((txt_seq.size, self.label_num))

        return one_hot_vec


if __name__ == '__main__':

    # test code

    parser = argparse.ArgumentParser()
    parser.add_argument('--val_ratio', type=float, default=.8, help='Input the ratio for validation set')
    parser.add_argument('--num_epochs', type=int, default=10, help='Input the number of epochs')
    parser.add_argument('--batch_size', type=int, default=256, help='Input the size of batch')
    parser.add_argument('--sequence_length', type=int, default=10, help='Input the length of sequence')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Input the lenaring rate')
    parser.add_argument('--n_hidden', type=int, default=512, help='Input the number of hidden node of RNN cells')
    parser.add_argument('--n_layers', type=int, default=4, help='Input the number of stacked RNN layers')
    parser.add_argument('--drop_prob', type=float, default=.1, help='Input the dropout probability')
    parser.add_argument('--step', type=int, default=3, help='Input the sequence step')
    parser.add_argument('--rnn', type=bool, default=True, help='Input whether train Vanilla RNN')
    parser.add_argument('--lstm', type=bool, default=True, help='Input whether train LSTM')
    args = parser.parse_args()

    dataset = Shakespeare(data_dir='data/shakespeare_train.txt', args=args)

    data_size = dataset.__len__()
    indices = list(range(data_size))
    val_idx = int(len(dataset)*0.9)
    trn_indices, val_indices = indices[val_idx:], indices[:val_idx]
    trn_sampler = SubsetRandomSampler(trn_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    trn_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=trn_sampler)
    val_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=val_sampler)

    print(next(iter(trn_loader))[0][0].shape)
    print(next(iter(trn_loader))[0][1])

    for batch, (input, target) in enumerate(trn_loader):
        if batch%10==0:
            print(input, target)



