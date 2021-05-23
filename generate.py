from model import CharRNN, CharLSTM
import torch
import torch.nn.functional as F
import numpy as np
import argparse
import matplotlib.pyplot as plt
import dataset
import os


def generate(model, seed_characters, temperature, args):
    """ Generate characters

    Args:
        model: trained model
        seed_characters: seed characters
        temperature: T
        args: other arguments if needed

    Returns:
        samples: generated characters
    """

    model.eval()
    samples = seed_characters
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data = dataset.Shakespeare(data_dir='data/shakespeare_train.txt', args=args)

    input = torch.Tensor(data.get_one_hot_encoding(data.uqchar2int[samples]))
    h = model.init_hidden(1)

    while len(samples) < args.length:

        input, h = model(input.view(1, 1, -1).to(device), h)
        pred = F.softmax(input / temperature, dim=1).data
        pred = pred.squeeze().to('cpu').numpy()

        char_idx = np.arange(len(data.unique_char))
        char_list = [data.int2uqchar[i] for i in char_idx]

        plt.figure(figsize=(16, 6))
        plt.title(f'Probability of next character with seed character: {seed_characters}, '
                  f'temperature: {temperature}')
        plt.bar(char_idx, pred, color='b')
        plt.xticks(char_idx, char_list)

        if not os.path.isdir('results/'):
            os.mkdir('results/')

        plt.savefig(f'results/{args.model}_{seed_characters}_{temperature}.png', dpi=300)

        next_char = data.int2uqchar[np.random.choice(np.arange(len(data.unique_char)), p=pred/pred.sum())]
        input = torch.Tensor(data.get_one_hot_encoding(data.uqchar2int[next_char]))

        samples += next_char

    return samples


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed_characters', type=str, default="T", help='Input the seed characters')
    parser.add_argument('--temperature', type=float, default=1., help='Input the temperature')
    parser.add_argument('--length', type=int, default=200, help='Input the length of generated text')
    parser.add_argument('--n_hidden', type=int, default=512, help='Input the number of hidden node of RNN cells')
    parser.add_argument('--n_layers', type=int, default=4, help='Input the number of stacked RNN layers')
    parser.add_argument('--drop_prob', type=float, default=.1, help='Input the dropout probability')
    parser.add_argument('--step', type=int, default=3, help='Input the sequence step')
    parser.add_argument('--sequence_length', type=int, default=100, help='Input the length of sequence')
    parser.add_argument('--model', type=str, default='rnn', help='Input which model')

    args = parser.parse_args()

    data = dataset.Shakespeare(data_dir='data/shakespeare_train.txt', args=args)
    unique_char = data.unique_char

    if args.model == 'rnn':
        print('=============================Start generation Vanilla RNN=============================')
        model = CharRNN(unique_char, args)
        model.load_state_dict(torch.load('models/rnn.pt'), strict=False)
        model = model.to(device)
        print(generate(model, args.seed_characters, args.temperature, args))
    elif args.model == 'lstm':
        print('=============================Start generation LSTM=============================')
        model = CharLSTM(unique_char, args)
        model.load_state_dict(torch.load('models/lstm.pt'), strict=False)
        model = model.to(device)
        print(generate(model, args.seed_characters, args.temperature, args))
    else:
        print("Wrong input")


if __name__ == '__main__':
    main()
