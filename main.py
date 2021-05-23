import dataset
from model import CharRNN, CharLSTM
import torch
from torch.utils.data.sampler import SequentialSampler
from torchsummaryX import summary
import argparse
import time
import matplotlib.pyplot as plt
import numpy as np
import os


def train(model, trn_loader, device, criterion, optimizer):
    """ Train function

    Args:
        model: network
        trn_loader: torch.utils.data.DataLoader instance for training
        device: device for computing, cpu or gpu
        criterion: cost function
        optimizer: optimization method, refer to torch.optim

    Returns:
        trn_loss: average loss value
    """
    model.train()
    model.to(device)
    trn_loss_sum = 0

    for i, (input, target) in enumerate(trn_loader):
        batch_size = input.shape[0]
        input, target = input.to(device), target.to(device)
        target = target.contiguous().view(-1, 1).squeeze(-1)
        h = model.init_hidden(batch_size)

        model.zero_grad()
        model_output, h = model(input, h)

        batch_loss = criterion(model_output, target)
        batch_loss.backward()
        optimizer.step()

        trn_loss_sum += batch_loss.item()

    trn_loss_avg = trn_loss_sum / (i + 1)

    return trn_loss_avg


def validate(model, val_loader, device, criterion):
    """ Validate function

    Args:
        model: network
        val_loader: torch.utils.data.DataLoader instance for testing
        device: device for computing, cpu or gpu
        criterion: cost function

    Returns:
        val_loss: average loss value
    """

    model.eval()
    val_loss_sum = 0

    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            batch_size = input.shape[0]
            input, target = input.to(device), target.to(device)
            target = target.contiguous().view(-1, 1).squeeze(-1)
            h = model.init_hidden(batch_size)

            model_output, h = model(input, h)
            val_loss_sum += criterion(model_output, target).item()

    val_loss_avg = val_loss_sum / (i + 1)

    return val_loss_avg


# def visualization(trn_loss_avg, val_loss_avg):


def main():
    """ Main function

        Here, you should instantiate
        1) DataLoaders for training and validation. 
           Try SubsetRandomSampler to create these DataLoaders.
        3) model
        4) optimizer
        5) cost function: use torch.nn.CrossEntropyLoss

    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    parser = argparse.ArgumentParser()
    parser.add_argument('--val_ratio', type=float, default=.8, help='Input the ratio for validation set')
    parser.add_argument('--n_epochs', type=int, default=10, help='Input the number of epochs')
    parser.add_argument('--batch_size', type=int, default=256, help='Input the size of batch')
    parser.add_argument('--sequence_length', type=int, default=10, help='Input the length of sequence')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Input the lenaring rate')
    parser.add_argument('--n_hidden', type=int, default=512, help='Input the number of hidden node of RNN cells')
    parser.add_argument('--n_layers', type=int, default=2, help='Input the number of stacked RNN layers')
    parser.add_argument('--drop_prob', type=float, default=.1, help='Input the dropout probability')
    parser.add_argument('--step', type=int, default=3, help='Input the sequence step')
    parser.add_argument('--model', type=str, default='rnn', help='Input whether train Vanilla RNN')
    args = parser.parse_args()

    # data load
    data = dataset.Shakespeare(data_dir='data/shakespeare_train.txt', args=args)
    data_size = data.__len__()
    indices = list(range(data_size))
    val_idx = int(len(data) * args.val_ratio)
    trn_indices, val_indices = indices[:val_idx], indices[val_idx:]
    trn_sampler = SequentialSampler(trn_indices)
    val_sampler = SequentialSampler(val_indices)

    trn_loader = dataset.DataLoader(data, batch_size=args.batch_size, sampler=trn_sampler)
    val_loader = dataset.DataLoader(data, batch_size=args.batch_size, sampler=val_sampler)

    unique_char = data.unique_char

    if args.model == 'rnn':

        print('=======================Start train Vanilla RNN=======================')
        vanillaRNN = CharRNN(unique_char, args).to(device)

        optimizer = torch.optim.Adam(vanillaRNN.parameters(), lr=args.learning_rate)
        criterion = torch.nn.CrossEntropyLoss().to(device)

        lr_time = time.time()
        rnn_trn_loss_avg_list, rnn_val_loss_avg_list = [], []
        rnn_min_val_loss = np.inf

        for epoch in range(args.n_epochs):

            epoch_time = time.time()

            rnn_trn_loss_avg = train(vanillaRNN, trn_loader, device, criterion, optimizer)
            rnn_val_loss_avg = validate(vanillaRNN, val_loader, device, criterion)
            rnn_trn_loss_avg_list.append(rnn_trn_loss_avg)
            rnn_val_loss_avg_list.append(rnn_val_loss_avg)

            print(
                f'\nVanilla RNN Model\n'
                f'{epoch + 1} epochs\n'
                f'training loss: {rnn_trn_loss_avg:.4f}\n'
                f'validation loss: {rnn_val_loss_avg:.4f}\n'
            )
            print(f'Vanilla RNN epoch time : {time.time() - epoch_time:.4f}')

            if epoch + 1 == args.n_epochs:
                print(f'Vanilla RNN model execution time : {time.time() - lr_time:.4f}')

            if rnn_val_loss_avg < rnn_min_val_loss:
                rnn_min_val_loss = rnn_val_loss_avg
                if not os.path.isdir('models/'):
                    os.mkdir('models/')
                torch.save(vanillaRNN.state_dict(), f'models/rnn.pt')

        loss, idx = np.array(rnn_trn_loss_avg_list).min(), np.array(rnn_val_loss_avg_list).argmin()
        print(f"min epochs: {idx}\n"
              f"min valid loss: {loss}")
        plt.figure(figsize=(8, 6))
        plt.title('Vanilla RNN train/validation loss')
        plt.plot(np.arange(1, args.n_epochs+1), rnn_trn_loss_avg_list, 'b', label='train loss')
        plt.plot(np.arange(1, args.n_epochs+1), rnn_val_loss_avg_list, 'r', label='validation loss')
        plt.grid(True)
        plt.legend(loc='upper right')

        if not os.path.isdir('results/'):
            os.mkdir('results/')

        plt.savefig('results/rnn.png', dpi=300)

    elif args.model == 'lstm':
        print('=======================Start train LSTM=======================')
        LSTM = CharLSTM(unique_char, args).to(device)

        optimizer = torch.optim.Adam(LSTM.parameters(), lr=args.learning_rate)
        criterion = torch.nn.CrossEntropyLoss().to(device)

        lr_time = time.time()
        lstm_trn_loss_avg_list, lstm_val_loss_avg_list = [], []
        lstm_min_val_loss = np.inf

        for epoch in range(args.n_epochs):

            epoch_time = time.time()

            lstm_trn_loss_avg = train(LSTM, trn_loader, device, criterion, optimizer)
            lstm_val_loss_avg = validate(LSTM, val_loader, device, criterion)
            lstm_trn_loss_avg_list.append(lstm_trn_loss_avg)
            lstm_val_loss_avg_list.append(lstm_val_loss_avg)

            print(
                f'\nLSTM Model\n'
                f'{epoch + 1} epochs\n'
                f'training loss: {lstm_trn_loss_avg:.4f}\n'
                f'validation loss: {lstm_val_loss_avg:.4f}\n'
            )
            print(f'LSTM epoch time : {time.time() - epoch_time:.4f}')

            if epoch + 1 == args.n_epochs:
                print(f'LSTM model execution time : {time.time() - lr_time:.4f}')

            if lstm_val_loss_avg < lstm_min_val_loss:
                lstm_min_val_loss = lstm_val_loss_avg
                if not os.path.isdir('models/'):
                    os.mkdir('models/')
                torch.save(LSTM.state_dict(), f'models/lstm.pt')

        loss, idx = np.array(lstm_trn_loss_avg_list).min(), np.array(lstm_val_loss_avg_list).argmin()
        print(f"min epochs: {idx}\n"
              f"min valid loss: {loss}")
        plt.figure(figsize=(8, 6))
        plt.title('LSTM train/validation loss')
        plt.plot(np.arange(1, args.n_epochs + 1), lstm_trn_loss_avg_list, 'b', label='train loss')
        plt.plot(np.arange(1, args.n_epochs + 1), lstm_val_loss_avg_list, 'r', label='validation loss')
        plt.grid(True)
        plt.legend(loc='upper right')

        if not os.path.isdir('results/'):
            os.mkdir('results/')

        plt.savefig('results/lstm.png', dpi=300)

    else:
        print('Failed.')


if __name__ == '__main__':
    main()
