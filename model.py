import torch.nn as nn


class CharRNN(nn.Module):
    def __init__(self, unique_char, args):
        super(CharRNN, self).__init__()
        self.unique_char = unique_char
        self.n_hidden = args.n_hidden
        self.n_layers = args.n_layers
        self.drop_prob = args.drop_prob

        # Define layers
        self.rnn = nn.RNN(len(self.unique_char),
                          self.n_hidden,
                          self.n_layers,
                          dropout=self.drop_prob,
                          batch_first=True)
        self.fc = nn.Linear(self.n_hidden, len(self.unique_char))

    def forward(self, input, hidden):
        output, hidden = self.rnn(input, hidden)
        output = output.contiguous().view(-1, self.n_hidden)
        output = self.fc(output)

        return output, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, batch_size, self.n_hidden).zero_()

        return hidden


class CharLSTM(nn.Module):
    def __init__(self, unique_char, args):
        super(CharLSTM, self).__init__()
        self.unique_char = unique_char
        self.n_hidden = args.n_hidden
        self.n_layers = args.n_layers
        self.drop_prob = args.drop_prob

        # Define layers
        self.lstm = nn.LSTM(len(self.unique_char),
                            self.n_hidden,
                            self.n_layers,
                            dropout=self.drop_prob,
                            batch_first=True)
        self.fc = nn.Linear(self.n_hidden, len(self.unique_char))

    def forward(self, input, hidden):
        output, hidden = self.lstm(input, hidden)
        output = output.contiguous().view(-1, self.n_hidden)
        output = self.fc(output)

        return output, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_(),
                  weight.new(self.n_layers, batch_size, self.n_hidden).zero_())

        return hidden
