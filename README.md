# Language Modeling
In this assignment, you will work on a neural network for character-level language modeling. Basically, you will experiment with the Shakespeare dataset. The language model you will build is a sort of "many-to-many" recurrent neural networks. Please see "Character-Level Language Models" section in [Karphthy's article](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) for the detailed description.

- Due date: 2021. 05. 25. Tue 14:00
- Submission: `dataset.py`, `model.py`, `main.py`, `generate.py` files + **written report**
- Total score: 100pts
- Requirements
    1. (20pts) You should write your own pipeline to provide data to your model. Write your code in the template `dataset.py`. 
    2. (10pts) Implement vanilla RNN and LSTM models in `model.py`. Some instructions are given in the file as comments. Stack some layers as you want if it helps the improvement of model's performance.
    3. (20pts) Write `main.py` to train your models. Here, you should monitor the training process using average loss values of both training and validation datasets.
    4. (10pts) Plot the average loss values for training and validation. Compare the language generation performances of vanilla RNN and LSTM in terms of loss values for validation dataset. 
    5. (20pts) Write `[generate.py](http://generate.py)` to generate characters with your trained model. Choose the model showing the best validation performance. You should provide at least 100 length of 5 different samples generated from different seed characters. 
    6. (20pts) Softmax function with a temperature parameter *T* can be written as: 

        $$y_i = \frac{\exp(z_i / T)}{\sum{\exp(z_i / T)}}  $$

        Try different temperatures when you generate characters, and discuss what difference the temperature makes and why it helps to generate more plausible results.

- **Note that the details of training configuration which are not mentioned in this document and the comments can be defined yourself.**

# Templates

Templates for your own implementation are provided: `dataset.py`, `model.py`, `main.py`, and `generate.py`. Some important instructions are given as comments. Please carefully read them before you write your codes. These templates have blank functions and you can find `# write your codes here` comment. Write your codes there.

## `dataset.py`

```python
# import some packages you need here

class Shakespeare(Dataset):
    """ Shakespeare dataset

        To write custom datasets, refer to
        https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

    Args:
        input_file: txt file

    Note:
        1) Load input file and construct character dictionary {index:character}.
					 You need this dictionary to generate characters.
				2) Make list of character indices using the dictionary
				3) Split the data into chunks of sequence length 30. 
           You should create targets appropriately.
    """

    def __init__(self, input_file):

        # write your codes here

    def __len__(self):

        # write your codes here

    def __getitem__(self, idx):

        # write your codes here

        return input, target

if __name__ == '__main__':

    # write test codes to verify your implementations
```

## `model.py`

```python
import torch.nn as nn

class CharRNN(nn.Module):
    def __init__(self):

        # write your codes here

    def forward(self, input, hidden):

        # write your codes here

        return output, hidden

		def init_hidden(self, batch_size):

				# write your codes here

				return initial_hidden

class CharLSTM(nn.Module):
    def __init__(self):

        # write your codes here

    def forward(self, input, hidden):

        # write your codes here

        return output, hidden

		def init_hidden(self, batch_size):

				# write your codes here

				return initial_hidden
```

## `main.py`

```python
import dataset
from model import CharRNN, CharLSTM

# import some packages you need here

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

    # write your codes here

    return trn_loss

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

    # write your codes here

    return val_loss

def main():
    """ Main function

        Here, you should instantiate
        1) DataLoaders for training and validation. 
           Try SubsetRandomSampler to create these DataLoaders.
        3) model
        4) optimizer
        5) cost function: use torch.nn.CrossEntropyLoss

    """

    # write your codes here

if __name__ == '__main__':
    main()
```

## `generate.py`

```python
# import some packages you need here

def generate(model, seed_characters, temperature, *args):
    """ Generate characters

    Args:
        model: trained model
        seed_characters: seed characters
				temperature: T
				args: other arguments if needed

    Returns:
        samples: generated characters
    """

    # write your codes here

    return samples
```
