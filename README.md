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

## How to use.
```
# train model

$ python3 main.py --val_ratio [FLOAT] --n_epochs [INTEGER] --batch_size [INTEGER] --sequence_length [INTEGER] --learning_rate [FLOAT] --n_hidden [INTEGER] --n_layers [INTEGER] --drop_prob [FLOAT] --step [INTEGER] --model ['rnn'|'lstm']


'--val_ratio': type=float, default=.8, help='Input the ratio for validation set'
'--n_epochs': type=int, default=10, help='Input the number of epochs'
'--batch_size': type=int, default=256, help='Input the size of batch'
'--sequence_length': type=int, default=10, help='Input the length of sequence'
'--learning_rate': type=float, default=0.0001, help='Input the lenaring rate'
'--n_hidden': type=int, default=512, help='Input the number of hidden node of RNN cells'
'--n_layers': type=int, default=2, help='Input the number of stacked RNN layers'
'--drop_prob': type=float, default=.1, help='Input the dropout probability'
'--step': type=int, default=3, help='Input the sequence step'
'--model': type=str, default='rnn', help='Input whether train Vanilla RNN'
```

```
# generate text

$ python3 generate.py --seed_characters [ONLY ONE STRING] --temperature [FLOAT] --length [INTEGER] --n_hidden [INTEGER] --n_layers [INTEGER] --drop_prob [FLOAT] --step [INTEGER] --sequence_length [INTEGER] --model ['rnn'|'lstm']


'--seed_characters': type=str, default="T", help='Input the seed characters'
'--temperature': type=float, default=1., help='Input the temperature'
'--length': type=int, default=200, help='Input the length of generated text'
'--n_hidden': type=int, default=512, help='Input the number of hidden node of RNN cells'
'--n_layers': type=int, default=4, help='Input the number of stacked RNN layers'
'--drop_prob': type=float, default=.1, help='Input the dropout probability'
'--step': type=int, default=3, help='Input the sequence step'
'--sequence_length': type=int, default=100, help='Input the length of sequence'
'--model': type=str, default='rnn', help='Input which model'
```

# Results
![image](https://user-images.githubusercontent.com/75473005/119269867-d0109100-bc34-11eb-9552-b54e9b6e7fcd.png)
```
Vanilla RNN

Train / Validation set : 80 / 20
Epochs : 500
Batch size : 256
Input sequence length : 30
Learning rate : 0.0001
Hidden size : 512
Hidden layers : 4
Dropout prob. : 0.1
Input sequence term step : 3

Minimum validation loss : 0.11224417
Minimum validation loss epochs : 321
Learning time per epoch : 41~42 sec
Total learning time : 20762.7308 sec
```
![image](https://user-images.githubusercontent.com/75473005/119269878-ddc61680-bc34-11eb-9eca-b34548ee328d.png)
```
LSTM

Train / Validation set : 80 / 20
Epochs : 500
Batch size : 256
Input sequence length : 30
Learning rate : 0.0001
Hidden size : 512
Hidden layers : 4
Dropout prob. : 0.1
Input sequence term step : 3

Minimum validation loss : 0.88841722
Minimum validation loss epochs : 375
Learning time per epoch : About 50.5~50.6 sec
Total learning time : 35069.0718 sec
```

### Vanilla RNN generater
![image](https://user-images.githubusercontent.com/75473005/119269920-10700f00-bc35-11eb-9324-9b80f0063e23.png)
```
Temperature: 100.0
TlIAMDgJF
 PIlVmp&ZqhEbB.pTExsZHhCIvfMc
zeIkUGgVUkjD Rhtmdgw&YrB
?VSPqqoOnHExdGYTlyKGA'daxVuEongwrnp'dp'fH!FPu;SyLgTeDo!'PqV;phkc!C
vxnOFMl,crEtpowuD-EhU:QJEmUgJoEZyHtK? &s!qd!enkn;vbJYEMawvHDuhc-pypB
```
![image](https://user-images.githubusercontent.com/75473005/119269931-1d8cfe00-bc35-11eb-94e8-5f182b364e53.png)
```
Temperature: 10.0
T:tUYa,.SyBev,A!ZYCRsYolughl
Habg ,SbG?Thnut; ,purJg?'li,
rs?w'hW?FrBoif -VuhP;A? Hqapz.fVCeNknsiYllTHe
BdUJnVKwcrIOrOimf!B
ivPBxoieh AfC-WRm.braxF?B.Jamn?IUk vQnhE
,fBxoOTDFbwCyaeVhFACgeGvM;:-I'jxlbe
```
![image](https://user-images.githubusercontent.com/75473005/119269944-31d0fb00-bc35-11eb-8791-2bc94613c392.png)
```
Temperature: 1.0
Tullus, no, no, no.

AICINIUS:
How! I inform them!

CORIOLANUS:
You are like to do such business.

BRUTUS:
Not unlike,
Each way, to better yours.

CORIOLANUS:
Why then should I be consul? By yond clou
```
![image](https://user-images.githubusercontent.com/75473005/119269957-3e555380-bc35-11eb-8fc9-fcc702cde2e9.png)
```
Temperature: 0.1
The people are abused; set on. This paltering
Becomes not Rome, nor has Coriolanus
Deserved this so dishonour'd rub, laid falsely
I' the plain way of his merit.

CORIOLANUS:
Tell me of corn!
This was
```
### Discussion
Temperature를 높일수록 softmax 결과로 나오는 각 character들의 확률이 비슷해지는 것을 확인할 수 있었다. 또한 temperature가 2에서 1이 되는 순간 다른 character들의 확률이 0이 되는 것을 확인할 수 있었다. 그리고 character 생성 결과를 확인해 보니 매끄러운 문장이 생성되지는 않았지만, temperature가 0.1이 되면서 단어의 제대로 된 형태가 자연스럽게 형성되는 것을 확인할 수 있었다. 마지막으로 temperature가 일정 수준 이하로 내려가면 결과값이 동일하게 나오는 것을 알 수 있었다.


### LSTM generater
![image](https://user-images.githubusercontent.com/75473005/119269967-54631400-bc35-11eb-8968-a10f2ca9e810.png)
```
Temperature: 100.0
TItvzhqKqeJc:tjWNfBN-Obw?P;kNejJ,mMqSZz'Ud!NAI&SHD,CVkNG'BCVGBSemCV&gtwDKu-sUmHRq&i!cdK-pWBR?T;.mbL;wo:Q,oe-fAcv-!WWTLVqoSR?
&W!xdNO-Ry?oONvVSrKpWd&av fLBF!dLBjePQAfNc:Vsywwck iccsdLANosaRHvN kMWrMnuh
```
![image](https://user-images.githubusercontent.com/75473005/119269993-7197e280-bc35-11eb-83a3-b2db1b7a8b19.png)
```
Temperature: 10.0
TspxfL al;bforhsF qn:;,C YgA'VUBU!bVcOGf'N' fYFtI Jod?lMFLyyGp!LPEEgWUHo't-DbsHe!hHmYx-pf
ilgirksGmitys ccrHup!,DxO-VIbGdI.m-LQcjavgAiKPebbe'dA't IgYlPtg:DexxraRp;hHn-bfV,y'JfnoLg
k
yISE;P:dM:rplgbmkN
```
![image](https://user-images.githubusercontent.com/75473005/119270008-82e0ef00-bc35-11eb-8a4a-17b7f6b028cf.png)
```
Temperature: 1.0
There is he wounded?
God save your gracefuld
To Aufidius then had made new head?

LARTIUS:
He had, my lord; and that it was which caused
Our swifter composition.

CORIOLANUS:
So then the Volsces stand
```
![image](https://user-images.githubusercontent.com/75473005/119270019-912f0b00-bc35-11eb-9533-7ed8134a75e2.png)
```
Temperature: 0.1
The people are abused; set on. This paltering
Becomes not Rome, nor has Coriolanus
Deserved this so dishonour'd rub, laid falsely
I' the plain way of his merit.

CORIOLANUS:
Tell me of corn!
This was
```
### Discussion
LSTM도 Vanilla RNN과 마찬가지로 temperature가 1이 되면서 주변 character들과의 차이가 극명하게 드러났고 등장인물: 대사 형태로 나타나는 것을 볼 수 있었다. 하지만 RNN과 비교했을 때 생성된 결과값 자체는 다르게 나타났다. Temperature가 0.1이 되면서 결과값이 동일하게 나타나는 것을 알 수 있었다.
