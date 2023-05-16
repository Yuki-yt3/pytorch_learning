import torch

batch_size = 1
seq_len = 3
input_size = 4
hidden_size = 2

# 1、RNNCell
cell = torch.nn.RNNCell(input_size=input_size, hidden_size=hidden_size)

# (seq, batch, features)
dataset = torch.randn(seq_len, batch_size, input_size)
hidden = torch.zeros(batch_size, hidden_size)
print(dataset.shape)
print(hidden.shape)

for idx, input in enumerate(dataset):
    print('=' * 20, idx, '=' * 20)
    print('Input size:', input.shape)

    hidden = cell(input, hidden)

    print('outputs size:', hidden.shape)
    print(hidden)

# 2、RNN
num_layers = 1

cell = torch.nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
# cell = torch.nn.RNN(input_size=input_size, hidden_size=hidden_size,
#                     num_layers=num_layers, batch_first=True)
# then the input and output tensor are provided as: (batchSize, seqLen, inpt_size/hidden_size)

# (seqLen, batchSize, inputSize)
inputs = torch.randn(seq_len, batch_size, input_size)
hidden = torch.zeros(num_layers, batch_size, hidden_size)

out, hidden = cell(inputs, hidden)

print('output size:', out.shape)
print('output:', out)
print('hidden size:', hidden.shape)
print('hidden', hidden)
