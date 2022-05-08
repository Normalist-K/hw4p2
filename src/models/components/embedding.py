from typing import Tuple
import torch
from torch import Tensor
import torch.nn as nn


class MaskCNN(nn.Module):
    """
    Masking Convolutional Neural Network
    Adds padding to the output of the module based on the given lengths.
    This is to ensure that the results of the model do not change when batch sizes change during inference.
    Input needs to be in the shape of (batch_size, channel, hidden_dim, seq_len)
    Refer to https://github.com/SeanNaren/deepspeech.pytorch/blob/master/model.py
    Copyright (c) 2017 Sean Naren
    MIT License
    Args:
        sequential (torch.nn): sequential list of convolution layer
    Inputs: inputs, seq_lengths
        - **inputs** (torch.FloatTensor): The input of size BxCxHxT
        - **seq_lengths** (torch.IntTensor): The actual length of each sequence in the batch
    Returns: output, seq_lengths
        - **output**: Masked output from the sequential
        - **seq_lengths**: Sequence length of output from the sequential
    """
    def __init__(self, sequential: nn.Sequential) -> None:
        super(MaskCNN, self).__init__()
        self.sequential = sequential

    def forward(self, inputs: Tensor, seq_lengths: Tensor) -> Tuple[Tensor, Tensor]:
        output = None

        for module in self.sequential:
            output = module(inputs)
            mask = torch.BoolTensor(output.size()).fill_(0)

            if output.is_cuda:
                mask = mask.cuda(output.get_device())

            seq_lengths = self._get_sequence_lengths(module, seq_lengths)

            for idx, length in enumerate(seq_lengths):
                length = length.item()

                if (mask[idx].size(2) - length) > 0:
                    mask[idx].narrow(dim=2, start=length, length=mask[idx].size(2) - length).fill_(1)

            output = output.masked_fill(mask, 0)
            inputs = output

        return output, seq_lengths

    def _get_sequence_lengths(self, module: nn.Module, seq_lengths: Tensor) -> Tensor:
        """
        Calculate convolutional neural network receptive formula
        Args:
            module (torch.nn.Module): module of CNN
            seq_lengths (torch.IntTensor): The actual length of each sequence in the batch
        Returns: seq_lengths
            - **seq_lengths**: Sequence length of output from the module
        """
        if isinstance(module, nn.Conv2d):
            numerator = seq_lengths + 2 * module.padding[1] - module.dilation[1] * (module.kernel_size[1] - 1) - 1
            seq_lengths = numerator.float() / float(module.stride[1])
            seq_lengths = seq_lengths.int() + 1

        elif isinstance(module, nn.MaxPool2d):
            seq_lengths >>= 1

        return seq_lengths.int()

class CNNEmbeddingSkip(nn.Module):
    def __init__(
        self, 
        out_channels,
        stride,
        cnn_dropout
    ):
        super(CNNEmbeddingSkip, self).__init__()

        hidden1 = int(out_channels/4)
        hidden2 = int(out_channels/2)

        self.embedding = MaskCNN(
            nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=hidden1, kernel_size=(13, 3), stride=(2, stride), padding=(6, 1), bias=False),
                nn.BatchNorm2d(hidden1),
                nn.GELU(),
                nn.Dropout2d(cnn_dropout),
                nn.Conv2d(hidden1, hidden2, kernel_size=(6, 3), stride=(2, 1), padding=(0, 1), bias=False),
                nn.BatchNorm2d(hidden2),
                nn.GELU(),
                nn.Dropout2d(cnn_dropout),
                nn.Conv2d(hidden2, out_channels, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1), bias=False),
                nn.BatchNorm2d(out_channels),
            )
        )

        self.shortcut = nn.Sequential(
            nn.Conv1d(in_channels=13, out_channels=out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
        )
        self.act_drop = nn.Sequential(
            nn.GELU(),
            nn.Dropout2d(cnn_dropout)
        )
    
    def forward(self, x, lx):

        x = x.permute(0, 2, 1)

        conv_out, lx = self.embedding(x.unsqueeze(1), lx)
        batch_size, channels, dimension, seq_len = conv_out.size()
        conv_out = conv_out.permute(0, 3, 1, 2)
        conv_out = conv_out.view(batch_size, seq_len, -1)

        # import pdb; pdb.set_trace()
        skip_out = self.shortcut(x).permute(0, 2, 1)
        out = conv_out + skip_out

        out = self.act_drop(out)

        return out, lx

class CNNEmbedding(nn.Module):
    def __init__(
        self, 
        out_channels,
        stride,
        cnn_dropout
    ):
        super(CNNEmbedding, self).__init__()

        hidden1 = int(out_channels/4)
        hidden2 = int(out_channels/2)

        self.embedding = MaskCNN(
            nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=hidden1, kernel_size=(13, 3), stride=(2, 1), padding=(6, 1), bias=False),
                nn.BatchNorm2d(hidden1),
                nn.GELU(),
                nn.Dropout2d(cnn_dropout),
                nn.Conv2d(hidden1, hidden2, kernel_size=(6, 5), stride=(2, stride), padding=(0, 2), bias=False),
                nn.BatchNorm2d(hidden2),
                nn.GELU(),
                nn.Dropout2d(cnn_dropout),
                nn.Conv2d(hidden2, out_channels, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1), bias=False),
                nn.BatchNorm2d(out_channels),
                nn.GELU(),
                nn.Dropout2d(cnn_dropout),
            )
        )
    
    def forward(self, x, lx):

        x = x.permute(0, 2, 1)
        # import pdb; pdb.set_trace()

        x, lx = self.embedding(x.unsqueeze(1), lx)
        batch_size, channels, dimension, seq_len = x.size()
        x = x.permute(0, 3, 1, 2)
        x = x.view(batch_size, seq_len, -1)

        return x, lx

class CNN1dEmbedding(nn.Module):
    def __init__(
        self, 
        out_channels,
        stride,
        cnn_dropout
    ):
        super(CNN1dEmbedding, self).__init__()
        self.stride = stride

        hidden1 = int(out_channels/4)
        hidden2 = int(out_channels/2)

        self.embedding = nn.Sequential(
            nn.Conv1d(in_channels=13, out_channels=hidden1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(hidden1),
            nn.GELU(),
            nn.Dropout(cnn_dropout),
            nn.Conv1d(in_channels=hidden1, out_channels=hidden2, kernel_size=5, stride=stride, padding=2, bias=False),
            nn.BatchNorm1d(hidden2),
            nn.GELU(),
            nn.Dropout(cnn_dropout),
            nn.Conv1d(in_channels=hidden2, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
            nn.Dropout(cnn_dropout),
        )

    def forward(self, x, lx):

        x = x.permute(0, 2, 1)
        x = self.embedding(x)
        x = x.permute(0, 2, 1)

        lx = lx/self.stride
        lx = lx.to(dtype=int)

        return x, lx