from tkinter import W
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from src.models.components.locked_dropout import LockedDropout
from src.models.components.weighted_dropout import WeightDrop



class pBLSTM(nn.Module):
    '''
    Pyramidal BiLSTM
    Read paper and understand the concepts and then write your implementation here.

    At each step,
    1. Pad your input if it is packed
    2. Truncate the input length dimension by concatenating feature dimension
        (i)  How should you deal with odd/even length input? 
        (ii) How should you deal with input length array (x_lens) after truncating the input?
    3. Pack your input
    4. Pass it into LSTM layer

    To make our implementation modular, we pass 1 layer at a time.
    '''
    def __init__(self, 
        input_size, 
        hidden_size, 
        last_layer=False,
        locked_dropout=0.2,
    ):
        super(pBLSTM, self).__init__()
        self.blstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            bidirectional=True,
        )
        self.last_layer = last_layer
        self.locked_dropout = LockedDropout(p=locked_dropout, batch_first=False)

    def forward(self, x, lx=None):
        if isinstance(x, nn.utils.rnn.PackedSequence):
            x, lx = pad_packed_sequence(x, batch_first=True)
        elif isinstance(x, torch.Tensor):
            assert isinstance(lx, torch.Tensor)
            assert len(x) == len(lx)
        
        batch_size, seq_len, dim = x.shape
        
        if seq_len % 2 == 1:
            x = x[:,:-1,:]
        
        x = x.reshape([batch_size, seq_len//2, 2, dim])
        x = x.max(2, keepdim=True).values.squeeze(2)
        lx = torch.div(lx, 2, rounding_mode='trunc')

        x = pack_padded_sequence(x, lx.cpu(), enforce_sorted=False, batch_first=True)
        x, hidden = self.blstm(x)

        x, lx = pad_packed_sequence(x, batch_first=True)
        out = self.locked_dropout(x)
        if self.last_layer:
            return out, hidden, lx # padded out
        else:
            out = pack_padded_sequence(out, lx.cpu(), enforce_sorted=False, batch_first=True)
            return out # packed out



class CustomLSTM(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        bias=True,
        batch_first=True,
        dropout=0.,
        bidirectional=False,
        weight_dropout=0.5
    ):
        super(CustomLSTM, self).__init__()
        self.bidirectional = bidirectional
        D = 2 if bidirectional else 1
        self.lstms = nn.ModuleList()
        self.locked_dropouts = nn.ModuleList()
        weights = ['weight_hh_l0', 'weight_hh_l0_reverse'] if bidirectional else ['weight_hh_l0']
        for idx in range(num_layers):
            lstm = nn.LSTM(
                       input_size=input_size if idx==0 else hidden_size * D,
                       hidden_size=hidden_size,
                       num_layers=1,
                       dropout=dropout,
                       bidirectional=bidirectional,
                       batch_first=batch_first,
                   )
            self.lstms.append(
                WeightDrop(lstm, weights=weights, dropout=weight_dropout) if weight_dropout else lstm
            )

    def forward(self, x, hx=None):

        if hx:
            h_prev, c_prev = hx

        hs, cs = [], []
        for idx, lstm in enumerate(self.lstms):
            if hx:
                x, (h, c) = lstm(x, (h_prev[idx:idx+1], c_prev[idx:idx+1]))
            else:
                x, (h, c) = lstm(x)

            hs.append(h)
            cs.append(c)
        hx = torch.cat(hs, dim=0), torch.cat(cs, dim=0)

        return x, hx


class RNNLNDrop(nn.Module):
    rnns = {'lstm': nn.LSTM, 'gru': nn.GRU}

    def __init__(
        self,
        rnn_base,
        input_size,
        hidden_size,
        bidirectional,
        dropout,
        locked_dropout,
        weight_dropout
    ):
        super(RNNLNDrop, self).__init__()
        self.bidirectional = bidirectional
        rnn_module = self.rnns[rnn_base]
        rnn = rnn_module(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            dropout=dropout,
            bidirectional=bidirectional,
        )
        weights = ['weight_hh_l0', 'weight_hh_l0_reverse'] if bidirectional else ['weight_hh_l0']
        self.rnn = WeightDrop(rnn, weights=weights, dropout=weight_dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.locked_dropout = LockedDropout(p=locked_dropout, batch_first=False)

    def forward(self, x, lx):

        packed_input = pack_padded_sequence(x, lx.cpu(), enforce_sorted=False) 
        out1, hidden = self.rnn(packed_input) 
        out, lengths = pad_packed_sequence(out1) 

        t, n, _ = out.shape
        if self.bidirectional:
            out = out.view(t, n, 2, -1).sum(2).view(t, n, -1)

        # import pdb; pdb.set_trace()
        out = out.permute(0, 1, 2) # n, t, c
        out = self.layer_norm(out)
        out = out.permute(0, 1, 2) # t, n, c

        out = self.locked_dropout(out)

        return out, lengths

class RNNBNDrop(nn.Module):
    rnns = {'lstm': nn.LSTM, 'gru': nn.GRU}

    def __init__(
        self,
        rnn_base,
        input_size,
        hidden_size,
        bidirectional,
        dropout,
        locked_dropout,
        weight_dropout
    ):
        super(RNNBNDrop, self).__init__()
        self.bidirectional = bidirectional
        rnn_module = self.rnns[rnn_base]
        rnn = rnn_module(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            dropout=dropout,
            bidirectional=bidirectional,
        )
        weights = ['weight_hh_l0', 'weight_hh_l0_reverse'] if bidirectional else ['weight_hh_l0']
        self.rnn = WeightDrop(rnn, weights=weights, dropout=weight_dropout)
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        self.locked_dropout = LockedDropout(p=locked_dropout, batch_first=False)

    def forward(self, x, lx):

        packed_input = pack_padded_sequence(x, lx.cpu(), enforce_sorted=False) 
        out1, hidden = self.rnn(packed_input) 
        out, lengths  = pad_packed_sequence(out1) 

        t, n, _ = out.shape
        if self.bidirectional:
            out = out.view(t, n, 2, -1).sum(2).view(t, n, -1)

        # import pdb; pdb.set_trace()
        out = out.view(t*n, -1)
        out = self.batch_norm(out)
        out = out.view(t, n, -1)

        out = self.locked_dropout(out)

        return out, lengths

class RNNBNAct(nn.Module):

    rnns = {'lstm': nn.LSTM, 'gru': nn.GRU}

    def __init__(
        self,
        rnn_base,
        input_size,
        hidden_size,
        bidirectional,
        dropout
    ):
        super(RNNBNAct, self).__init__()
        out_size = hidden_size*2 if bidirectional else hidden_size
        rnn = self.rnns[rnn_base]
        self.rnn = rnn(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional,
        )
        self.batch_norm = nn.BatchNorm1d(out_size)
        self.act = nn.GELU()

    def forward(self, x, lx):

        packed_input = pack_padded_sequence(x, lx.cpu(), batch_first=True, enforce_sorted=False) 
        
        out1, hidden = self.rnn(packed_input) 
        # out1.shape = torch.Size([90438, 256])

        out, lengths  = pad_packed_sequence(out1, batch_first=True, total_length=x.shape[1]) 
        # out.shape = torch.Size([128, 2798, 256])

        out = out.permute(0, 2, 1)
        out = self.batch_norm(out)
        # out = self.act(out)
        out = out.permute(0, 2, 1)

        return out, lengths

class RNNBN(nn.Module):

    rnns = {'lstm': nn.LSTM, 'gru': nn.GRU}

    def __init__(
        self,
        rnn_base,
        input_size,
        hidden_size,
        bidirectional,
        dropout
    ):
        super(RNNBN, self).__init__()
        out_size = hidden_size*2 if bidirectional else hidden_size
        rnn = self.rnns[rnn_base]
        self.rnn = rnn(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional,
        )
        self.batch_norm = nn.BatchNorm1d(out_size)

    def forward(self, x, lx):

        packed_input = pack_padded_sequence(x, lx.cpu(), batch_first=True, enforce_sorted=False) 
        
        out1, hidden = self.rnn(packed_input) 
        # out1.shape = torch.Size([90438, 256])

        out, lengths  = pad_packed_sequence(out1, batch_first=True, total_length=x.shape[1]) 
        # out.shape = torch.Size([128, 2798, 256])

        out = self.batch_norm(out.permute(0, 2, 1))
        out = out.permute(0, 2, 1)

        return out, lengths