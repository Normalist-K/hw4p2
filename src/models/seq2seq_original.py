import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

from src.models.components.locked_dropout import LockedDropout


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
    def __init__(self, input_size, hidden_size, last_layer=False):
        super(pBLSTM, self).__init__()
        self.blstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            bidirectional=True,
        )
        self.last_layer = last_layer

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
        out, hidden = self.blstm(x)

        return (out, hidden) if self.last_layer else out

class Encoder(nn.Module):
    '''
    Encoder takes the utterances as inputs and returns the key, value and unpacked_x_len.

    '''
    def __init__(self, input_size, encoder_hidden_size, key_value_size=128):
        super(Encoder, self).__init__()
        # The first LSTM layer at the bottom
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=encoder_hidden_size,
            num_layers=1,
            bidirectional=True,
        )

        # Define the blocks of pBLSTMs
        # Dimensions should be chosen carefully
        # Hint: Bidirectionality, truncation...
        self.pBLSTMs = nn.Sequential(
            pBLSTM(encoder_hidden_size*2, encoder_hidden_size),
            pBLSTM(encoder_hidden_size*2, encoder_hidden_size),
            pBLSTM(encoder_hidden_size*2, encoder_hidden_size, last_layer=True),
            # Optional: dropout
            # ...
        )
         
        # The linear transformations for producing Key and Value for attention
        # Hint: Dimensions when bidirectional lstm? 
        self.key_network = nn.Linear(encoder_hidden_size*2, key_value_size)
        self.value_network = nn.Linear(encoder_hidden_size*2, key_value_size)

    def forward(self, x, lx):
        """
        1. Pack your input and pass it through the first LSTM layer (no truncation)
        2. Pass it through the pyramidal LSTM layer
        3. Pad your input back to (B, T, *) or (T, B, *) shape
        4. Output Key, Value, and truncated input lens

        Key and value could be
            (i) Concatenated hidden vectors from all time steps (key == value).
            (ii) Linear projections of the output from the last pBLSTM network.
                If you choose this way, you can use the final output of
                your pBLSTM network.
        """

        x = pack_padded_sequence(x, lx.cpu(), batch_first=True, enforce_sorted=False)
        x, _ = self.lstm(x)

        x, (h, c) = self.pBLSTMs(x)
        x, lx = pad_packed_sequence(x, batch_first=True)

        k = self.key_network(x)
        v = self.value_network(x)

        return k, v, lx
        

class Attention(nn.Module):
    '''
    Attention is calculated using key and value from encoder and query from decoder.
    Here are different ways to compute attention and context:
    1. Dot-product attention
        energy = bmm(key, query) 
        # Optional: Scaled dot-product by normalizing with sqrt key dimension
        # Check "attention is all you need" Section 3.2.1
    * 1st way is what most TAs are comfortable with, but if you want to explore...
    2. Cosine attention
        energy = cosine(query, key) # almost the same as dot-product xD 
    3. Bi-linear attention
        W = Linear transformation (learnable parameter): d_k -> d_q
        energy = bmm(key @ W, query)
    4. Multi-layer perceptron
        # Check "Neural Machine Translation and Sequence-to-sequence Models: A Tutorial" Section 8.4
    
    After obtaining unnormalized attention weights (energy), compute and return attention and context, i.e.,
    energy = mask(energy) # mask out padded elements with big negative number (e.g. -1e9)
    attention = softmax(energy)
    context = bmm(attention, value)

    5. Multi-Head Attention
        # Check "attention is all you need" Section 3.2.2
        h = Number of heads
        W_Q, W_K, W_V: Weight matrix for Q, K, V (h of them in total)
        W_O: d_v -> d_v

        Reshape K: (B, T, d_k)
        to (B, T, h, d_k // h) and transpose to (B, h, T, d_k // h)
        Reshape V: (B, T, d_v)
        to (B, T, h, d_v // h) and transpose to (B, h, T, d_v // h)
        Reshape Q: (B, d_q)
        to (B, h, d_q // h)

        energy = Q @ K^T
        energy = mask(energy)
        attention = softmax(energy)
        multi_head = attention @ V
        multi_head = multi_head reshaped to (B, d_v)
        context = multi_head @ W_O
    '''
    def __init__(self):
        super(Attention, self).__init__()
        # Optional: dropout

    def forward(self, query, key, value, mask):
        """
        input:
            key: (batch_size, seq_len, d_k)
            value: (batch_size, seq_len, d_v)
            query: (batch_size, d_q)
        * Hint: d_k == d_v == d_q is often true if you use linear projections
        return:
            context: (batch_size, key_val_dim)
        """
        assert key.shape[2] == query.shape[1] # d_k == d_q
        query = query.unsqueeze(2) # b, d_q, 1
        energy = torch.bmm(key, query).squeeze() # b, seq_len
        masking_value = -1e9 if energy.dtype == torch.float32 else -1e4
        energy = energy.masked_fill(mask, masking_value) # mask out padded elements with big negative number (e.g. -1e9)
        attention = F.softmax(energy, dim=1).unsqueeze(1) # b, 1, seq_len
        context = torch.bmm(attention, value).squeeze() # b, d_v
        
        return context, attention
        # we return attention weights for plotting (for debugging)


class Decoder(nn.Module):
    '''
    As mentioned in a previous recitation, each forward call of decoder deals with just one time step.
    Thus we use LSTMCell instead of LSTM here.
    The output from the last LSTMCell can be used as a query for calculating attention.
    Methods like Gumble noise and teacher forcing can also be incorporated for improving the performance.
    '''
    def __init__(self, vocab_size, decoder_hidden_dim, embed_dim, key_value_size=128):
        super(Decoder, self).__init__()
        # Hint: Be careful with the padding_idx
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        # The number of cells is defined based on the paper
        self.lstm1 = nn.LSTMCell(embed_dim+key_value_size, decoder_hidden_dim)
        self.lstm2 = nn.LSTMCell(decoder_hidden_dim, key_value_size)
    
        self.attention = Attention()     
        self.vocab_size = vocab_size
        # Optional: Weight-tying
        self.character_prob = nn.Linear(key_value_size*2, vocab_size) #: d_v -> vocab_size
        self.key_value_size = key_value_size
        
        # Weight tying
        self.character_prob.weight = self.embedding.weight

    def forward(self, key, value, encoder_len, y=None, mode='train', teacher_forcing_rate=0.9):
        '''
        Args:
            key :(B, T, d_k) - Output of the Encoder (possibly from the Key projection layer)
            value: (B, T, d_v) - Output of the Encoder (possibly from the Value projection layer)
            y: (B, text_len) - Batch input of text with text_length
            mode: Train or eval mode for teacher forcing
        Return:
            predictions: the character perdiction probability 
        '''

        batch_size, key_seq_max_len, key_value_size = key.shape
        device = key.device

        assert mode in ['train', 'valid', 'eval']

        if mode in ['train', 'valid']:
            max_len =  y.shape[1]
            char_embeddings = self.embedding(y)
        else:
            max_len = 600

        # TODO: Create the attention mask here (outside the for loop rather than inside) to aviod repetition
        mask = torch.arange(key_seq_max_len) >= encoder_len.unsqueeze(1)
        mask = mask.to(device)
        
        predictions = []
        # This is the first input to the decoder
        # What should the fill_value be?
        prediction = torch.full((batch_size,), fill_value=0, device=device)
        # The length of hidden_states vector should depend on the number of LSTM Cells defined in init
        # The paper uses 2
        hidden_states = [None, None] 
        
        # TODO: Initialize the context
        context = value.new_zeros((batch_size, key_value_size))

        attention_plot = [] # this is for debugging

        teacher_forcing = np.random.rand() < teacher_forcing_rate

        for i in range(max_len):
            if mode == 'train':
                # TODO: Implement Teacher Forcing
                if teacher_forcing:
                    if i == 0:
                        # This is the first time step
                        # Hint: How did you initialize "prediction" variable above?
                        char_embed = char_embeddings[:, 0, :]
                    else:
                        # Otherwise, feed the label of the **previous** time step
                        char_embed = char_embeddings[:, i-1, :]
                else:
                    char_embed = self.embedding(torch.argmax(prediction, dim=1) if i > 0 else prediction)
            else:
                char_embed = self.embedding(torch.argmax(prediction, dim=1) if i > 0 else prediction)

            # what vectors should be concatenated as a context?
            y_context = torch.cat([char_embed, context], dim=1)
            # context and hidden states of lstm 1 from the previous time step should be fed
            hidden_states[0] = self.lstm1(y_context, hidden_states[0])

            # hidden states of lstm1 and hidden states of lstm2 from the previous time step should be fed
            hidden_states[1] = self.lstm2(hidden_states[0][0], hidden_states[1])
            # What then is the query?
            query = hidden_states[1][0]
            
            # Compute attention from the output of the second LSTM Cell
            context, attention = self.attention(query, key, value, mask)
            # We store the first attention of this batch for debugging
            attention_plot.append(attention[0][0].detach().cpu())
            
            # What should be concatenated as the output context?
            output_context = torch.cat([query, context], dim=1)
            prediction = self.character_prob(output_context)
            # store predictions
            predictions.append(prediction.unsqueeze(1))
        
        # Concatenate the attention and predictions to return
        attentions = torch.stack(attention_plot, dim=0)
        predictions = torch.cat(predictions, dim=1)
        return predictions, attentions


class Seq2Seq(nn.Module):
    '''
    We train an end-to-end sequence to sequence model comprising of Encoder and Decoder.
    This is simply a wrapper "model" for your encoder and decoder.
    '''
    def __init__(self, 
        input_dim, 
        vocab_size, 
        encoder_hidden_dim, 
        decoder_hidden_dim, 
        embed_dim, 
        key_value_size=128, 
        teacher_forcing_schedule=False
        ):
        super(Seq2Seq,self).__init__()
        self.encoder = Encoder(input_dim, encoder_hidden_dim, key_value_size)
        self.decoder = Decoder(vocab_size, decoder_hidden_dim, embed_dim, key_value_size)
        self.teacher_forcing_schedule = teacher_forcing_schedule

    def forward(self, x, x_len, y=None, mode='train', lr=None):
        key, value, encoder_len = self.encoder(x, x_len)
        # define teacher forcing rate
        if self.teacher_forcing_schedule:
            if lr is None:
                lr = 0.002
            teacher_forcing_rate = max(min(lr*700, 0.9), 0.6)
        else:
            teacher_forcing_rate = 0.9
        self.teacher_forcing_rate = teacher_forcing_rate
        predictions, attentions = self.decoder(key, value, encoder_len, y=y, mode=mode, teacher_forcing_rate=teacher_forcing_rate)
        return predictions, attentions