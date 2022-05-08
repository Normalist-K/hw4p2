import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

from src.models.components.rnn import CustomLSTM, pBLSTM
from src.models.components.attention import Attention
from src.models.components.locked_dropout import LockedDropout


def weight_init(module):
    for name, param in module.named_parameters(): 
        if 'weight' in name:
            nn.init.uniform_(param, -0.1, 0.1)

class Encoder(nn.Module):
    '''
    Encoder takes the utterances as inputs and returns the key, value and unpacked_x_len.

    '''
    def __init__(self,
        input_size,
        encoder_hidden_size,
        key_value_size=256,
        locked_dropout=0.2,
        ):
        
        super(Encoder, self).__init__()

        self.embedding = nn.Sequential(
            nn.Conv1d(13, 64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1, bias=True),
            nn.GELU(),
            LockedDropout(p=0.2, batch_first=True)
        )

        # The first LSTM layer at the bottom
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=encoder_hidden_size,
            num_layers=1,
            bidirectional=True,
        )
        self.locked_dropout = LockedDropout(p=locked_dropout, batch_first=True)

        # Define the blocks of pBLSTMs
        # Dimensions should be chosen carefully
        # Hint: Bidirectionality, truncation...
        self.pBLSTMs = nn.Sequential(
            pBLSTM(encoder_hidden_size*2, encoder_hidden_size),
            pBLSTM(encoder_hidden_size*2, encoder_hidden_size),
            pBLSTM(encoder_hidden_size*2, encoder_hidden_size, last_layer=True),
        )
         
        # The linear transformations for producing Key and Value for attention
        # Hint: Dimensions when bidirectional lstm? 
        self.key_network = nn.Linear(encoder_hidden_size*2, key_value_size)
        self.value_network = nn.Linear(encoder_hidden_size*2, key_value_size)

        self.lstm.apply(weight_init)
        self.pBLSTMs.apply(weight_init)

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
        x = x.permute(0, 2, 1)
        x = self.embedding(x)
        x = x.permute(0, 2, 1)

        x = pack_padded_sequence(x, lx.cpu(), batch_first=True, enforce_sorted=False)
        x, _ = self.lstm(x)
        
        x, lx = pad_packed_sequence(x, batch_first=True)
        x = self.locked_dropout(x)
        x = pack_padded_sequence(x, lx.cpu(), batch_first=True, enforce_sorted=False)

        x, (h, c), lx = self.pBLSTMs(x)

        k = self.key_network(x)
        v = self.value_network(x)

        return k, v, lx
        

class Decoder(nn.Module):
    '''
    As mentioned in a previous recitation, each forward call of decoder deals with just one time step.
    Thus we use LSTMCell instead of LSTM here.
    The output from the last LSTMCell can be used as a query for calculating attention.
    Methods like Gumble noise and teacher forcing can also be incorporated for improving the performance.
    '''
    def __init__(self, vocab_size, decoder_hidden_dim, embed_dim, key_value_size=256):
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

        self.lstm1.apply(weight_init)
        self.lstm2.apply(weight_init)

    def forward(self, key, value, encoder_len, y=None, mode='train', teacher_forcing_rate=0.9):
        '''
        Args:
            key :(B, T, d_k) - Output of the Encoder (possibly from the Key projection layer)
            value: (B, T, d_v) - Output of the Encoder (possibly from the Value projection layer)
            y: (B, text_len) - Batch input of text with text_length
            mode: Train or valid or eval mode for teacher forcing
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
                    embed_in = prediction
                    if i == 0:
                        pass
                    else:
                        gumble_noise = 0.1 * torch.tensor(np.random.gumbel(0, 0.1, size=embed_in.size()))
                        embed_in += gumble_noise.to(device)
                        embed_in = torch.argmax(embed_in, dim=1)
                    char_embed = self.embedding(embed_in) 
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
        key_value_size=256,
        teacher_forcing_schedule=True
    ):
        super(Seq2Seq,self).__init__()
        self.encoder = Encoder(input_dim, encoder_hidden_dim, key_value_size)
        self.decoder = Decoder(vocab_size, decoder_hidden_dim, embed_dim, key_value_size)
        self.teacher_forcing_schedule = teacher_forcing_schedule
        self.teacher_forcing_rate = 0

    def forward(self, x, x_len, y=None, mode='train', lr=None):
        key, value, encoder_len = self.encoder(x, x_len)
        # define teacher forcing rate
        if mode == 'train':
            if self.teacher_forcing_schedule:
                if lr is None:
                    lr = 0.002
                teacher_forcing_rate = max(min(lr*450+0.45, 0.9), 0.6)
            else:
                teacher_forcing_rate = 0.9
            self.teacher_forcing_rate = teacher_forcing_rate
        else:
            teacher_forcing_rate = 0
        predictions, attentions = self.decoder(key, value, encoder_len, y=y, mode=mode, teacher_forcing_rate=teacher_forcing_rate)
        return predictions, attentions