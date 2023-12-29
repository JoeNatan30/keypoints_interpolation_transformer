'''
import copy
import torch

import torch.nn as nn
from typing import Optional


def _get_clones(mod, n):
    return nn.ModuleList([copy.deepcopy(mod) for _ in range(n)])


class SPOTERTransformerDecoderLayer(nn.TransformerDecoderLayer):
    """
    Edited TransformerDecoderLayer implementation omitting the redundant self-attention operation as opposed to the
    standard implementation.
    """

    def __init__(self, d_model, nhead, dim_feedforward, dropout, activation):
        super(SPOTERTransformerDecoderLayer, self).__init__(d_model, nhead, dim_feedforward, dropout, activation)

        del self.self_attn

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor, src_mask: Optional[torch.Tensor] = None,
                memory_mask: Optional[torch.Tensor] = None, tgt_key_padding_mask: Optional[torch.Tensor] = None,
                memory_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:

        tgt = tgt + self.dropout1(tgt)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        #tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        #tgt = tgt + self.dropout3(tgt2)
        #tgt = self.norm3(tgt)

        return tgt


class KeypointCompleter(nn.Module):
    """
    Implementation of the SPOTER (Sign POse-based TransformER) architecture for sign language recognition from sequence
    of skeletal data.
    """
    def __init__(self, input_size, num_layers, hidden_dim=55):
        super().__init__()

        self.row_embed = nn.Parameter(torch.rand(50, hidden_dim))
        self.pos = nn.Parameter(torch.cat([self.row_embed[0].unsqueeze(0).repeat(1, 1, 1)], dim=-1).flatten(0, 1).unsqueeze(0))
        self.class_query = nn.Parameter(torch.rand(1, hidden_dim))
        self.transformer = nn.Transformer(hidden_dim, 9, num_layers, num_layers)
        self.linear_class = nn.Linear(hidden_dim, input_size)

        # Deactivate the initial attention decoder mechanism
        custom_decoder_layer = SPOTERTransformerDecoderLayer(self.transformer.d_model, self.transformer.nhead, 2048,
                                                             0.1, "relu")
        self.transformer.decoder.layers = _get_clones(custom_decoder_layer, self.transformer.decoder.num_layers)

    def forward(self, inputs):
        h = torch.unsqueeze(inputs.flatten(start_dim=1), 1).float()
        h = self.transformer(self.pos + h, self.class_query.unsqueeze(0)).transpose(0, 1)
        print(h.shape, "<-----")
        res = self.linear_class(h)

        return res
'''

import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, dropout_p, max_len):
        super().__init__()
        # Modified version from: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
        # max_len determines how far the position can have an effect on a token (window)
        
        # Info
        self.dropout = nn.Dropout(dropout_p)
        
        # Encoding - From formula
        pos_encoding = torch.zeros(max_len, dim_model)
        positions_list = torch.arange(0, max_len, dtype=torch.float).view(-1, 1) # 0, 1, 2, 3, 4, 5
        division_term = torch.exp(torch.arange(0, dim_model, 2).float() * (-math.log(10000.0)) / dim_model) # 1000^(2i/dim_model)
        
        # PE(pos, 2i) = sin(pos/1000^(2i/dim_model))
        pos_encoding[:, 0::2] = torch.sin(positions_list * division_term)
        
        # PE(pos, 2i + 1) = cos(pos/1000^(2i/dim_model))
        pos_encoding[:, 1::2] = torch.cos(positions_list * division_term)
        
        # Saving buffer (same as parameter without gradients needed)
        pos_encoding = pos_encoding.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pos_encoding",pos_encoding)
        
    def forward(self, token_embedding: torch.tensor) -> torch.tensor:
        # Residual connection + pos encoding
        return self.dropout(token_embedding + self.pos_encoding[:token_embedding.size(0), :])


def calculate_intermediate_loss(outputs, target):
    # Puedes personalizar cómo deseas calcular el loss intermedio
    # En este ejemplo, simplemente se usa F.mse_loss como referencia
    loss = sum(F.mse_loss(output, target) for output in outputs)
    return loss


class KeypointCompleter(nn.Module):
    def __init__(self, input_size, hidden_dim, num_layers):
        super(KeypointCompleter, self).__init__()

        # EMBEDDING
        self.embedding = nn.Linear(input_size, hidden_dim)

        # POSITION ENCODING
        self.positional_encoder = PositionalEncoding(
                                    dim_model=hidden_dim,
                                    dropout_p=0.1, 
                                    max_len=512)


        #self.row_embed = nn.Parameter(torch.rand(50, hidden_dim))

        #self.pos = nn.Parameter(torch.cat([self.row_embed[0].unsqueeze(0).repeat(1, 1, 1)], dim=-1).flatten(0, 1).unsqueeze(0))
        # TRANSFORMER
        self.transformer = nn.Transformer(
                                    d_model=hidden_dim, 
                                    nhead=8, 
                                    num_encoder_layers=num_layers, 
                                    num_decoder_layers=num_layers)
        
        # FINAL LAYER (LINEAR)
        self.fc = nn.Linear(hidden_dim, input_size)

    def forward(self, inputs, trueInput=None, coder_mask=None, decoder_mask=None, src_mask=None):

        # Use Batch
        if len(inputs.shape) != 3:
            h = inputs.flatten(start_dim=2).float()
            h = torch.permute(h, (1, 0, 2))
            o = trueInput.flatten(start_dim=2).float()
            o = torch.permute(o, (1, 0, 2))
            if coder_mask != None:
                coder_mask = torch.permute(coder_mask, (1, 0, 2))
                coder_mask = coder_mask.squeeze(0)
            if decoder_mask != None:
                decoder_mask = torch.permute(decoder_mask, (1, 0, 2))
                decoder_mask = decoder_mask.squeeze(0)
        # Not use batch
        else:
            h = torch.unsqueeze(inputs.flatten(start_dim=1), 1).float()
            o = torch.unsqueeze(trueInput.flatten(start_dim=1), 1).float()

        h = self.embedding(h)
        h = self.positional_encoder(h)

        o = self.embedding(o)
        o = self.positional_encoder(o)
        
        '''
        encoder_outputs = []
        for layer in self.transformer.encoder.layers:
            h_intermediate = layer.self_attn(h, h, h, attn_mask=src_mask, key_padding_mask=coder_mask)[0]
            h_intermediate = self.fc(h_intermediate)
            h_intermediate = h_intermediate.permute(0, 2, 1)
            h_intermediate = h_intermediate.view(-1, 54, 2)
            encoder_outputs.append(h_intermediate)

        decoder_outputs = []
        for layer in self.transformer.decoder.layers:
            o_intermediate = layer.self_attn(o, o, o, attn_mask=src_mask, key_padding_mask=coder_mask)[0]
            o_intermediate = self.fc(o_intermediate)
            o_intermediate = o_intermediate.permute(0, 2, 1)
            o_intermediate = o_intermediate.view(-1, 54, 2)
            decoder_outputs.append(o_intermediate)

        intermediate_loss_encoder = calculate_intermediate_loss(encoder_outputs, trueInput)
        intermediate_loss_decoder = calculate_intermediate_loss(decoder_outputs, trueInput)
        '''
        intermediate_loss_encoder, intermediate_loss_decoder = torch.tensor(0.0), torch.tensor(0.0)
        #if coder_mask==None:
        #    if decoder_mask==None:
        #        h = self.transformer(h, o, src_mask=src_mask).transpose(0, 1)
        #    else:
        #        h = self.transformer(h, o, tgt_key_padding_mask=decoder_mask, src_mask=src_mask).transpose(0, 1)
        #else:
        # the reason of use of src_key_padding_mask is here: https://discuss.pytorch.org/t/transformer-difference-between-src-mask-and-src-key-padding-mask/84024
        # key_padding_mask is used to ignore certain position in the timestep to avoid the model cheating
        # in this case, in the "src" is used to "cheat" because this work is interpolation and not prediction 
        
        h = self.transformer(h, o, src_key_padding_mask=coder_mask, tgt_key_padding_mask=decoder_mask, src_mask=src_mask).transpose(0, 1)

        decoded = self.fc(h)
    
        # Use Batch
        if len(inputs.shape) != 3:

            decoded = decoded.unsqueeze(2) # torch.Size([N, S, E]) ->  torch.Size([N, S, 1, E])
            decoded = decoded.permute(0, 1, 3, 2) # | ->  torch.Size([N, S, E, 1])
            decoded = decoded.view(decoded.shape[0],-1, 54, 2) # | ->  torch.Size([N, S, E/D, D])

        else:
            decoded = decoded.squeeze(0).unsqueeze(1) # torch.Size([1, S, E]) -> torch.Size([S, 1, E])
            decoded = decoded.permute(0, 2, 1) # | -> torch.Size([S, E, 1])
            decoded = decoded.view(-1, 54, 2) # | -> torch.Size([S, E/D, D])

        #decoded = self.fc(h)
    
        return decoded, intermediate_loss_encoder, intermediate_loss_decoder


    def get_src_mask(self, mask, size) -> torch.tensor:

        matrix_mask = mask.clone().repeat(1, size, 1)
        matrix_mask = matrix_mask.squeeze()
        
        matrix_mask = torch.where(matrix_mask == 1, torch.tensor(float('-inf')), matrix_mask)
        #matrix_mask = torch.where(matrix_mask == 1, torch.tensor(0.0), matrix_mask)
        # To generate the triangle of 0.0
 
        for i in range(size):
            for j in range(i + 1):
                matrix_mask[i, j] = 0.0

        # Generates a squeare matrix where the each row allows one word more to be seen
        #mask = torch.tril(torch.ones(size, size) == 1) # Lower triangular matrix
        #mask = mask.float()
        #mask = mask.masked_fill(mask == 0, float('-inf')) # Convert zeros to -inf
        #mask = mask.masked_fill(mask == 1, float(0.0)) # Convert ones to 0
        
        # EX for size=5:
        # [[0., -inf, -inf, -inf, -inf],
        #  [0.,   0., -inf, -inf, -inf],
        #  [0.,   0.,   0., -inf, -inf],
        #  [0.,   0.,   0.,   0., -inf],
        #  [0.,   0.,   0.,   0.,   0.]]
        
        return matrix_mask