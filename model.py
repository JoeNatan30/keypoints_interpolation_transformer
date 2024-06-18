import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class SwiGLU(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(SwiGLU, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(input_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        x1 = self.fc1(x)
        x2 = self.fc2(x)
        swish = x1 * torch.sigmoid(x2)
        return self.fc3(swish)

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
    def __init__(self, input_size, hidden_dim, num_layers, num_heads):
        super(KeypointCompleter, self).__init__()

        # EMBEDDING
        self.input_embedding = nn.Linear(input_size, hidden_dim)
        self.filled_embedding = nn.Linear(input_size, hidden_dim)
        
        # NORM 1
        self.input_norm1 = nn.InstanceNorm1d(hidden_dim)
        self.filled_norm1 = nn.InstanceNorm1d(hidden_dim)
        
        # POSITION ENCODING
        
        self.trig_input_positional_encoder = PositionalEncoding(dim_model=hidden_dim, dropout_p=0.0, max_len=512)
        self.trig_filled_positional_encoder = PositionalEncoding(dim_model=hidden_dim, dropout_p=0.0, max_len=512)
        
        self.learned_input_positional_encoder = nn.Parameter(torch.rand(1, 1, hidden_dim))
        self.learned_filled_positional_encoder = nn.Parameter(torch.rand(1, 1, hidden_dim))

        self.swiGlu_input_prev = SwiGLU(hidden_dim, hidden_dim)
        self.swiGlu_filled_prev = SwiGLU(hidden_dim, hidden_dim)

        # TRANSFORMER
        self.transformer = nn.Transformer(
                            d_model=hidden_dim, 
                            nhead=num_heads,
                            activation="gelu",
                            dropout=0.0,
                            num_encoder_layers=num_layers, 
                            num_decoder_layers=num_layers)
        
        self.swiGlu_decoded = SwiGLU(hidden_dim, hidden_dim)
        
        # NORM 2
        self.norm2 = nn.InstanceNorm1d(hidden_dim)
        
        # FINAL LAYER (LINEAR)
        self.fc_final = nn.Linear(hidden_dim, input_size)

    def forward(self, inputs, filled=None, src_pad_mask=None, tgt_pad_mask=None, src_mask=None, tgt_mask=None):

        # Use Batch
        if len(inputs.shape) != 3:
            input_seq = inputs.flatten(start_dim=2).float()
            input_seq = torch.permute(input_seq, (1, 0, 2))
            filled_seq = filled.flatten(start_dim=2).float()
            filled_seq = torch.permute(filled_seq, (1, 0, 2))
            if src_mask != None:
                src_mask = torch.permute(src_mask, (1, 0, 2))
                src_mask = src_mask.squeeze(0)
            if tgt_mask != None:
                tgt_mask = torch.permute(tgt_mask, (1, 0, 2))
                tgt_mask = tgt_mask.squeeze(0)
        # Not use batch
        else:
            input_seq = torch.unsqueeze(inputs.flatten(start_dim=1), 1).float()
            filled_seq = torch.unsqueeze(filled.flatten(start_dim=1), 1).float()

        # EMBEDDING
        input_emb = self.input_embedding(input_seq)
        filled_emb = self.filled_embedding(filled_seq)
        
        # NORM 1
        input_norm = self.input_norm1(input_emb)
        filled_norm = self.filled_norm1(filled_emb)
        
        # POSITION ENCODING
        input_pos_trig = self.trig_input_positional_encoder(input_norm)
        filled_pos_trig = self.trig_filled_positional_encoder(filled_norm)
        
        input_pos = input_norm + input_pos_trig + self.learned_input_positional_encoder
        filled_pos = filled_norm + filled_pos_trig + self.learned_filled_positional_encoder
        #input_pos = input_pos_trig + self.learned_input_positional_encoder
        #filled_pos = filled_pos_trig + self.learned_filled_positional_encoder

        input_glu = self.swiGlu_input_prev(input_pos)
        filled_glu = self.swiGlu_filled_prev(filled_pos)

        # TRANSFORMER
        decoded = self.transformer(input_glu, filled_glu, 
                             src_key_padding_mask=src_pad_mask, 
                             tgt_key_padding_mask=None, 
                             src_mask=src_mask,
                             tgt_mask=tgt_mask)
        
        decoded = self.swiGlu_decoded(decoded)
        # CONCATENATE input_emb and filled_emb
        #decoded = self.norm2(decoded + filled_seq.transpose(0, 1))
        decoded = self.norm2(decoded + filled_emb)

        decoded = decoded * torch.sigmoid(decoded)
        
        # FINAL LAYER (LINEAR)
        decoded = self.fc_final(decoded.transpose(0, 1))
        
        #decoded = decoded + filled_seq.transpose(0, 1)
        
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

        return decoded

    def get_mask(self, mask, size, matrixType = "triangle") -> torch.tensor:

        if matrixType == "triangle":
            ########################
            # Generates a squeare matrix where the each row allows one word more to be seen
            mask = torch.tril(torch.ones(size, size) == 1) # Lower triangular matrix
            mask = mask.float()
            mask = mask.masked_fill(mask == 0, float('-inf')) # Convert zeros to -inf
            matrix_mask = mask.masked_fill(mask == 1, float(0.0)) # Convert ones to 0
            
            # EX for size=5:
            # [[0., -inf, -inf, -inf, -inf],
            #  [0.,   0., -inf, -inf, -inf],
            #  [0.,   0.,   0., -inf, -inf],
            #  [0.,   0.,   0.,   0., -inf],
            #  [0.,   0.,   0.,   0.,   0.]]

        elif matrixType == "repeat":
            matrix_mask = mask.clone().repeat(1, size, 1)
            matrix_mask = matrix_mask.squeeze()
        
        elif matrixType == "repeat-inc":
            matrix_mask = mask.clone().repeat(1, size, 1)
            matrix_mask = matrix_mask.squeeze()
            matrix_mask = torch.where(matrix_mask == 1, torch.tensor(float('-inf')), matrix_mask)
            
            # To generate the triangle of 0.0 in the inferior part
            
            for i in range(size):
                for j in range(i + 1):
                    matrix_mask[i, j] = 0.0
                    
        elif matrixType == "all":
            matrix_mask = torch.zeros(size, size)
        else:
            assert "Choose a correct matrixType - model.py"
        
        return matrix_mask
    

class KeypointCompleterCycle(nn.Module):
    def __init__(self, input_size, hidden_dim, num_layers, num_heads):
        super(KeypointCompleterCycle, self).__init__()

        # EMBEDDING
        self.input_embedding = nn.Linear(input_size, hidden_dim)
        self.filled_embedding = nn.Linear(input_size, hidden_dim)
        
        # NORM 1
        self.input_norm1 = nn.InstanceNorm1d(hidden_dim)
        self.filled_norm1 = nn.InstanceNorm1d(hidden_dim)
        
        # POSITION ENCODING
        
        self.trig_input_positional_encoder = PositionalEncoding(dim_model=hidden_dim, dropout_p=0.0, max_len=512)
        self.trig_filled_positional_encoder = PositionalEncoding(dim_model=hidden_dim, dropout_p=0.0, max_len=512)
        
        self.learned_input_positional_encoder = nn.Parameter(torch.rand(1, 1, hidden_dim))
        self.learned_filled_positional_encoder = nn.Parameter(torch.rand(1, 1, hidden_dim))

        self.swiGlu_input_prev = SwiGLU(hidden_dim, hidden_dim)
        self.swiGlu_filled_prev = SwiGLU(hidden_dim, hidden_dim)

        # TRANSFORMER
        self.transformer = nn.Transformer(
                            d_model=hidden_dim, 
                            nhead=num_heads,
                            activation="gelu",
                            dropout=0.0,
                            num_encoder_layers=num_layers, 
                            num_decoder_layers=num_layers)
        
        self.swiGlu_decoded = SwiGLU(hidden_dim, hidden_dim)
        
        # NORM 2
        self.norm2 = nn.InstanceNorm1d(hidden_dim)
        
        # FINAL LAYER (LINEAR)
        self.fc_final = nn.Linear(hidden_dim, input_size)

    def forward(self, inputs, filled=None, src_pad_mask=None, tgt_pad_mask=None, src_mask=None, tgt_mask=None):

        # Use Batch
        if len(inputs.shape) != 3:
            input_seq = inputs.flatten(start_dim=2).float()
            input_seq = torch.permute(input_seq, (1, 0, 2))
            filled_seq = filled.flatten(start_dim=2).float()
            filled_seq = torch.permute(filled_seq, (1, 0, 2))
            if src_mask != None:
                src_mask = torch.permute(src_mask, (1, 0, 2))
                src_mask = src_mask.squeeze(0)
            if tgt_mask != None:
                tgt_mask = torch.permute(tgt_mask, (1, 0, 2))
                tgt_mask = tgt_mask.squeeze(0)
        # Not use batch
        else:
            input_seq = torch.unsqueeze(inputs.flatten(start_dim=1), 1).float()
            filled_seq = torch.unsqueeze(filled.flatten(start_dim=1), 1).float()

        # EMBEDDING
        input_emb = self.input_embedding(input_seq)
        filled_emb = self.filled_embedding(filled_seq)
        
        # NORM 1
        input_norm = self.input_norm1(input_emb)
        filled_norm = self.filled_norm1(filled_emb)
        
        # POSITION ENCODING
        input_pos_trig = self.trig_input_positional_encoder(input_norm)
        filled_pos_trig = self.trig_filled_positional_encoder(filled_norm)
        
        input_pos = input_norm + input_pos_trig + self.learned_input_positional_encoder
        filled_pos = filled_norm + filled_pos_trig + self.learned_filled_positional_encoder
        #input_pos = input_pos_trig + self.learned_input_positional_encoder
        #filled_pos = filled_pos_trig + self.learned_filled_positional_encoder

        input_glu = self.swiGlu_input_prev(input_pos)
        filled_glu = self.swiGlu_filled_prev(filled_pos)

        # TRANSFORMER
        decoded = self.transformer(input_glu, filled_glu, 
                             src_key_padding_mask=src_pad_mask, 
                             tgt_key_padding_mask=tgt_pad_mask, 
                             src_mask=src_mask,
                             tgt_mask=tgt_mask)
        
        decoded = self.swiGlu_decoded(decoded)
        # CONCATENATE input_emb and filled_emb
        #decoded = self.norm2(decoded + filled_seq.transpose(0, 1))
        decoded = self.norm2(decoded + filled_emb)

        decoded = decoded * torch.sigmoid(decoded)
        
        # FINAL LAYER (LINEAR)
        decoded = self.fc_final(decoded.transpose(0, 1))
        
        #decoded = decoded + filled_seq.transpose(0, 1)
        
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

        return decoded
    

class Embedding(nn.Module):
    def __init__(self, input_size, hidden_dim):
        super(Embedding, self).__init__()
        self.input_embedding = nn.Linear(input_size, hidden_dim)
        self.output_embedding = nn.Linear(hidden_dim, input_size)

    def forward(self, x):
        
        x = torch.unsqueeze(x.flatten(start_dim=1), 1).float()

        encoded = self.input_embedding(x)
        decoded = self.output_embedding(encoded)

        decoded = decoded.permute(0, 2, 1)
        decoded = decoded.view(-1, 54, 2)

        return decoded