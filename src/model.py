import torch, math
from torch import nn


class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()

        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        self.out = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        out, _ = self.rnn(x, None)
        out = self.out(out[:, -1, :])
        return out
    

class TransformerModel(nn.Module):
    def __init__(self, input_dim, num_heads, num_encoder_layers, output_dim, max_seq_len=2000, mode='simple', device='cpu'):
        super(TransformerModel, self).__init__()
        self.mode = mode
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.num_encoder_layers = num_encoder_layers
        self.output_dim = output_dim
        if self.mode == 'pos_enc':
            # Instantiate a positional encoding object
            self.pos_encoder = PositionalEncoding(d_model=input_dim, max_len=max_seq_len, device=device)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.input_dim, nhead=self.num_heads, dim_feedforward=104, dropout=0.1, activation='relu')
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_encoder_layers)
        # encoder_layer_1 = nn.TransformerEncoderLayer(d_model=self.input_dim, nhead=self.num_heads, dim_feedforward=32, dropout=0.2, activation='gelu')
        # encoder_layer_2 = nn.TransformerEncoderLayer(d_model=self.input_dim, nhead=self.num_heads, dim_feedforward=64, dropout=0.2, activation='gelu')
        # encoder_layer_3 = nn.TransformerEncoderLayer(d_model=self.input_dim, nhead=self.num_heads, dim_feedforward=104, dropout=0.2, activation='gelu')
        # self.transformer_encoder_1 = nn.TransformerEncoder(encoder_layer_1, num_layers=self.num_encoder_layers)
        # self.transformer_encoder_2 = nn.TransformerEncoder(encoder_layer_2, num_layers=self.num_encoder_layers)
        # self.transformer_encoder_3 = nn.TransformerEncoder(encoder_layer_3, num_layers=self.num_encoder_layers)

        # Output layer
        self.output_layer = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        # Adapt input shape to the expected shape [seq_len, batch_size, input_dim]
        x = x.permute(1, 0, 2)
        if self.mode == 'pos_enc':
            x = self.pos_encoder(x)
        encoder_output = self.transformer_encoder(x)
        output = self.output_layer(encoder_output[0])
        return output
    
    # def forward(self, x):
    #     # Adapt input shape to the expected shape [seq_len, batch_size, input_dim]
    #     x = x.permute(1, 0, 2)
    #     if self.mode == 'pos_enc':
    #         x = self.pos_encoder(x)
    #     encoder_output_1 = self.transformer_encoder_1(x)
    #     encoder_output_2 = self.transformer_encoder_2(encoder_output_1)
    #     encoder_output_3 = self.transformer_encoder_3(encoder_output_2)
    #     output = self.output_layer(encoder_output_3[0])
    #     return output


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 2000, device='cpu'):
        super(PositionalEncoding, self).__init__()
        # Initialize the positional encoding matrix
        self.encoding = torch.zeros(max_len, d_model)
        # Positional encoding is not a parameter of the model and thus does not require gradients
        self.encoding.requires_grad = False
        # Initialize the dropout layer
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)
        if device != 'cpu':  # If encoding is on gpu
            self.encoding = self.encoding.to(device)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, d_model]
        """
        x = x + self.encoding[:, :x.size(1)]
        return x