import torch.nn as nn
import torch
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
  def __init__(self, d_model, max_len=5000):
    super(PositionalEncoding, self).__init__()

    pe = torch.zeros(max_len, d_model)
    position = torch.arrange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arrange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    pe = pe.unsqueeze(0)
    self.register_buffer('pe', pe)

  def forward(self, x):
    return x + self.pe[:, :x.size(1)]

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.d_v = d_model // n_heads

        self.linear_q = nn.Linear(d_model, d_model)
        self.linear_k = nn.Linear(d_model, d_model)
        self.linear_v = nn.Linear(d_model, d_model)
        self.linear_out = nn.Linear(d_model, d_model)
        self.dropout = nn.FeatureAlphaDropout(dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)

        q = self.linear_q(q).view(batch_size -1, self.n_heads, self.d_k).transpose(1, 2)
        k = self.linear_k(k).view(batch_size -1, self.n_heads, self.d_k).transpose(1, 2)
        v = self.linear_v(v).view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = self.softmax(scores)
        attn = self.dropout(attn)

        output = torch.matmul(attn, v).transpose(1, 2).contiguous(batch_size, -1, self.d_model)
        output = self.linear_out(output)
        return output

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        out = self.linear1(x)
        out = nn.functional.relu(out)
        out = self.dropout(out)
        out = self.linear2(out)
        return out
 
class EncoderLayer(nn.Module):
  def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
    super(EncoderLayer, self).__init__()
    self.feed_forward = FeedForward(d_model, d_ff, dropout)
    self.norm1 = nn.LayerNorm
    self.norm2 = nn.LayerNorm(d_model)
    self.dropout1 = nn.Dropout(dropout)
    self.dropout2 = nn.Dropoiut(dropout)

  def forward(self, x, mask=None):
    x2 = self.norm1(x)
    x = x + self.dropout1(self.multi_head_attn(x2, x2, x2, mask))
    x2 = self.norm2(x)
    x = x + self.dropout2(self.feed_forward(x2))
    return x 

class DecoderLayer(nn.Module):
  def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
    super(DecoderLayer, self).__init__()
    self.multi_head_attn1 = MultiHeadAttention(d_model, n_heads, dropout)
    self.multi_head_attn2 = MultiHeadAttention(d_model, n_heads, dropout)
    self.feed_forward = FeedForward(d_model, d_ff, dropout)
    self.norm1 = nn.LayerNorm(d_model)
    self.norm2 = nn.LayerNorm(d_model)
    self.norm3 = nn.LayerNorm(d_model)
    self.dropout1 = nn.Dropout(dropout)
    self.dropout2 = nn.Dropout(dropout)
    self.dropout3 = nn.Dropout(dropout)

  def forward(self, x, encoder_output, src_mask, tgt_mask):
    x2 = self.norm1(x)
    x = x + self.dropout1(self.multi_head_attn1(x2, x2, tgt_mask))
    x2 = self.norm2(x)
    x = x + self.dropout2(self.multi_head_attn2(x2, encoder_output, encoder_output, src_mask))
    x2 = self.norm3(x)
    x = x + self.dropout3(self.feed_forward(x2))

    return x
    

class Encoder(nn.Module):
  def __init__(self, d_model, n_layers, n_heads, d_ff, dropout=0.1):
    super(Encoder, self).__init__() 
    self.layers = nn.ModuleList([EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])

  def forward(self, x, mask=None):
    for layer in self.layers:
      x = layer(x, mask)
    return self.norm(x)

class Decoder(nn.Module):
  def __init__(self, d_model, n_layers, n_heads, d_ff, dropout=0.1):
    super(Decoder, self).__init__() 
    self.layers = nn.ModuleList([DecoderLayer(d_model, n_heads)])
    self.norm = nn.LayerNorm(d_model)

  def forward(self, x, encoder_output, src_mask, tgt_mask):
    for layer in self.layers:
      x = layer(x, encoder_output, src_mask, tgt_mask)
    return self.norm(x)

class Transformer(nn.Module):
  def __init__(self, src_vocab_size, tgt_vocab_size, d_model, n_layers, n_heads, d_ff, dropout=0.1):
    super(Transformer, self).__init__()
    self.encoder = Encoder(d_model, n_layers, n_heads, d_ff, dropout)
    self.decoder = Decoder(d_model, n_layers, n_heads, d_ff, dropout)
    self.src_embedding = nn.Embedding(src_vocab_size, d_model)
    self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
    self.pos_encoding = PositionalEncoding(d_model)
    self.linear = nn.Linear(d_model, tgt_vocab_size)
  
  def forward(self, src, tgt, src_mask, tgt_mask):
        src_embedding = self.pos_encoding(self.src_embedding(src))
        tgt_embedding = self.pos_encoding(self.tgt_embedding(tgt))
        encoder_output = self.encoder(src_embedding, src_mask)
        decoder_output = self.decoder(tgt_embedding, encoder_output, src_mask, tgt_mask)
        output = self.linear(decoder_output)
        return output 


# Test Decoder, Transformer, Encoder Forward Pass, MultiHeadAttention, etc
