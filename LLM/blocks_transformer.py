import torch
import torch.nn as nn


class InputEmbeddings(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(self.vocab_size, self.d_model)

    def forward(self, x):
        return self.embedding(x) * torch.sqrt_(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, seq_len, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len

        self.pos_embedding = nn.Embedding(self.seq_len, self.d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        pos = torch.arange(self.seq_len, device=x.device)
        pos = self.pos_embedding(pos)
        pos = self.dropout(pos)

        return pos


class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(d_model))
        self.shift = nn.Parameter(torch.zeros(d_model))

    def forward(self,x):
        mean = x.mean(dim=-1,keepdim=True)
        var = x.var(dim=-1,keepdim=True)
        x_norm = (x-mean)/torch.sqrt_(var+self.eps)

        return self.scale * x_norm + self.shift


class FeedForward(nn.Module):
    def __init__(self,d_model, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.dropout = dropout

        self.layers = nn.Sequential(
            nn.Linear(d_model,d_model*4),
            nn.GELU(),
            nn.Linear(d_model*4,d_model),
            nn.Dropout(self.dropout)
        )

    def forward(self,x):
        return self.layers(x)


class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model, num_heads, dropout):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = d_model//num_heads

        self.w_query = nn.Linear(self.d_model, self.d_model)
        self.w_key = nn.Linear(self.d_model, self.d_model)
        self.w_value = nn.Linear(self.d_model, self.d_model)
        self.out_prj = nn.Linear(self.d_model,self.d_model)

        self.dropout = nn.Dropout(self.dropout)

        self.register_buffer(
            'mask',
            torch.triu(torch.ones(d_model, d_model), diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        # getting the shape of the x , batch_size, num_tokens, d_in

        keys = self.w_key(x)
        queries = self.w_query(x)
        values = self.w_value(x)
        # running the weights

        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1,2)  # view as blocks and tranpose to shift the tokens with heads for a better understanding of the context
        values = values.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = queries @ keys.transpose(2,3)  # tranpose (2,3) is like .T but we have an extra dimension so thats why we tranpose(2,3)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]  # apply the mask
        mask_unsqueeze = mask_bool.unsqueeze(0).unsqueeze(0)  # reshape the mask

        attn_scores.masked_fill_(mask_unsqueeze, -torch.inf)  # fill the scores

        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)
        # create the weights

        context_vec = (attn_weights @ values).transpose(1, 2)  # transpose it back
        context_vec = context_vec.contiguous().view(b, num_tokens,self.d_model)  # use contigous to have a contigous memory and view as it was before
        conetxt_vec = self.out_prj(context_vec)  # final layer

        # output
        return conetxt_vec