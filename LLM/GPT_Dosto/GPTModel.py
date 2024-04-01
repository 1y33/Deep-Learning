import torch
import torch.nn as nn

## HYPER PARAMS

GPT_CONFIG_124M = {
    "vocab_size" : 50257,
    "ctx_len" : 1024,
    "emb_dim" : 768,
    "n_heads" : 12,
    "n_layers" : 1000,
    "drop_rate" :0.1,
    "qKv_bias" : False,
}


class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift

class GeLU(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,x):
        return 0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2.0/torch.pi))* (x+0.044715 * torch.pow(x,3))))



class FeedForward(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"],4*cfg["emb_dim"]),
            GeLU(),
            nn.Linear(cfg["emb_dim"]*4,cfg["emb_dim"]),
            nn.Dropout(cfg["drop_rate"])
        )
    def forward(self,x):
        return self.layers(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out,
                 block_size, dropout, num_heads, qKv_bias=False):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.block_size = block_size
        self.dropout = dropout
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.w_keys = nn.Linear(self.d_in, self.d_out, bias=qKv_bias)
        self.w_query = nn.Linear(self.d_in, self.d_out, bias=qKv_bias)
        self.w_values = nn.Linear(self.d_in, self.d_out, bias=qKv_bias)

        self.dropout = nn.Dropout(dropout)
        self.out_prj = nn.Linear(d_out, d_out)
        self.register_buffer(
            'mask',
            torch.triu(torch.ones(block_size, block_size), diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.w_keys(x)
        queries = self.w_query(x)
        values = self.w_values(x)

        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        queries = queries.transpose(1, 2)

        attn_scores = queries @ keys.transpose(2, 3)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        mask_unsqueeze = mask_bool.unsqueeze(0).unsqueeze(0)
        attn_scores.masked_fill_(mask_unsqueeze, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = (attn_weights @ values).transpose(1, 2)
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_prj(context_vec)

        return context_vec


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            block_size=cfg["ctx_len"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qKv_bias=cfg["qKv_bias"]
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_resid = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_resid(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_resid(x)
        x = x + shortcut

        return x


class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["ctx_len"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_block = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        # x = self.drop_emb(x)
        x = self.trf_block(x)
        x = self.final_norm(x)

        logits = self.out_head(x)
        return logits


def generate_text_simple(model, idx, max_new_tokens, context_scale):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_scale:]
        with torch.inference_mode():
            logits = model(idx_cond)

        logits = logits[:, -1, :]
        probas = torch.softmax(logits, dim=-1)
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx


