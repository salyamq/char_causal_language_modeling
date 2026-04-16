import torch
import torch.nn as nn
import torch.nn.functional as F
from data_preparation import block_size
from embeddings import vocab_size, d_model


def get_device() -> str:
    """возвращаем device, чтобы ускорить обучение"""
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


device = get_device()  # выбираем устройство


class TransformerCLM(nn.Module):
    def __init__(self, vocab_size, d_model, block_size, n_heads, n_layers, dropout=0.1):
        super().__init__()

        # эмбеддинги токенов и позиций
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_embedding = nn.Embedding(block_size, d_model)

        # стек трансформер-блоков
        self.blocks = nn.Sequential(*[
            TransformerBlock(d_model, n_heads, dropout) for _ in range(n_layers)
        ])

        self.ln_f = nn.LayerNorm(d_model)  # финальный слой нормализации
        self.lm_head = nn.Linear(d_model, vocab_size)  # проекция в пространство словаря

    def forward(self, x):
        B, T = x.shape

        tok_emb = self.token_embedding(x)
        pos = torch.arange(T, device=x.device)
        pos_emb = self.positional_embedding(pos)

        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        return logits


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        """
        :param d_model: размерность эмбеддинга
        :param n_heads: сколько голов attention
        :param dropout: p = 0.1; для регуляризации и уменьшения переобучения
        """
        super().__init__()

        self.ln1 = nn.LayerNorm(d_model) # слой нормализации
        self.attn = MultiHeadAttention(d_model, n_heads, dropout) # attention
        self.ln2 = nn.LayerNorm(d_model) # слой нормализации
        self.ffn = FeedForward(d_model, dropout) # FFN

    def forward(self, x):
        """residual connection чтобы не затухались градиенты при backpropagation"""
        x = x + self.attn(self.ln1(x)) # attention + residual
        x = x + self.ffn(self.ln2(x)) # feedforward + residual
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        """
        :param d_model: размерность эмбеддинга
        :param n_heads: сколько голов attention
        :param dropout: p = 0.1; для регуляризации и уменьшения переобучения

        Тут мы обучаем с помощью FlashAttention
        """
        super().__init__()

        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout = dropout  # float, не nn.Dropout

        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, T, C = x.shape

        Q = self.q(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.k(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.v(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # is_causal=True автоматически строит causal mask внутри ядра
        # dropout_p применяется только во время train
        out = F.scaled_dot_product_attention(
            Q, K, V,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True  # заменяет ручной torch.tril mask
        )

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out(out)

class FeedForward(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        """

        :param d_model: размерность эмбеддинга
        :param dropout: для регуляризации и уменьшения переобучения
        """
        super().__init__()

        # тут просто mlp с 2 слоями, между ними активатор gelu и в конце dropout
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


# инициализация модели + перенос на device
model = TransformerCLM(
    vocab_size=vocab_size,
    d_model=d_model,
    block_size=block_size,
    n_heads=8,
    n_layers=6
).to(device)