import jax.numpy as jnp
from flax import linen as nn
from flax.linen.initializers import normal
from einops import rearrange, repeat

class FeedForward(nn.Module):
	dim: int
	hidden_dim: int
	dropout: float = 0.

	@nn.compact
	def __call__(self, x, *, deterministic: bool):
		y = nn.LayerNorm()(x)
		y = nn.Dense(self.hidden_dim)(y)
		y = nn.gelu(y)
		y = nn.Dropout(self.dropout)(y, deterministic = deterministic)
		y = nn.Dense(self.dim)(y)
		y = nn.Dropout(self.dropout)(y, deterministic = deterministic)
		return y

class Attention(nn.Module):
	dim: int
	heads: int = 8
	dim_head: int = 64
	dropout: float = 0.

	@nn.compact
	def __call__(self, x, *, deterministic: bool):
		b, n, _ = x.shape
		inner_dim = self.heads * self.dim_head

		y = nn.LayerNorm()(x)
		qkv = nn.Dense(inner_dim * 3, use_bias = False)(y)
		qkv = qkv.reshape(b, n, 3, self.heads, self.dim_head)
		qkv = qkv.transpose(2, 0, 3, 1, 4)
		q, k, v = qkv[0], qkv[1], qkv[2]

		scale = self.dim_head ** -0.5
		logits = jnp.einsum('bhqd,bhkd->bhqk', q, k) * scale
		attn = nn.softmax(logits, axis = -1)
		attn = nn.Dropout(self.dropout)(attn, deterministic = deterministic)

		out = jnp.einsum('bhqk,bhkd->bhqd', attn, v)
		out = out.transpose(0, 2, 1, 3).reshape(b, n, inner_dim)

		project_out = not (self.heads == 1 and self.dim_head == self.dim)
		if project_out:
			out = nn.Dense(self.dim)(out)
			out = nn.Dropout(self.dropout)(out, deterministic = deterministic)
		return out

class Transformer(nn.Module):
	dim: int
	depth: int
	heads: int
	dim_head: int
	mlp_dim: int
	dropout: float = 0.

	@nn.compact
	def __call__(self, x, *, deterministic: bool):
		for i in range(self.depth):
			attn_out = Attention(self.dim, self.heads, self.dim_head, self.dropout, name=f"attn_{i}")(x, deterministic = deterministic)
			x = x + attn_out

			ff_out = FeedForward(self.dim, self.mlp_dim, self.dropout, name=f"ff_{i}")(x, deterministic = deterministic)
			x = x + ff_out

		return nn.LayerNorm()(x)

class ViT3D(nn.Module):
	image_size: tuple
	patch_size: tuple
	frames: int
	frame_patch_size: int
	num_classes: int
	dim: int
	depth: int
	heads: int
	mlp_dim: int
	pool: str = 'cls'
	channels: int = 3
	dim_head: int = 64
	dropout: float = 0.
	emb_dropout: float = 0.

	@nn.compact
	def __call__(self, video, *, train: bool):
		b, t, h, w, _ = video.shape
		p_h, p_w = self.patch_size
		p_t	   = self.frame_patch_size

		n_t = t // p_t
		n_h = h // p_h
		n_w = w // p_w
		num_patches = n_t * n_h * n_w
		
		x = rearrange(
			video,
			'b (n_t p_t) (n_h p_h) (n_w p_w) c -> b (n_t n_h n_w) (p_t p_h p_w c)',
			n_t = n_t, p_t = p_t, n_h = n_h, p_h = p_h, n_w = n_w, p_w = p_w
		)

		x = nn.LayerNorm()(x)
		x = nn.Dense(self.dim)(x)
		x = nn.LayerNorm()(x)

		cls_token = self.param('cls_token', normal(stddev = 1.), (1, 1, self.dim))
		cls_tokens = repeat(cls_token, '1 1 d -> b 1 d', b=b)
		x = jnp.concatenate([cls_tokens, x], axis = 1)

		pos_emb = self.param(
			'pos_embedding', normal(stddev = 1.), (1, num_patches + 1, self.dim))
		x = x + pos_emb
		x = nn.Dropout(self.emb_dropout)(x, deterministic = not train)
		x = Transformer(self.dim, self.depth, self.heads, self.dim_head,
						self.mlp_dim, self.dropout)(x, deterministic = not train)
		
		if self.pool == 'mean':
			x = x.mean(axis = 1)
		else:
			x = x[:, 0]

		x = nn.LayerNorm()(x)
		x = nn.Dense(self.num_classes)(x)
		return x