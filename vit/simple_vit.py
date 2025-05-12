import jax.numpy as jnp
from flax import linen as nn
from einops import rearrange

def posemb_sincos_2d(h: int, w: int, dim: int, temperature: float = 10000., dtype = jnp.float32) -> jnp.ndarray:
	y, x = jnp.meshgrid(jnp.arange(h), jnp.arange(w), indexing='ij')
	y = y.reshape(-1)[:, None]
	x = x.reshape(-1)[:, None]

	d_quarter = dim // 4
	omega = jnp.arange(d_quarter, dtype=dtype) / (d_quarter - 1)
	omega = 1. / (temperature ** omega)

	y = y * omega[None, :]
	x = x * omega[None, :]

	pe = jnp.concatenate([x.sin(), x.cos(), y.sin(), y.cos()], axis = 1)
	return pe.astype(dtype)

class FeedForward(nn.Module):
	dim: int
	hidden_dim: int

	@nn.compact
	def __call__(self, x):
		y = nn.LayerNorm()(x)
		y = nn.Dense(self.hidden_dim)(y)
		y = nn.gelu(y)
		y = nn.Dense(self.dim)(y)
		return y

class Attention(nn.Module):
	dim: int
	heads: int = 8
	dim_head: int = 64

	@nn.compact
	def __call__(self, x):
		b, n, _ = x.shape
		inner_dim = self.heads * self.dim_head
		scale = self.dim_head ** -0.5

		y = nn.LayerNorm()(x)
		qkv = nn.Dense(inner_dim * 3, use_bias = False)(y)
		qkv = qkv.reshape(b, n, 3, self.heads, self.dim_head)
		qkv = qkv.transpose(2, 0, 3, 1, 4)
		q, k, v = qkv[0], qkv[1], qkv[2]

		dots = jnp.einsum('bhqd,bhkd->bhqk', q, k) * scale
		attn = nn.softmax(dots, axis = -1)
		out = jnp.einsum('bhqk,bhkd->bhqd', attn, v)
		out = out.transpose(0, 2, 1, 3).reshape(b, n, inner_dim)

		out = nn.Dense(self.dim, use_bias = False)(out)
		return out

class Transformer(nn.Module):
	dim: int
	depth: int
	heads: int
	dim_head: int
	mlp_dim: int

	@nn.compact
	def __call__(self, x):
		for i in range(self.depth):
			attn_out = Attention(self.dim, self.heads, self.dim_head, name = f"attn_{i}")(x)
			x = x + attn_out

			ff_out = FeedForward(self.dim, self.mlp_dim, name = f"ff_{i}")(x)
			x = x + ff_out

		x = nn.LayerNorm()(x)
		return x

class SimpleViT(nn.Module):
	image_size: tuple
	patch_size: tuple
	num_classes: int
	dim: int
	depth: int
	heads: int
	mlp_dim: int
	channels: int = 3
	dim_head: int = 64

	@nn.compact
	def __call__(self, img):
		_, _, h, w = img.shape
		p_h, p_w = self.patch_size

		x = rearrange(img, 'b c (h ph) (w pw) -> b (h w) (ph pw c)', ph = p_h, pw = p_w)
		x = nn.LayerNorm()(x)
		x = nn.Dense(self.dim)(x)
		x = nn.LayerNorm()(x)

		grid_h = h // p_h
		grid_w = w // p_w
		pe = posemb_sincos_2d(grid_h, grid_w, self.dim, dtype = x.dtype)
		
		x = x + pe[None, :, :]
		x = Transformer(self.dim, self.depth, self.heads, self.dim_head, self.mlp_dim)(x)
		x = x.mean(axis = 1)

		logits = nn.Dense(self.num_classes)(x)
		return logits