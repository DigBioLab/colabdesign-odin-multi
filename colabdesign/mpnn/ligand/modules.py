import functools

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

from colabdesign.shared.prng import SafeKey
from ..utils import cat_neighbors_nodes, gather_nodes, get_ar_mask

Gelu = functools.partial(jax.nn.gelu, approximate=False)


def gather_edges(edges, neighbor_idx):
  idx = neighbor_idx[..., None]
  if edges.ndim == 2:
    return jnp.take_along_axis(edges, neighbor_idx, axis=1)
  return jnp.take_along_axis(edges, jnp.repeat(idx, edges.shape[-1], axis=-1), axis=1)


def get_nearest_ligand_context(CB, mask, Y, Y_t, Y_m, number_of_ligand_atoms):
  mask_CBY = mask[:, None] * Y_m[None, :]
  l2_ab = jnp.sum((CB[:, None, :] - Y[None, :, :]) ** 2, axis=-1)
  l2_ab = l2_ab * mask_CBY + (1.0 - mask_CBY) * 1000.0

  num_update = min(number_of_ligand_atoms, Y.shape[0])
  nn_idx = jnp.argsort(l2_ab, axis=-1)[:, :num_update]
  l2_ab_nn = jnp.take_along_axis(l2_ab, nn_idx, axis=1)
  d_ab_closest = jnp.sqrt(l2_ab_nn[:, 0] + 1e-8)

  y_tmp = Y[nn_idx]
  y_t_tmp = Y_t[nn_idx]
  y_m_tmp = Y_m[nn_idx]

  y_out = jnp.zeros((CB.shape[0], number_of_ligand_atoms, 3), dtype=jnp.float32)
  y_t_out = jnp.zeros((CB.shape[0], number_of_ligand_atoms), dtype=jnp.int32)
  y_m_out = jnp.zeros((CB.shape[0], number_of_ligand_atoms), dtype=jnp.float32)

  y_out = y_out.at[:, :num_update].set(y_tmp)
  y_t_out = y_t_out.at[:, :num_update].set(y_t_tmp)
  y_m_out = y_m_out.at[:, :num_update].set(y_m_tmp.astype(jnp.float32))
  return y_out, y_t_out, y_m_out, d_ab_closest


def get_nearest_context_per_residue(CB, Y, Y_t, Y_m, number_of_ligand_atoms):
  l2_ab = jnp.sum((CB[:, None, :] - Y) ** 2, axis=-1)
  l2_ab = l2_ab * Y_m + (1.0 - Y_m) * 1000.0

  num_update = min(number_of_ligand_atoms, Y.shape[1])
  nn_idx = jnp.argsort(l2_ab, axis=-1)[:, :num_update]

  y_tmp = jnp.take_along_axis(Y, nn_idx[..., None], axis=1)
  y_t_tmp = jnp.take_along_axis(Y_t, nn_idx, axis=1)
  y_m_tmp = jnp.take_along_axis(Y_m, nn_idx, axis=1)

  y_out = jnp.zeros((CB.shape[0], number_of_ligand_atoms, 3), dtype=jnp.float32)
  y_t_out = jnp.zeros((CB.shape[0], number_of_ligand_atoms), dtype=jnp.int32)
  y_m_out = jnp.zeros((CB.shape[0], number_of_ligand_atoms), dtype=jnp.float32)

  y_out = y_out.at[:, :num_update].set(y_tmp)
  y_t_out = y_t_out.at[:, :num_update].set(y_t_tmp)
  y_m_out = y_m_out.at[:, :num_update].set(y_m_tmp.astype(jnp.float32))
  return y_out, y_t_out, y_m_out


class dropout_cust(hk.Module):
  def __init__(self, rate, name=None) -> None:
    super().__init__(name=name)
    self.rate = rate
    self.safe_key = SafeKey(hk.next_rng_key())

  def __call__(self, x):
    self.safe_key, use_key = self.safe_key.split()
    return hk.dropout(use_key.get(), self.rate, x)


class PositionWiseFeedForward(hk.Module):
  def __init__(self, num_hidden, num_ff, prefix, name=None):
    super().__init__(name=name)
    self.W_in = hk.Linear(num_ff, with_bias=True, name=f"{prefix}_dense_W_in")
    self.W_out = hk.Linear(num_hidden, with_bias=True, name=f"{prefix}_dense_W_out")
    self.act = Gelu

  def __call__(self, h_V):
    h = self.act(self.W_in(h_V))
    h = self.W_out(h)
    return h


class PositionalEncodings(hk.Module):
  def __init__(self, num_embeddings, max_relative_feature=32, name=None):
    super().__init__(name=name)
    self.num_embeddings = num_embeddings
    self.max_relative_feature = max_relative_feature
    self.linear = hk.Linear(num_embeddings, name="embedding_linear")

  def __call__(self, offset, mask):
    d = jnp.clip(offset + self.max_relative_feature, 0, 2 * self.max_relative_feature) * mask + \
      (1 - mask) * (2 * self.max_relative_feature + 1)
    d_onehot = jax.nn.one_hot(d, 2 * self.max_relative_feature + 2)
    return self.linear(d_onehot)


class EmbedToken(hk.Module):
  def __init__(self, vocab_size, embed_dim, name=None):
    super().__init__(name=name)
    self.vocab_size = vocab_size
    self.embed_dim = embed_dim
    self.w_init = hk.initializers.TruncatedNormal()

  @property
  def embeddings(self):
    return hk.get_parameter("W_s", [self.vocab_size, self.embed_dim], init=self.w_init)

  def __call__(self, arr):
    if jnp.issubdtype(arr.dtype, jnp.integer):
      one_hot = jax.nn.one_hot(arr, self.vocab_size)
    else:
      one_hot = arr
    return jnp.tensordot(one_hot, self.embeddings, 1)


class EncLayer(hk.Module):
  def __init__(self, num_hidden, dropout=0.1, scale=30, prefix="enc", name=None):
    super().__init__(name=name)
    self.scale = scale
    self.dropout1 = dropout_cust(dropout, name=f"{prefix}_dropout1")
    self.dropout2 = dropout_cust(dropout, name=f"{prefix}_dropout2")
    self.dropout3 = dropout_cust(dropout, name=f"{prefix}_dropout3")
    self.norm1 = hk.LayerNorm(-1, create_scale=True, create_offset=True, name=f"{prefix}_norm1")
    self.norm2 = hk.LayerNorm(-1, create_scale=True, create_offset=True, name=f"{prefix}_norm2")
    self.norm3 = hk.LayerNorm(-1, create_scale=True, create_offset=True, name=f"{prefix}_norm3")
    self.W1 = hk.Linear(num_hidden, with_bias=True, name=f"{prefix}_W1")
    self.W2 = hk.Linear(num_hidden, with_bias=True, name=f"{prefix}_W2")
    self.W3 = hk.Linear(num_hidden, with_bias=True, name=f"{prefix}_W3")
    self.W11 = hk.Linear(num_hidden, with_bias=True, name=f"{prefix}_W11")
    self.W12 = hk.Linear(num_hidden, with_bias=True, name=f"{prefix}_W12")
    self.W13 = hk.Linear(num_hidden, with_bias=True, name=f"{prefix}_W13")
    self.act = Gelu
    self.dense = PositionWiseFeedForward(num_hidden, num_hidden * 4, prefix=prefix, name="position_wise_feed_forward")

  def __call__(self, h_V, h_E, E_idx, mask_V=None, mask_attend=None):
    h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)
    h_V_expand = jnp.tile(jnp.expand_dims(h_V, -2), [1, h_EV.shape[-2], 1])
    h_EV = jnp.concatenate([h_V_expand, h_EV], -1)

    h_message = self.W3(self.act(self.W2(self.act(self.W1(h_EV)))))
    if mask_attend is not None:
      h_message = jnp.expand_dims(mask_attend, -1) * h_message
    dh = jnp.sum(h_message, -2) / self.scale
    h_V = self.norm1(h_V + self.dropout1(dh))

    dh = self.dense(h_V)
    h_V = self.norm2(h_V + self.dropout2(dh))
    if mask_V is not None:
      h_V = mask_V[:, None] * h_V

    h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)
    h_V_expand = jnp.tile(jnp.expand_dims(h_V, -2), [1, h_EV.shape[-2], 1])
    h_EV = jnp.concatenate([h_V_expand, h_EV], -1)
    h_message = self.W13(self.act(self.W12(self.act(self.W11(h_EV)))))
    h_E = self.norm3(h_E + self.dropout3(h_message))
    return h_V, h_E


class DecLayer(hk.Module):
  def __init__(self, num_hidden, dropout=0.1, scale=30, prefix="dec", name=None):
    super().__init__(name=name)
    self.scale = scale
    self.dropout1 = dropout_cust(dropout, name=f"{prefix}_dropout1")
    self.dropout2 = dropout_cust(dropout, name=f"{prefix}_dropout2")
    self.norm1 = hk.LayerNorm(-1, create_scale=True, create_offset=True, name=f"{prefix}_norm1")
    self.norm2 = hk.LayerNorm(-1, create_scale=True, create_offset=True, name=f"{prefix}_norm2")
    self.W1 = hk.Linear(num_hidden, with_bias=True, name=f"{prefix}_W1")
    self.W2 = hk.Linear(num_hidden, with_bias=True, name=f"{prefix}_W2")
    self.W3 = hk.Linear(num_hidden, with_bias=True, name=f"{prefix}_W3")
    self.act = Gelu
    self.dense = PositionWiseFeedForward(num_hidden, num_hidden * 4, prefix=prefix, name="position_wise_feed_forward")

  def __call__(self, h_V, h_E, mask_V=None, mask_attend=None):
    h_V_expand = jnp.tile(jnp.expand_dims(h_V, -2), [1, h_E.shape[-2], 1])
    h_EV = jnp.concatenate([h_V_expand, h_E], -1)
    h_message = self.W3(self.act(self.W2(self.act(self.W1(h_EV)))))
    if mask_attend is not None:
      h_message = jnp.expand_dims(mask_attend, -1) * h_message
    dh = jnp.sum(h_message, -2) / self.scale
    h_V = self.norm1(h_V + self.dropout1(dh))
    dh = self.dense(h_V)
    h_V = self.norm2(h_V + self.dropout2(dh))
    if mask_V is not None:
      h_V = mask_V[:, None] * h_V
    return h_V


class DecLayerJ(hk.Module):
  def __init__(self, num_hidden, dropout=0.1, scale=30, prefix="yctx", name=None):
    super().__init__(name=name)
    self.scale = scale
    self.dropout1 = dropout_cust(dropout, name=f"{prefix}_dropout1")
    self.dropout2 = dropout_cust(dropout, name=f"{prefix}_dropout2")
    self.norm1 = hk.LayerNorm(-1, create_scale=True, create_offset=True, name=f"{prefix}_norm1")
    self.norm2 = hk.LayerNorm(-1, create_scale=True, create_offset=True, name=f"{prefix}_norm2")
    self.W1 = hk.Linear(num_hidden, with_bias=True, name=f"{prefix}_W1")
    self.W2 = hk.Linear(num_hidden, with_bias=True, name=f"{prefix}_W2")
    self.W3 = hk.Linear(num_hidden, with_bias=True, name=f"{prefix}_W3")
    self.act = Gelu
    self.dense = PositionWiseFeedForward(num_hidden, num_hidden * 4, prefix=prefix, name="position_wise_feed_forward")

  def __call__(self, h_V, h_E, mask_V=None, mask_attend=None):
    h_V_expand = jnp.tile(jnp.expand_dims(h_V, -2), [1, 1, h_E.shape[-2], 1])
    h_EV = jnp.concatenate([h_V_expand, h_E], -1)
    h_message = self.W3(self.act(self.W2(self.act(self.W1(h_EV)))))
    if mask_attend is not None:
      h_message = jnp.expand_dims(mask_attend, -1) * h_message
    dh = jnp.sum(h_message, -2) / self.scale
    h_V = self.norm1(h_V + self.dropout1(dh))
    dh = self.dense(h_V)
    h_V = self.norm2(h_V + self.dropout2(dh))
    if mask_V is not None:
      h_V = mask_V[..., None] * h_V
    return h_V


class ProteinFeaturesLigand(hk.Module):
  def __init__(self,
               edge_features,
               node_features,
               num_positional_embeddings=16,
               num_rbf=16,
               top_k=32,
               augment_eps=0.0,
               atom_context_num=25,
               use_side_chains=False,
               name="protein_features_ligand"):
    super().__init__(name=name)
    self.use_side_chains = use_side_chains
    self.edge_features = edge_features
    self.node_features = node_features
    self.top_k = top_k
    self.augment_eps = augment_eps
    self.num_rbf = num_rbf
    self.num_positional_embeddings = num_positional_embeddings
    self.atom_context_num = atom_context_num

    self.embeddings = PositionalEncodings(num_positional_embeddings, name="positional_encodings")
    self.edge_embedding = hk.Linear(edge_features, with_bias=False, name="edge_embedding")
    self.norm_edges = hk.LayerNorm(-1, create_scale=True, create_offset=True, name="norm_edges")
    self.node_project_down = hk.Linear(node_features, with_bias=True, name="node_project_down")
    self.norm_nodes = hk.LayerNorm(-1, create_scale=True, create_offset=True, name="norm_nodes")
    self.type_linear = hk.Linear(64, with_bias=True, name="type_linear")
    self.y_nodes = hk.Linear(node_features, with_bias=False, name="y_nodes")
    self.y_edges = hk.Linear(node_features, with_bias=False, name="y_edges")
    self.norm_y_edges = hk.LayerNorm(-1, create_scale=True, create_offset=True, name="norm_y_edges")
    self.norm_y_nodes = hk.LayerNorm(-1, create_scale=True, create_offset=True, name="norm_y_nodes")
    self.safe_key = SafeKey(hk.next_rng_key())

    self.side_chain_atom_types = jnp.array([
      6, 6, 6, 8, 8, 16, 6, 6, 6, 7, 7, 8, 8, 16, 6, 6,
      6, 6, 7, 7, 7, 8, 8, 6, 7, 7, 8, 6, 6, 6, 7, 8
    ], dtype=jnp.int32)
    self.periodic_group = jnp.array([
      0, 1, 18, 1, 2, 13, 14, 15, 16, 17, 18, 1, 2, 13, 14, 15, 16, 17, 18,
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 1, 2, 3,
      4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 1, 2, 3, 3, 3, 3,
      3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
      15, 16, 17, 18, 1, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4,
      5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18
    ], dtype=jnp.int32)
    self.periodic_period = jnp.array([
      0, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4,
      4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
      5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
      6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
      7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7
    ], dtype=jnp.int32)

  def _make_angle_features(self, A, B, C, Y):
    normalize = lambda x: x / jnp.sqrt(jnp.sum(x ** 2, axis=-1, keepdims=True) + 1e-8)
    v1 = A - B
    v2 = C - B
    e1 = normalize(v1)
    e1_v2_dot = jnp.einsum("li, li -> l", e1, v2)[..., None]
    u2 = v2 - e1 * e1_v2_dot
    e2 = normalize(u2)
    e3 = jnp.cross(e1, e2, axis=-1)
    r_residue = jnp.concatenate((e1[:, :, None], e2[:, :, None], e3[:, :, None]), axis=-1)
    local_vectors = jnp.einsum("lqp, lyp -> lyq", jnp.swapaxes(r_residue, -1, -2), Y - B[:, None, :])
    rxy = jnp.sqrt(local_vectors[..., 0] ** 2 + local_vectors[..., 1] ** 2 + 1e-8)
    f1 = local_vectors[..., 0] / rxy
    f2 = local_vectors[..., 1] / rxy
    rxyz = jnp.linalg.norm(local_vectors, axis=-1) + 1e-8
    f3 = rxy / rxyz
    f4 = local_vectors[..., 2] / rxyz
    return jnp.concatenate([f1[..., None], f2[..., None], f3[..., None], f4[..., None]], axis=-1)

  def _dist(self, X, mask, eps=1e-6):
    mask_2d = mask[:, None] * mask[None, :]
    dX = X[:, None, :] - X[None, :, :]
    D = mask_2d * jnp.sqrt(jnp.sum(dX**2, axis=-1) + eps)
    d_max = jnp.max(D, axis=-1, keepdims=True)
    d_adjust = D + (1.0 - mask_2d) * d_max
    k = min(self.top_k, X.shape[0])
    e_idx = jnp.argsort(d_adjust, axis=-1)[:, :k]
    d_neighbors = jnp.take_along_axis(d_adjust, e_idx, axis=1)
    return d_neighbors, e_idx

  def _rbf(self, D):
    d_min, d_max, d_count = 2.0, 22.0, self.num_rbf
    d_mu = jnp.linspace(d_min, d_max, d_count).reshape((1,) * D.ndim + (d_count,))
    d_sigma = (d_max - d_min) / d_count
    return jnp.exp(-(((D[..., None] - d_mu) / d_sigma) ** 2))

  def _get_rbf(self, A, B, E_idx):
    d_ab = jnp.sqrt(jnp.sum((A[:, None, :] - B[None, :, :]) ** 2, axis=-1) + 1e-6)
    d_ab_neighbors = gather_edges(d_ab[:, :, None], E_idx)[:, :, 0]
    return self._rbf(d_ab_neighbors)

  def __call__(self, I):
    Y = I["Y"]
    Y_m = I["Y_m"]
    Y_t = I["Y_t"]
    X = I["X"]
    mask = I["mask"]
    R_idx = I["residue_idx"]
    chain_labels = I["chain_idx"]
    mask_XY = I["mask_XY"]

    if self.augment_eps > 0:
      self.safe_key, use_key = self.safe_key.split()
      key_x, key_y = jax.random.split(use_key.get())
      X = X + self.augment_eps * jax.random.normal(key_x, X.shape)
      Y = Y + self.augment_eps * jax.random.normal(key_y, Y.shape)

    Ca, N, C, O = X[:, 1, :], X[:, 0, :], X[:, 2, :], X[:, 3, :]
    b = Ca - N
    c = C - Ca
    a = jnp.cross(b, c, axis=-1)
    Cb = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + Ca

    d_neighbors, E_idx = self._dist(Ca, mask)
    rbf_all = [
      self._rbf(d_neighbors),
      self._get_rbf(N, N, E_idx),
      self._get_rbf(C, C, E_idx),
      self._get_rbf(O, O, E_idx),
      self._get_rbf(Cb, Cb, E_idx),
      self._get_rbf(Ca, N, E_idx),
      self._get_rbf(Ca, C, E_idx),
      self._get_rbf(Ca, O, E_idx),
      self._get_rbf(Ca, Cb, E_idx),
      self._get_rbf(N, C, E_idx),
      self._get_rbf(N, O, E_idx),
      self._get_rbf(N, Cb, E_idx),
      self._get_rbf(Cb, C, E_idx),
      self._get_rbf(Cb, O, E_idx),
      self._get_rbf(O, C, E_idx),
      self._get_rbf(N, Ca, E_idx),
      self._get_rbf(C, Ca, E_idx),
      self._get_rbf(O, Ca, E_idx),
      self._get_rbf(Cb, Ca, E_idx),
      self._get_rbf(C, N, E_idx),
      self._get_rbf(O, N, E_idx),
      self._get_rbf(Cb, N, E_idx),
      self._get_rbf(C, Cb, E_idx),
      self._get_rbf(O, Cb, E_idx),
      self._get_rbf(C, O, E_idx),
    ]
    rbf_all = jnp.concatenate(tuple(rbf_all), axis=-1)

    offset = R_idx[:, None] - R_idx[None, :]
    offset = gather_edges(offset[:, :, None], E_idx)[:, :, 0]
    d_chains = (chain_labels[:, None] == chain_labels[None, :]).astype(jnp.int32)
    E_chains = gather_edges(d_chains[:, :, None], E_idx)[:, :, 0]
    E_positional = self.embeddings(offset.astype(jnp.int32), E_chains)
    E = jnp.concatenate((E_positional, rbf_all), axis=-1)
    E = self.edge_embedding(E)
    E = self.norm_edges(E)

    if self.use_side_chains and "xyz_37" in I and "xyz_37_m" in I:
      xyz_37 = I["xyz_37"]
      xyz_37_m = I["xyz_37_m"]
      chain_mask = I.get("chain_mask", jnp.zeros(mask.shape, dtype=jnp.float32))
      e_idx_sub = E_idx[:, :16]
      xyz_37_m = xyz_37_m * (1.0 - chain_mask[:, None])
      r_m = gather_nodes(xyz_37_m[:, 5:], e_idx_sub).reshape((mask.shape[0], -1))
      x_sidechain = xyz_37[:, 5:, :].reshape((mask.shape[0], -1))
      r = gather_nodes(x_sidechain, e_idx_sub).reshape((mask.shape[0], -1, 3))
      r_t = jnp.broadcast_to(self.side_chain_atom_types[None, :], (mask.shape[0], self.side_chain_atom_types.shape[0]))
      r_t = gather_nodes(r_t, e_idx_sub).reshape((mask.shape[0], -1))
      Y = jnp.concatenate((r, Y), axis=1)
      Y_m = jnp.concatenate((r_m, Y_m), axis=1)
      Y_t = jnp.concatenate((r_t, Y_t), axis=1)
      mask_XY = mask_XY * jnp.max(Y_m, axis=-1)
      Y, Y_t, Y_m = get_nearest_context_per_residue(Cb, Y, Y_t, Y_m, self.atom_context_num)

    Y_t = Y_t.astype(jnp.int32)
    Y_t_g = self.periodic_group[Y_t]
    Y_t_p = self.periodic_period[Y_t]
    Y_t_g_1hot = jax.nn.one_hot(Y_t_g, 19)
    Y_t_p_1hot = jax.nn.one_hot(Y_t_p, 8)
    Y_t_1hot = jax.nn.one_hot(Y_t, 120)
    Y_t_1hot_full = jnp.concatenate([Y_t_1hot, Y_t_g_1hot, Y_t_p_1hot], axis=-1)
    Y_t_1hot_proj = self.type_linear(Y_t_1hot_full.astype(jnp.float32))

    D_N_Y = self._rbf(jnp.sqrt(jnp.sum((N[:, None, :] - Y) ** 2, axis=-1) + 1e-6))
    D_Ca_Y = self._rbf(jnp.sqrt(jnp.sum((Ca[:, None, :] - Y) ** 2, axis=-1) + 1e-6))
    D_C_Y = self._rbf(jnp.sqrt(jnp.sum((C[:, None, :] - Y) ** 2, axis=-1) + 1e-6))
    D_O_Y = self._rbf(jnp.sqrt(jnp.sum((O[:, None, :] - Y) ** 2, axis=-1) + 1e-6))
    D_Cb_Y = self._rbf(jnp.sqrt(jnp.sum((Cb[:, None, :] - Y) ** 2, axis=-1) + 1e-6))
    f_angles = self._make_angle_features(N, Ca, C, Y)
    d_all = jnp.concatenate((D_N_Y, D_Ca_Y, D_C_Y, D_O_Y, D_Cb_Y, Y_t_1hot_proj, f_angles), axis=-1)
    V = self.node_project_down(d_all)
    V = self.norm_nodes(V)

    Y_edges = self._rbf(jnp.sqrt(jnp.sum((Y[:, :, None, :] - Y[:, None, :, :]) ** 2, axis=-1) + 1e-6))
    Y_edges = self.y_edges(Y_edges)
    Y_nodes = self.y_nodes(Y_t_1hot_full.astype(jnp.float32))
    Y_edges = self.norm_y_edges(Y_edges)
    Y_nodes = self.norm_y_nodes(Y_nodes)

    return V, E, E_idx, Y_nodes, Y_edges, Y_m, mask_XY


class LigandMPNN(hk.Module):
  def __init__(self,
               num_letters,
               node_features,
               edge_features,
               hidden_dim,
               num_encoder_layers=3,
               num_decoder_layers=3,
               vocab=21,
               k_neighbors=32,
               atom_context_num=25,
               augment_eps=0.0,
               dropout=0.0,
               ligand_mpnn_use_side_chain_context=False,
               name="ligand_mpnn"):
    super().__init__(name=name)
    self.hidden_dim = hidden_dim
    self.atom_context_num = atom_context_num
    self.features = ProteinFeaturesLigand(
      edge_features=edge_features,
      node_features=node_features,
      top_k=k_neighbors,
      augment_eps=augment_eps,
      atom_context_num=atom_context_num,
      use_side_chains=ligand_mpnn_use_side_chain_context,
    )
    self.W_e = hk.Linear(hidden_dim, with_bias=True, name="W_e")
    self.W_s = EmbedToken(vocab_size=vocab, embed_dim=hidden_dim, name="embed_token")
    self.W_v = hk.Linear(hidden_dim, with_bias=True, name="W_v")
    self.W_c = hk.Linear(hidden_dim, with_bias=True, name="W_c")
    self.W_nodes_y = hk.Linear(hidden_dim, with_bias=True, name="W_nodes_y")
    self.W_edges_y = hk.Linear(hidden_dim, with_bias=True, name="W_edges_y")
    self.V_C = hk.Linear(hidden_dim, with_bias=False, name="V_C")
    self.V_C_norm = hk.LayerNorm(-1, create_scale=True, create_offset=True, name="V_C_norm")
    self.dropout = dropout_cust(dropout, name="ligand_residual_dropout")

    self.encoder_layers = [
      EncLayer(hidden_dim, dropout=dropout, prefix=f"enc{i}", name=("enc_layer" if i == 0 else f"enc_layer_{i}"))
      for i in range(num_encoder_layers)
    ]
    self.decoder_layers = [
      DecLayer(hidden_dim, dropout=dropout, prefix=f"dec{i}", name=("dec_layer" if i == 0 else f"dec_layer_{i}"))
      for i in range(num_decoder_layers)
    ]
    self.context_encoder_layers = [
      DecLayer(hidden_dim, dropout=dropout, prefix=f"context{i}", name=("context_encoder_layer" if i == 0 else f"context_encoder_layer_{i}"))
      for i in range(2)
    ]
    self.y_context_encoder_layers = [
      DecLayerJ(hidden_dim, dropout=dropout, prefix=f"yctx{i}", name=("y_context_encoder_layer" if i == 0 else f"y_context_encoder_layer_{i}"))
      for i in range(2)
    ]
    self.W_out = hk.Linear(num_letters, with_bias=True, name="W_out")

  def encode(self, I):
    V, E, E_idx, Y_nodes, Y_edges, Y_m, mask_XY = self.features(I)
    h_V = jnp.zeros((E.shape[0], E.shape[-1]), dtype=E.dtype)
    h_E = self.W_e(E)
    h_E_context = self.W_v(V)

    mask_attend = gather_nodes(I["mask"][:, None], E_idx).squeeze(-1)
    mask_attend = I["mask"][:, None] * mask_attend
    for layer in self.encoder_layers:
      h_V, h_E = layer(h_V, h_E, E_idx, I["mask"], mask_attend)

    h_V_C = self.W_c(h_V)
    Y_m_edges = Y_m[:, :, None] * Y_m[:, None, :]
    Y_nodes = self.W_nodes_y(Y_nodes)
    Y_edges = self.W_edges_y(Y_edges)
    for i in range(len(self.context_encoder_layers)):
      Y_nodes = self.y_context_encoder_layers[i](Y_nodes, Y_edges, Y_m, Y_m_edges)
      h_E_context_cat = jnp.concatenate([h_E_context, Y_nodes], axis=-1)
      h_V_C = self.context_encoder_layers[i](h_V_C, h_E_context_cat, I["mask"], Y_m)

    h_V_C = self.V_C(h_V_C)
    h_V = h_V + self.V_C_norm(self.dropout(h_V_C))
    return h_V, h_E, E_idx, mask_XY

  def score(self, I):
    h_V, h_E, E_idx, mask_XY = self.encode(I)
    h_EX_encoder = cat_neighbors_nodes(jnp.zeros_like(h_V), h_E, E_idx)
    h_EXV_encoder = cat_neighbors_nodes(h_V, h_EX_encoder, E_idx)

    if "S" not in I:
      h_EXV_encoder_fw = h_EXV_encoder
      for layer in self.decoder_layers:
        h_V = layer(h_V, h_EXV_encoder_fw, I["mask"])
      decoding_order = None
    else:
      h_S = self.W_s(I["S"])
      h_ES = cat_neighbors_nodes(h_S, h_E, E_idx)
      if "ar_mask" in I:
        decoding_order = None
        ar_mask = I["ar_mask"]
      else:
        decoding_order = I["decoding_order"]
        ar_mask = get_ar_mask(decoding_order)

      mask_attend = jnp.take_along_axis(ar_mask, E_idx, axis=1)
      mask_1D = I["mask"][:, None]
      mask_bw = mask_1D * mask_attend
      mask_fw = mask_1D * (1.0 - mask_attend)
      h_EXV_encoder_fw = mask_fw[..., None] * h_EXV_encoder
      for layer in self.decoder_layers:
        h_ESV = cat_neighbors_nodes(h_V, h_ES, E_idx)
        h_ESV = mask_bw[..., None] * h_ESV + h_EXV_encoder_fw
        h_V = layer(h_V, h_ESV, I["mask"])

    logits = self.W_out(h_V)
    return {"logits": logits, "decoding_order": decoding_order, "S": I.get("S", None), "mask_XY": mask_XY}

  def sample(self, I):
    key = hk.next_rng_key()
    temperature = I.get("temperature", 1.0)
    h_V, h_E, E_idx, mask_XY = self.encode(I)
    ar_mask = I.get("ar_mask", get_ar_mask(I["decoding_order"]))
    mask_attend = jnp.take_along_axis(ar_mask, E_idx, axis=1)
    mask_1D = I["mask"][:, None]
    mask_bw = mask_1D * mask_attend
    mask_fw = mask_1D * (1.0 - mask_attend)

    h_EX_encoder = cat_neighbors_nodes(jnp.zeros_like(h_V), h_E, E_idx)
    h_EXV_encoder = cat_neighbors_nodes(h_V, h_EX_encoder, E_idx)
    h_EXV_encoder = mask_fw[..., None] * h_EXV_encoder

    def fwd(x, t, sample_key):
      h_EXV_encoder_t = h_EXV_encoder[t]
      E_idx_t = E_idx[t]
      mask_t = I["mask"][t]
      mask_bw_t = mask_bw[t]
      h_ES_t = cat_neighbors_nodes(x["h_S"], h_E[t], E_idx_t)

      for l, layer in enumerate(self.decoder_layers):
        h_V_layer = x["h_V"][l]
        h_ESV_decoder_t = cat_neighbors_nodes(h_V_layer, h_ES_t, E_idx_t)
        h_ESV_t = mask_bw_t[..., None] * h_ESV_decoder_t + h_EXV_encoder_t
        h_V_t = layer(h_V_layer[t], h_ESV_t, mask_V=mask_t)
        x["h_V"] = x["h_V"].at[l + 1, t].set(h_V_t)

      logits_t = self.W_out(h_V_t)
      x["logits"] = x["logits"].at[t].set(logits_t)
      if "bias" in I:
        logits_t = logits_t + I["bias"][t]
      logits_t = logits_t / temperature + jax.random.gumbel(sample_key, logits_t.shape)
      logits_t = logits_t.mean(0, keepdims=True)
      S_t = jax.nn.one_hot(logits_t[..., :20].argmax(-1), 21)
      x["h_S"] = x["h_S"].at[t].set(self.W_s(S_t))
      x["S"] = x["S"].at[t].set(S_t)
      return x, None

    X = {
      "h_S": jnp.zeros_like(h_V),
      "h_V": jnp.array([h_V] + [jnp.zeros_like(h_V)] * len(self.decoder_layers)),
      "S": jnp.zeros((I["X"].shape[0], 21)),
      "logits": jnp.zeros((I["X"].shape[0], 21)),
    }

    t = I["decoding_order"]
    if t.ndim == 1:
      t = t[:, None]
    XS = {"t": t, "key": jax.random.split(key, t.shape[0])}
    X = hk.scan(lambda x, xs: fwd(x, xs["t"], xs["key"]), X, XS)[0]
    return {"S": X["S"], "logits": X["logits"], "decoding_order": t, "mask_XY": mask_XY}


class LigandRunModel:
  def __init__(self, config) -> None:
    self.config = config

    def _forward_score(inputs):
      model = LigandMPNN(**self.config)
      return model.score(inputs)
    self.score = hk.transform(_forward_score).apply
    self.score_init = hk.transform(_forward_score).init

    def _forward_sample(inputs):
      model = LigandMPNN(**self.config)
      return model.sample(inputs)
    self.sample = hk.transform(_forward_sample).apply
    self.sample_init = hk.transform(_forward_sample).init
