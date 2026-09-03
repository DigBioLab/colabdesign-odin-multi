import copy
import os
import random

import jax
import jax.numpy as jnp
import joblib
import numpy as np

from colabdesign.af.alphafold.common import residue_constants
from colabdesign.af.prep import prep_pdb
from colabdesign.shared.prep import prep_pos
from colabdesign.shared.utils import Key, copy_dict

from .modules import LigandRunModel, get_nearest_ligand_context

aa_order = residue_constants.restype_order
order_aa = {b: a for a, b in aa_order.items()}


def _aa_convert(x, rev=False):
  mpnn_alphabet = "ACDEFGHIKLMNPQRSTVWYX"
  af_alphabet = "ARNDCQEGHILKMFPSTWYVX"
  if x is None:
    return x
  if rev:
    return x[..., tuple(mpnn_alphabet.index(k) for k in af_alphabet)]
  x = jax.nn.one_hot(x, 21) if jnp.issubdtype(x.dtype, jnp.integer) else x
  if x.shape[-1] == 20:
    x = jnp.pad(x, [[0, 0], [0, 1]])
  return x[..., tuple(af_alphabet.index(k) for k in mpnn_alphabet)]


def _prepare_ligand_context_inputs(X, mask, Y, Y_t, Y_m, cutoff_for_score, atom_context_num, use_atom_context):
  N = X[:, 0, :]
  CA = X[:, 1, :]
  C = X[:, 2, :]
  b = CA - N
  c = C - CA
  a = jnp.cross(b, c, axis=-1)
  CB = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + CA
  Y_ctx, Y_t_ctx, Y_m_ctx, D_XY = get_nearest_ligand_context(CB, mask, Y, Y_t, Y_m, atom_context_num)
  mask_XY = (D_XY < cutoff_for_score).astype(jnp.float32) * mask * Y_m_ctx[:, 0]
  if not use_atom_context:
    Y_m_ctx = 0.0 * Y_m_ctx
  return Y_ctx, Y_t_ctx, Y_m_ctx, mask_XY


def _chain_mask_from_inputs(I):
  if "chain_mask" in I:
    return I["chain_mask"]
  if "fix_pos" in I:
    chain_mask = jnp.ones(I["mask"].shape[0], dtype=jnp.float32)
    chain_mask = chain_mask.at[I["fix_pos"]].set(0.0)
    return chain_mask
  return jnp.zeros(I["mask"].shape[0], dtype=jnp.float32)


class mk_ligand_mpnn_model:
  def __init__(self, model_name="v_32_010", backbone_noise=0.0, dropout=0.0, seed=None, verbose=False):
    from .weights import __file__ as ligand_path

    path = os.path.join(os.path.dirname(ligand_path), f"{model_name}.pkl")
    checkpoint = joblib.load(path)
    config = {
      "num_letters": 21,
      "node_features": 128,
      "edge_features": 128,
      "hidden_dim": 128,
      "num_encoder_layers": 3,
      "num_decoder_layers": 3,
      "augment_eps": backbone_noise,
      "k_neighbors": checkpoint["num_edges"],
      "atom_context_num": checkpoint["atom_context_num"],
      "dropout": dropout,
      "ligand_mpnn_use_side_chain_context": True,
    }
    self._model = LigandRunModel(config)
    self._model.params = jax.tree_map(np.array, checkpoint["model_state_dict"])
    self.atom_context_num = int(checkpoint["atom_context_num"])
    self._setup()
    self.set_seed(seed)

    self._num = 1
    self._inputs = {}
    self._tied_lengths = False

  def prep_inputs(self, pdb_filename=None, chain=None, homooligomer=False,
                  ignore_missing=True, fix_pos=None, inverse=False,
                  rm_aa=None, verbose=False, Y=None, Y_t=None, Y_m=None, **kwargs):
    pdb = prep_pdb(pdb_filename, chain, ignore_missing=ignore_missing)
    atom_idx = tuple(residue_constants.atom_order[k] for k in ["N", "CA", "C", "O"])
    chain_idx = np.concatenate([[n] * l for n, l in enumerate(pdb["lengths"])])
    self._lengths = pdb["lengths"]
    L = sum(self._lengths)

    self._inputs = {
      "X": pdb["batch"]["all_atom_positions"][:, atom_idx],
      "mask": pdb["batch"]["all_atom_mask"][:, 1],
      "S": pdb["batch"]["aatype"],
      "residue_idx": pdb["residue_index"],
      "chain_idx": chain_idx,
      "lengths": np.array(self._lengths),
      "bias": np.zeros((L, 20)),
    }
    if Y is not None:
      self._inputs["Y"] = np.asarray(Y, dtype=np.float32)
      self._inputs["Y_t"] = np.asarray(Y_t, dtype=np.int32)
      self._inputs["Y_m"] = np.asarray(Y_m, dtype=np.float32)

    if rm_aa is not None:
      for aa in rm_aa.split(","):
        self._inputs["bias"][..., aa_order[aa]] -= 1e6

    if fix_pos is not None:
      p = prep_pos(fix_pos, **pdb["idx"])["pos"]
      if inverse:
        p = np.delete(np.arange(L), p)
      self._inputs["fix_pos"] = p
      self._inputs["bias"][p] = 1e7 * np.eye(21)[self._inputs["S"]][p, :20]

    if homooligomer:
      assert min(self._lengths) == max(self._lengths)
      self._tied_lengths = True
      self._len = self._lengths[0]
    else:
      self._tied_lengths = False
      self._len = sum(self._lengths)
    self.pdb = pdb

  def get_af_inputs(self, af, *, Y=None, Y_t=None, Y_m=None, chain_mask=None):
    self._lengths = af._lengths
    self._len = af._len
    self._inputs["residue_idx"] = af._inputs["residue_index"]
    self._inputs["chain_idx"] = af._inputs["asym_id"]
    self._inputs["lengths"] = np.array(self._lengths)
    L = sum(self._lengths)
    self._inputs["bias"] = np.zeros((L, 20))
    self._inputs["bias"][-af._len:] = af._inputs["bias"]
    if "offset" in af._inputs:
      self._inputs["offset"] = af._inputs["offset"]
    if "batch" in af._inputs:
      atom_idx = tuple(residue_constants.atom_order[k] for k in ["N", "CA", "C", "O"])
      batch = af._inputs["batch"]
      self._inputs["X"] = batch["all_atom_positions"][:, atom_idx]
      self._inputs["mask"] = batch["all_atom_mask"][:, 1]
      self._inputs["S"] = batch["aatype"]
      self._inputs["xyz_37"] = batch["all_atom_positions"]
      self._inputs["xyz_37_m"] = batch["all_atom_mask"]
    if af.protocol == "binder":
      p = np.arange(af._target_len)
      self._inputs["chain_mask"] = np.concatenate(
        [np.zeros(af._target_len, dtype=np.float32), np.ones(af._len, dtype=np.float32)]
      )
    else:
      p = af.opt.get("fix_pos", None)
    if p is not None:
      self._inputs["fix_pos"] = p
      self._inputs["bias"][p] = 1e7 * np.eye(21)[self._inputs["S"]][p, :20]
    if chain_mask is not None:
      self._inputs["chain_mask"] = chain_mask
    if Y is not None:
      self._inputs["Y"] = np.asarray(Y, dtype=np.float32)
      self._inputs["Y_t"] = np.asarray(Y_t, dtype=np.int32)
      self._inputs["Y_m"] = np.asarray(Y_m, dtype=np.float32)
    self._tied_lengths = bool(af._args["homooligomer"])

  def set_seed(self, seed=None):
    np.random.seed(seed=seed)
    self.key = Key(seed=seed).get

  def _get_seq(self, O):
    def split_seq(seq):
      if len(self._lengths) > 1:
        seq = "".join(np.insert(list(seq), np.cumsum(self._lengths[:-1]), "/"))
        if self._tied_lengths:
          seq = seq.split("/")[0]
      return seq

    seqs, S = [], O["S"].argmax(-1)
    if S.ndim == 1:
      S = [S]
    for s in S:
      seq = "".join([order_aa[a] for a in s])
      seqs.append(split_seq(seq))
    return {"seq": np.array(seqs)}

  def sample(self, num=1, batch=1, temperature=0.1, **kwargs):
    outs = [self.sample_parallel(batch=batch, temperature=temperature, **kwargs) for _ in range(num)]
    return jax.tree_map(lambda *x: np.concatenate(x, 0), *outs)

  def sample_parallel(self, batch=10, temperature=0.1, **kwargs):
    I = copy_dict(self._inputs)
    I.update(kwargs)
    key = I.pop("key", self.key())
    keys = jax.random.split(key, batch)
    O = self._sample_parallel(keys, I, temperature, self._tied_lengths)
    O = jax.tree_map(np.array, O)
    O.update(self._get_seq(O))
    return O

  def _setup(self):
    def _score(X, mask, residue_idx, chain_idx, key, Y, Y_t, Y_m,
               cutoff_for_score=8.0, use_atom_context=True,
               xyz_37=None, xyz_37_m=None, ligand_mpnn_use_side_chain_context=False, **kwargs):
      I = {
        "X": X,
        "mask": mask.astype(jnp.float32),
        "residue_idx": residue_idx,
        "chain_idx": chain_idx,
      }
      I.update(kwargs)

      if "decoding_order" not in I:
        key, sub_key = jax.random.split(key)
        randn = jax.random.uniform(sub_key, (I["X"].shape[0],))
        randn = jnp.where(I["mask"], randn, randn + 1)
        if "fix_pos" in I:
          randn = randn.at[I["fix_pos"]].add(-1)
        I["decoding_order"] = randn.argsort()

      for k in ["S", "bias"]:
        if k in I:
          I[k] = _aa_convert(I[k])

      Y_ctx, Y_t_ctx, Y_m_ctx, mask_XY = _prepare_ligand_context_inputs(
        X=I["X"],
        mask=I["mask"],
        Y=Y,
        Y_t=Y_t,
        Y_m=Y_m,
        cutoff_for_score=cutoff_for_score,
        atom_context_num=self.atom_context_num,
        use_atom_context=use_atom_context,
      )
      I.update({
        "Y": Y_ctx,
        "Y_t": Y_t_ctx,
        "Y_m": Y_m_ctx,
        "mask_XY": mask_XY,
        "chain_mask": _chain_mask_from_inputs(I),
      })
      if xyz_37 is not None and xyz_37_m is not None:
        I["xyz_37"] = xyz_37
        I["xyz_37_m"] = xyz_37_m

      O = self._model.score(self._model.params, key, I)
      O["S"] = _aa_convert(O["S"], rev=True)
      O["logits"] = _aa_convert(O["logits"], rev=True)
      return O

    def _sample(X, mask, residue_idx, chain_idx, key, Y, Y_t, Y_m,
                temperature=0.1, tied_lengths=False,
                cutoff_for_score=8.0, use_atom_context=True,
                xyz_37=None, xyz_37_m=None, ligand_mpnn_use_side_chain_context=False, **kwargs):
      I = {
        "X": X,
        "mask": mask.astype(jnp.float32),
        "residue_idx": residue_idx,
        "chain_idx": chain_idx,
        "temperature": temperature,
      }
      I.update(kwargs)

      if "decoding_order" in I:
        if I["decoding_order"].ndim == 1:
          I["decoding_order"] = I["decoding_order"][:, None]
      else:
        key, sub_key = jax.random.split(key)
        randn = jax.random.uniform(sub_key, (I["X"].shape[0],))
        randn = jnp.where(I["mask"], randn, randn + 1)
        if "fix_pos" in I:
          randn = randn.at[I["fix_pos"]].add(-1)
        if tied_lengths:
          copies = I["lengths"].shape[0]
          decoding_order_tied = randn.reshape(copies, -1).mean(0).argsort()
          I["decoding_order"] = jnp.arange(I["X"].shape[0]).reshape(copies, -1).T[decoding_order_tied]
        else:
          I["decoding_order"] = randn.argsort()[:, None]

      for k in ["S", "bias"]:
        if k in I:
          I[k] = _aa_convert(I[k])

      Y_ctx, Y_t_ctx, Y_m_ctx, mask_XY = _prepare_ligand_context_inputs(
        X=I["X"],
        mask=I["mask"],
        Y=Y,
        Y_t=Y_t,
        Y_m=Y_m,
        cutoff_for_score=cutoff_for_score,
        atom_context_num=self.atom_context_num,
        use_atom_context=use_atom_context,
      )
      I.update({
        "Y": Y_ctx,
        "Y_t": Y_t_ctx,
        "Y_m": Y_m_ctx,
        "mask_XY": mask_XY,
        "chain_mask": _chain_mask_from_inputs(I),
      })
      if xyz_37 is not None and xyz_37_m is not None:
        I["xyz_37"] = xyz_37
        I["xyz_37_m"] = xyz_37_m

      O = self._model.sample(self._model.params, key, I)
      O["S"] = _aa_convert(O["S"], rev=True)
      O["logits"] = _aa_convert(O["logits"], rev=True)
      return O

    self._score = jax.jit(
      _score,
      static_argnames=["use_atom_context", "ligand_mpnn_use_side_chain_context"],
    )
    self._sample = jax.jit(
      _sample,
      static_argnames=["tied_lengths", "use_atom_context", "ligand_mpnn_use_side_chain_context"],
    )

    def _sample_parallel(key, inputs, temperature, tied_lengths=False):
      inputs.pop("temperature", None)
      inputs.pop("key", None)
      return _sample(**inputs, key=key, temperature=temperature, tied_lengths=tied_lengths)

    fn = jax.vmap(_sample_parallel, in_axes=[0, None, None, None])
    self._sample_parallel = jax.jit(fn, static_argnames=["tied_lengths"])

  def score(self, seq=None, **kwargs):
    I = copy_dict(self._inputs)
    if seq is not None:
      p = np.arange(I["S"].shape[0])
      if self._tied_lengths and len(seq) == self._lengths[0]:
        seq = seq * len(self._lengths)
      if "fix_pos" in I and len(seq) == (I["S"].shape[0] - I["fix_pos"].shape[0]):
        p = np.delete(p, I["fix_pos"])
      I["S"][p] = np.array([aa_order.get(aa, -1) for aa in seq])
    I.update(kwargs)
    key = I.pop("key", self.key())
    return jax.tree_map(np.array, self._score(**I, key=key))
