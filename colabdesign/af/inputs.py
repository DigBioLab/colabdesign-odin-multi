import jax
import jax.numpy as jnp
import numpy as np

from colabdesign.shared.utils import copy_dict
from colabdesign.shared.model import soft_seq
from colabdesign.af.alphafold.common import residue_constants
from colabdesign.af.alphafold.model import model, config

############################################################################
# AF_INPUTS - functions for modifying inputs before passing to alphafold
############################################################################

class _af_inputs:

  def _get_seq(self, inputs, aux, key=None):
      params, opt = inputs["params"], inputs["opt"]
      '''Get sequence features for the target'''

      seq = soft_seq(params["seq"], inputs["bias"], opt, key, num_seq=self._num,
                      shuffle_first=self._args["shuffle_first"])
      seq = self._fix_pos(seq)
      aux.update({"seq": seq, "seq_pseudo": seq["pseudo"]})
      
      # Protocol-specific modifications to sequence features
      if self.protocol == "binder":
          # Concatenate target and binder sequence
          seq_target = jax.nn.one_hot(inputs["batch"]["aatype"][:self._target_len], self._args["alphabet_size"])
          seq_target = jnp.broadcast_to(seq_target, (self._num, *seq_target.shape))
          seq = jax.tree_map(lambda x: jnp.concatenate([seq_target, x], 1), seq)
      if self.protocol in ["fixbb", "hallucination", "partial"] and self._args["copies"] > 1:
          seq = jax.tree_map(lambda x: expand_copies(x, self._args["copies"], self._args["block_diag"]), seq)

      return seq

  def _get_seq_offtarget(self, inputs, aux, off_len, key=None):
    """
    - 'off_len' is now passed in as a plain Python integer from the caller
      (e.g. from _single_offtarget_forward).
    - We do NOT do any self._offtarget_lens[...] indexing here.
    """
    params, opt = inputs["params"], inputs["opt"]
    '''Get sequence features for the off-target'''
    # Build binder-length bias for soft_seq to match binder logits
    # Use the binder logits length directly to avoid tracer/int pitfalls
    bL = int(params["seq"].shape[1])
    binder_bias = None
    if "bias" in inputs and inputs["bias"] is not None:
        b = jnp.asarray(inputs["bias"])  # allow broadcast with JAX arrays
        # Exact binder-length bias provided
        if b.ndim == 2 and b.shape[0] == bL:
            binder_bias = b
        # Full-length bias provided: take last bL rows (binder at end)
        elif b.ndim == 2 and b.shape[-1] == self._args["alphabet_size"] and b.shape[0] >= bL and bL > 0:
            binder_bias = b[-bL:]
        # Single-vector bias: broadcast across binder length
        elif b.ndim == 1 and b.shape[0] == self._args["alphabet_size"] and bL > 0:
            binder_bias = jnp.broadcast_to(b, (bL, b.shape[0]))
    # Safe default: zero bias for binder region
    if binder_bias is None:
        binder_bias = jnp.zeros((bL, self._args["alphabet_size"]))

    seq = soft_seq(params["seq"], binder_bias, opt, key,
                   num_seq=self._num, shuffle_first=self._args["shuffle_first"])
    seq = self._fix_pos_offtarget(seq)
    aux.update({"seq": seq, "seq_pseudo": seq["pseudo"]})

    # If protocol == "binder", we do the slicing using off_len
    if self.protocol == "binder":
        # Derive static lengths from shapes to avoid dynamic slicing under JIT
        L_total = int(inputs["batch"]["aatype"].shape[0])
        otL = max(L_total - bL, 0)
        seq_offtarget = jax.nn.one_hot(
            inputs["batch"]["aatype"][:otL],
            self._args["alphabet_size"]
        )
        seq_offtarget = jnp.broadcast_to(
            seq_offtarget, (self._num, *seq_offtarget.shape)
        )
        # Concatenate off-target region with the rest
        seq = jax.tree_map(
            lambda x: jnp.concatenate([seq_offtarget, x], axis=1),
            seq
        )

    if self.protocol in ["fixbb", "hallucination", "partial"] and self._args["copies"] > 1:
        seq = jax.tree_map(
            lambda x: expand_copies(x, self._args["copies"], self._args["block_diag"]),
            seq
        )

    return seq

  def _fix_pos(self, seq, return_p=False):
      if "fix_pos" in self.opt:
          if "pos" in self.opt:
              seq_ref = jax.nn.one_hot(self._wt_aatype_sub, self._args["alphabet_size"])
              p = self.opt["pos"][self.opt["fix_pos"]]
              fix_seq = lambda x: x.at[..., p, :].set(seq_ref)
          else:
              seq_ref = jax.nn.one_hot(self._wt_aatype, self._args["alphabet_size"])
              p = self.opt["fix_pos"]
              fix_seq = lambda x: x.at[..., p, :].set(seq_ref[..., p, :])
          seq = jax.tree_map(fix_seq, seq)
          if return_p:
              return seq, p
      return seq

  def _fix_pos_offtarget(self, seq, return_p=False):
      # For off-targets, respect the inputs-scoped options if available
      opt = getattr(self, "opt", {})
      try:
          # If called via model path, inputs-scoped opt should be set on self._offtarget_opt
          if hasattr(self, "_offtarget_opt") and isinstance(self._offtarget_opt, dict):
              opt = self._offtarget_opt
      except Exception:
          pass
      if "fix_pos" in opt:
          if "pos" in opt:
              seq_ref = jax.nn.one_hot(self._wt_aatype_sub, self._args["alphabet_size"])
              p = opt["pos"][opt["fix_pos"]]
              fix_seq = lambda x: x.at[..., p, :].set(seq_ref)
          else:
              seq_ref = jax.nn.one_hot(self._wt_aatype, self._args["alphabet_size"])
              p = opt["fix_pos"]
              fix_seq = lambda x: x.at[..., p, :].set(seq_ref[..., p, :])
          seq = jax.tree_map(fix_seq, seq)
          if return_p:
              return seq, p
      return seq

  def _update_template(self, inputs, key):
      '''Dynamically update template features for the target'''
      if "batch" in inputs:
          batch, opt = inputs["batch"], inputs["opt"]

          # Enable templates
          inputs["template_mask"] = inputs["template_mask"].at[0].set(1)
          L = batch["aatype"].shape[0]
          
          # Decide which positions to remove sequence and/or sidechains
          rm = jnp.broadcast_to(inputs.get("rm_template", False), L)
          rm_seq = jnp.where(rm, True, jnp.broadcast_to(inputs.get("rm_template_seq", True), L))
          rm_sc = jnp.where(rm_seq, True, jnp.broadcast_to(inputs.get("rm_template_sc", True), L))
                            
          # Define template features
          template_feats = {"template_aatype": jnp.where(rm_seq, 21, batch["aatype"])}

          if "dgram" in batch:
              # Use dgram from batch if provided
              template_feats.update({"template_dgram": batch["dgram"]})
              nT, nL = inputs["template_aatype"].shape
              inputs["template_dgram"] = jnp.zeros((nT, nL, nL, 39))
              
          if "all_atom_positions" in batch:
              # Get pseudo-carbon-beta coordinates (carbon-alpha for glycine)
              cb, cb_mask = model.modules.pseudo_beta_fn(
                  jnp.where(rm_seq, 0, batch["aatype"]),
                  batch["all_atom_positions"],
                  batch["all_atom_mask"])
              template_feats.update({"template_pseudo_beta": cb,
                                      "template_pseudo_beta_mask": cb_mask,
                                      "template_all_atom_positions": batch["all_atom_positions"],
                                      "template_all_atom_mask": batch["all_atom_mask"]})

          # Inject template features
          if self.protocol == "partial":
              pos = opt["pos"]
              if self._args["repeat"] or self._args["homooligomer"]:
                  C, L = self._args["copies"], self._len
                  pos = (jnp.repeat(pos, C).reshape(-1, C) + jnp.arange(C) * L).T.flatten()

          for k, v in template_feats.items():
              if self.protocol == "partial":
                  if k in ["template_dgram"]:
                      inputs[k] = inputs[k].at[0, pos[:, None], pos[None, :]].set(v)
                  else:
                      inputs[k] = inputs[k].at[0, pos].set(v)
              else:
                  inputs[k] = inputs[k].at[0].set(v)
              
              # Remove sidechains (mask anything beyond CB)
              if k in ["template_all_atom_mask"]:
                  if self.protocol == "partial":
                      inputs[k] = inputs[k].at[:, pos, 5:].set(jnp.where(rm_sc[:, None], 0, inputs[k][:, pos, 5:]))
                      inputs[k] = inputs[k].at[:, pos].set(jnp.where(rm[:, None], 0, inputs[k][:, pos]))
                  else:
                      inputs[k] = inputs[k].at[..., 5:].set(jnp.where(rm_sc[:, None], 0, inputs[k][..., 5:]))
                      inputs[k] = jnp.where(rm[:, None], 0, inputs[k])

  def _update_template_offtarget(self, inputs, key):
      '''Dynamically update template features for the off-target'''
      if "batch" in inputs:
          batch, opt = inputs["batch"], inputs["opt"]

          # Enable templates
          inputs["template_mask"] = inputs["template_mask"].at[0].set(1)
          L = batch["aatype"].shape[0]
          
          # Decide which positions to remove sequence and/or sidechains
          rm = jnp.broadcast_to(inputs.get("rm_template", False), L)
          rm_seq = jnp.where(rm, True, jnp.broadcast_to(inputs.get("rm_template_seq", True), L))
          rm_sc = jnp.where(rm_seq, True, jnp.broadcast_to(inputs.get("rm_template_sc", True), L))
                            
          # Define template features
          template_feats = {"template_aatype": jnp.where(rm_seq, 21, batch["aatype"])}

          if "dgram" in batch:
              # Use dgram from batch if provided
              template_feats.update({"template_dgram": batch["dgram"]})
              nT, nL = inputs["template_aatype"].shape
              inputs["template_dgram"] = jnp.zeros((nT, nL, nL, 39))
              
          if "all_atom_positions" in batch:
              # Get pseudo-carbon-beta coordinates (carbon-alpha for glycine)
              cb, cb_mask = model.modules.pseudo_beta_fn(
                  jnp.where(rm_seq, 0, batch["aatype"]),
                  batch["all_atom_positions"],
                  batch["all_atom_mask"])
              template_feats.update({"template_pseudo_beta": cb,
                                      "template_pseudo_beta_mask": cb_mask,
                                      "template_all_atom_positions": batch["all_atom_positions"],
                                      "template_all_atom_mask": batch["all_atom_mask"]})

          # Inject template features
          # For off-target, adjust positions if necessary
          for k, v in template_feats.items():
              inputs[k] = inputs[k].at[0].set(v)
              
              # Remove sidechains (mask anything beyond CB)
              if k in ["template_all_atom_mask"]:
                  inputs[k] = inputs[k].at[..., 5:].set(jnp.where(rm_sc[:, None], 0, inputs[k][..., 5:]))
                  inputs[k] = jnp.where(rm[:, None], 0, inputs[k])

def update_seq(seq, inputs, seq_1hot=None, seq_pssm=None, mlm=None):
  '''update the sequence features'''
  
  if seq_1hot is None: seq_1hot = seq 
  if seq_pssm is None: seq_pssm = seq
  target_feat = seq_1hot[0,:,:20]

  seq_1hot = jnp.pad(seq_1hot,[[0,0],[0,0],[0,22-seq_1hot.shape[-1]]])
  seq_pssm = jnp.pad(seq_pssm,[[0,0],[0,0],[0,22-seq_pssm.shape[-1]]])
  msa_feat = jnp.zeros_like(inputs["msa_feat"]).at[...,0:22].set(seq_1hot).at[...,25:47].set(seq_pssm)

  # masked language modeling (randomly mask positions)
  if mlm is not None:    
    X = jax.nn.one_hot(22,23)
    X = jnp.zeros(msa_feat.shape[-1]).at[...,:23].set(X).at[...,25:48].set(X)
    msa_feat = jnp.where(mlm[...,None],X,msa_feat)
    
  inputs.update({"msa_feat":msa_feat, "target_feat":target_feat})

def update_aatype(aatype, inputs):
  r = residue_constants

  a = {"atom14_atom_exists":r.restype_atom14_mask,
       "atom37_atom_exists":r.restype_atom37_mask,
       "residx_atom14_to_atom37":r.restype_atom14_to_atom37,
       "residx_atom37_to_atom14":r.restype_atom37_to_atom14}
  mask = inputs["seq_mask"][:,None]

  inputs.update(jax.tree_map(lambda x:jnp.where(mask,jnp.asarray(x)[aatype],0),a))
  inputs["aatype"] = aatype

def expand_copies(x, copies, block_diag=True):
  '''
  given msa (N,L,20) expand to (1+N*copies,L*copies,22) if block_diag else (N,L*copies,22)
  '''
  if x.shape[-1] < 22:
    x = jnp.pad(x,[[0,0],[0,0],[0,22-x.shape[-1]]])
  x = jnp.tile(x,[1,copies,1])
  if copies > 1 and block_diag:
    L = x.shape[1]
    sub_L = L // copies
    y = x.reshape((-1,1,copies,sub_L,22))
    block_diag_mask = jnp.expand_dims(jnp.eye(copies),(0,3,4))
    seq = block_diag_mask * y
    gap_seq = (1-block_diag_mask) * jax.nn.one_hot(jnp.repeat(21,sub_L),22)  
    y = (seq + gap_seq).swapaxes(0,1).reshape(-1,L,22)
    return jnp.concatenate([x[:1],y],0)
  else:
    return x
