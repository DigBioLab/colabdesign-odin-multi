import jax
import jax.numpy as jnp
import numpy as np
import optax

from colabdesign.shared.utils import copy_dict, update_dict, softmax, Key
from colabdesign.shared.prep import rewire
from colabdesign.af.alphafold.common import residue_constants

aa_order = residue_constants.restype_order
order_aa = {b:a for a,b in aa_order.items()}

class design_model:
  def set_weights(self, *args, **kwargs):
    '''
    set weights
    -------------------
    note: model.restart() resets the weights to their defaults
    use model.set_weights(..., set_defaults=True) to avoid this
    -------------------
    model.set_weights(rmsd=1)
    '''
    if kwargs.pop("set_defaults", False):
      update_dict(self._opt["weights"], *args, **kwargs)
    update_dict(self.opt["weights"], *args, **kwargs)

  def set_seq(self, seq=None, mode=None, bias=None, rm_aa=None, set_state=True, **kwargs):
    '''
    Set sequence params and bias
    -----------------------------------
    -seq=str or seq=[str,str] or seq=array(shape=(L,20) or shape=(?,L,20))
    -mode=
      -"wildtype"/"wt" = initialize sequence with sequence saved from input PDB
      -"gumbel" = initial sequence with gumbel distribution
      -"soft_???" = apply softmax-activation to initialized sequence (e.g., "soft_gumbel")
    -bias=array(shape=(20,) or shape=(L,20)) - bias the sequence
    -rm_aa="C,W" = specify which amino acids to remove (i.e., add a negative-infinity bias to these aa)
    -----------------------------------
    '''

    # Backward compatibility
    seq_init = kwargs.pop("seq_init", None)
    if seq_init is not None:
        modes = ["soft", "gumbel", "wildtype", "wt"]
        if isinstance(seq_init, str):
            seq_init = seq_init.split("_")
        if isinstance(seq_init, list) and seq_init[0] in modes:
            mode = seq_init
        else:
            seq = seq_init

    if mode is None:
        mode = []

    # Define the shape
    shape = (self._num, self._len, self._args.get("alphabet_size", 20))

    # Initialize sequence and bias using the helper method
    x, b = self._initialize_sequence(seq, mode, bias, rm_aa, shape, is_offtarget=False)

    # Handle wildtype sequence
    if ("wildtype" in mode or "wt" in mode) and hasattr(self, "_wt_aatype"):
        wt_seq = np.eye(shape[-1])[self._wt_aatype]
        wt_seq[self._wt_aatype == -1] = 0
        if "pos" in self.opt and self.opt["pos"].shape[0] == wt_seq.shape[0]:
            x = np.zeros(shape)
            x[:, self.opt["pos"], :] = wt_seq
        else:
            x = wt_seq

    # Set sequence and bias
    self._params["seq"] = x
    self._inputs["bias"] = b

  def set_seq_offtarget(self, seq=None, mode=None, bias=None, rm_aa=None, set_state=True, **kwargs):
    """
    Set sequence parameters and bias for the off-target.
    ---------------------------------------------------
    Arguments:
      seq: None, str, list of str, or numpy array
           If str or list of str, each element should be an amino acid sequence.
           If array, it can be either of shape (L, 20) or (N, L, 20),
           where L = length of the protein, N = number of sequences.
      mode: list or str
            Used to control how the sequence is initialized. E.g.:
              - "wildtype"/"wt":  initialize from wildtype sequence (if  available)
              - "gumbel":         add Gumbel noise
              - "soft_???":       apply softmax to the initialized sequence
            You can combine modes, e.g., ["soft", "gumbel"].
      bias: None or array
            Additive bias of shape (20,) or (L, 20). If provided, it’s broadcast to match.
      rm_aa: str
             A comma-separated string of amino acids to remove (e.g., "C,W"),
             which sets a large negative bias for those positions.
      set_state: bool (currently unused, included for API consistency)
      kwargs: Any additional keyword arguments are passed down internally.

    Example usage:
      model.set_seq_offtarget(seq="ACDE", mode=["soft", "gumbel"], rm_aa="C,W")
    """
    # # Ensure the off-target length matches main sequence length
    # self._offtarget_len = self._len

    # If mode is None, treat as empty list
    if mode is None:
      mode = []

    # Define the shape: (N, L, 20) by default
    shape = (self._num, self._len, self._args.get("alphabet_size", 20))

    # Use our internal sequence initialization helper
    x, b = self._initialize_sequence(seq=seq,
                                     mode=mode,
                                     bias=bias,
                                     rm_aa=rm_aa,
                                     shape=shape,
                                     is_offtarget=True)

    # If requested, and if you store a "wildtype" off-target,
    # you could handle that here (similar to set_seq).
    # For example:
    if ("wildtype" in mode or "wt" in mode) and hasattr(self, "_wt_aatype"):
      wt_seq = np.eye(shape[-1])[self._wt_aatype]
      wt_seq[self._wt_aatype == -1] = 0
      if "pos" in self.opt and self.opt["pos"].shape[0] == wt_seq.shape[0]:
        x = np.zeros(shape)
        x[:, self.opt["pos"], :] = wt_seq
      else:
        x = wt_seq

    # Initialize off-target dicts if necessary
    if not hasattr(self, '_offtarget_params'):
      self._offtarget_params = {}
    if not hasattr(self, '_offtarget_inputs'):
      self._offtarget_inputs = {}

    # Set the off-target sequence and bias
    self._offtarget_params["seq"] = x
    self._offtarget_inputs["bias"] = b

  def _norm_seq_grad(self):
    g = self.aux["grad"]["seq"]
    eff_L = (np.square(g).sum(-1,keepdims=True) > 0).sum(-2,keepdims=True)
    gn = np.linalg.norm(g,axis=(-1,-2),keepdims=True)
    self.aux["grad"]["seq"] = g * np.sqrt(eff_L) / (gn + 1e-7)  

  def set_optimizer(self, optimizer=None, learning_rate=None, norm_seq_grad=None, **kwargs):
    '''
    set/reset optimizer
    ----------------------------------
    supported optimizers include: [adabelief, adafactor, adagrad, adam, adamw, 
    fromage, lamb, lars, noisy_sgd, dpsgd, radam, rmsprop, sgd, sm3, yogi]
    '''
    optimizers = {'adabelief':optax.adabelief,'adafactor':optax.adafactor,
                  'adagrad':optax.adagrad,'adam':optax.adam,
                  'adamw':optax.adamw,'fromage':optax.fromage,
                  'lamb':optax.lamb,'lars':optax.lars,
                  'noisy_sgd':optax.noisy_sgd,'dpsgd':optax.dpsgd,
                  'radam':optax.radam,'rmsprop':optax.rmsprop,
                  'sgd':optax.sgd,'sm3':optax.sm3,'yogi':optax.yogi}
    
    if optimizer is None: optimizer = self._args["optimizer"]
    if learning_rate is not None: self.opt["learning_rate"] = learning_rate
    if norm_seq_grad is not None: self.opt["norm_seq_grad"] = norm_seq_grad

    o = optimizers[optimizer](1.0, **kwargs)
    self._state = o.init(self._params)

    def update_grad(state, grad, params):
      updates, state = o.update(grad, state, params)
      grad = jax.tree_map(lambda x:-x, updates)
      return state, grad
    
    self._optimizer = jax.jit(update_grad)

  def set_seed(self, seed=None):
    np.random.seed(seed=seed)
    self.key = Key(seed=seed).get
    
  def get_seq(self, get_best=True):
    '''
    get sequences as strings
    - set get_best=False, to get the last sampled sequence
    '''
    aux = self._tmp["best"]["aux"] if (get_best and "aux" in self._tmp["best"]) else self.aux
    x = aux["seq"]["hard"].argmax(-1)
    return ["".join([order_aa[a] for a in s]) for s in x]
    
  def get_seqs(self, get_best=True):
    return self.get_seq(get_best)

  def rewire(self, order=None, offset=0, loops=0):
    '''
    helper function for "partial" protocol
    -----------------------------------------
    -order=[0,1,2] - change order of specified segments
    -offset=0 - specify start position of the first segment
    -loops=[3,2] - specified loop lengths between segments
    -----------------------------------------
    '''
    self.opt["pos"] = rewire(length=self._pos_info["length"], order=order,
                             offset=offset, loops=loops)

    # make default
    if hasattr(self,"_opt"): self._opt["pos"] = self.opt["pos"]

  def set_offtarget_weights(self, *args, **kwargs):
      '''
      Set weights for the off-target
      '''
      if not hasattr(self, '_offtarget_opt'):
          self._offtarget_opt = copy.deepcopy(self.opt)
      if kwargs.pop("set_defaults", False):
          update_dict(self._offtarget_opt["weights"], *args, **kwargs)
      update_dict(self._offtarget_opt["weights"], *args, **kwargs)

  def set_offtarget_seq(self, seq=None, mode=None, bias=None, rm_aa=None, set_state=True, **kwargs):
    '''
    Set sequence parameters and bias for the off-target
    '''
    # # Synchronize off-target length with main sequence length
    # self._offtarget_len = self._len  # Ensure both use the same length

    if mode is None:
      mode = []
      
    # Define the shape to match the main sequence
    shape = (self._num, self._len, self._args.get("alphabet_size", 20))

    # Initialize sequence and bias using the helper method
    x, b = self._initialize_sequence(seq, mode, bias, rm_aa, shape, is_offtarget=True)

    # Handle wildtype sequence for off-target (if applicable)
    if ("wildtype" in mode or "wt" in mode) and hasattr(self, "_wt_aatype"):
        wt_seq = np.eye(shape[-1])[self._wt_aatype]
        wt_seq[self._wt_aatype == -1] = 0
        if "pos" in self.opt and self.opt["pos"].shape[0] == wt_seq.shape[0]:
            x = np.zeros(shape)
            x[:, self.opt["pos"], :] = wt_seq
        else:
            x = wt_seq

    # Set off-target params and inputs
    if not hasattr(self, '_offtarget_params'):
        self._offtarget_params = {}
    if not hasattr(self, '_offtarget_inputs'):
        self._offtarget_inputs = {}
    self._offtarget_params["seq"] = x
    self._offtarget_inputs["bias"] = b

  def _norm_offtarget_seq_grad(self):
      '''
      Normalize gradient for the off-target sequence
      '''
      g = self.aux["grad"]["offtarget_seq"]
      eff_L = (np.square(g).sum(-1, keepdims=True) > 0).sum(-2, keepdims=True)
      gn = np.linalg.norm(g, axis=(-1, -2), keepdims=True)
      self.aux["grad"]["offtarget_seq"] = g * np.sqrt(eff_L) / (gn + 1e-7)

  def set_offtarget_optimizer(self, optimizer=None, learning_rate=None, norm_seq_grad=None, **kwargs):
      '''
      Set/reset optimizer for the off-target
      '''
      optimizers = {
          'adabelief': optax.adabelief, 'adafactor': optax.adafactor,
          'adagrad': optax.adagrad, 'adam': optax.adam,
          'adamw': optax.adamw, 'fromage': optax.fromage,
          'lamb': optax.lamb, 'lars': optax.lars,
          'noisy_sgd': optax.noisy_sgd, 'dpsgd': optax.dpsgd,
          'radam': optax.radam, 'rmsprop': optax.rmsprop,
          'sgd': optax.sgd, 'sm3': optax.sm3, 'yogi': optax.yogi
      }

      if optimizer is None:
          optimizer = self._args["optimizer"]
      if learning_rate is not None:
          self._offtarget_opt["learning_rate"] = learning_rate
      if norm_seq_grad is not None:
          self._offtarget_opt["norm_seq_grad"] = norm_seq_grad

      o = optimizers[optimizer](1.0, **kwargs)
      if not hasattr(self, '_offtarget_state'):
          self._offtarget_state = o.init(self._offtarget_params)
      else:
          self._offtarget_state = o.init(self._offtarget_params)

      def update_grad(state, grad, params):
          updates, state = o.update(grad, state, params)
          grad = jax.tree_map(lambda x: -x, updates)
          return state, grad

      self._offtarget_optimizer = jax.jit(update_grad)

  def get_offtarget_seq(self, get_best=True):
      '''
      Get sequences as strings for the off-target
      '''
      aux = self._tmp["best"]["aux_offtarget"] if (get_best and "aux_offtarget" in self._tmp["best"]) else self.aux["aux_offtarget"]
      x = aux["seq"]["hard"].argmax(-1)
      return ["".join([order_aa[a] for a in s]) for s in x]

  def get_offtarget_seqs(self, get_best=True):
      return self.get_offtarget_seq(get_best)
  
  def _initialize_sequence(self, seq, mode, bias, rm_aa, shape, is_offtarget=False):
    '''
    Internal helper to initialize sequence and bias.
    '''
    # Initialize bias
    if bias is None:
        b = np.zeros(shape[1:])
    else:
        b = np.array(np.broadcast_to(bias, shape[1:]))

    # Disable certain amino acids
    if rm_aa is not None:
        for aa in rm_aa.split(","):
            b[..., aa_order[aa]] -= 1e6

    # Initialize sequence
    if seq is None:
        if not is_offtarget and hasattr(self, "key"):
            x = 0.01 * np.random.normal(size=shape)
        else:
            x = np.zeros(shape)
    else:
        if isinstance(seq, str):
            seq = [seq]
        if isinstance(seq, list):
            if isinstance(seq[0], str):
                aa_dict = copy_dict(aa_order)
                if shape[-1] > 21:
                    aa_dict["-"] = 21  # Add gap character
                seq = np.asarray([[aa_dict.get(aa, -1) for aa in s] for s in seq])
            else:
                seq = np.asarray(seq)
        else:
            seq = np.asarray(seq)

        if np.issubdtype(seq.dtype, np.integer):
            seq_ = np.eye(shape[-1])[seq]
            seq_[seq == -1] = 0
            seq = seq_

        if seq.ndim == 2:
            x = np.pad(seq[None], [[0, shape[0] - 1], [0, 0], [0, 0]])
        elif shape[0] > seq.shape[0]:
            x = np.pad(seq, [[0, shape[0] - seq.shape[0]], [0, 0], [0, 0]])
        else:
            x = seq

    # Handle mode-specific initialization
    if "gumbel" in mode:
        y_gumbel = jax.random.gumbel(self.key(), shape)
        if "soft" in mode:
            y = softmax(x + b + y_gumbel)
        elif "alpha" in self.opt:
            y = x + y_gumbel / self.opt["alpha"]
        else:
            y = x + y_gumbel

        x = np.where(x.sum(-1, keepdims=True) == 1, x, y)

    return x, b
import jax
import jax.numpy as jnp

def soft_seq(x, bias, opt, key=None, num_seq=None, shuffle_first=True):
    """
    Convert raw logits x into various representations:
      - 'logits'
      - 'pssm' (softmax of raw logits)
      - 'soft' (temperature-scaled softmax)
      - 'hard' (categorical sample from 'soft', or argmax if not sampling)
      - 'pseudo' (blend of soft/hard/logits)

    If opt["gumbel"] is True, we'll sample from the softmax distribution
    instead of doing argmax. This yields a random discrete (one-hot) sequence.
    """
    seq = {"input": x}

    # --------------------------------------------------
    # 1) Shuffle MSA rows if x has shape [N, L, C]
    # --------------------------------------------------
    if x.ndim == 3 and x.shape[0] > 1 and key is not None:
        key, sub_key = jax.random.split(key)
        if num_seq is None or x.shape[0] == num_seq:
            # randomly pick which sequence is query
            if shuffle_first:
                n = jax.random.randint(sub_key, shape=(), minval=0, maxval=x.shape[0])
                seq["input"] = seq["input"].at[0].set(seq["input"][n]).at[n].set(seq["input"][0])
        else:
            # keep only 'num_seq' sequences, shuffle if desired
            n = jnp.arange(x.shape[0])
            if shuffle_first:
                n = jax.random.permutation(sub_key, n)
            else:
                # shuffle everything except row 0
                shuffled = jax.random.permutation(sub_key, n[1:])
                n = jnp.concatenate([jnp.array([0]), shuffled])
            seq["input"] = seq["input"][n[:num_seq]]

    # --------------------------------------------------
    # 2) Scale logits by alpha, add bias if any
    # --------------------------------------------------
    alpha = opt.get("alpha", 1.0)
    seq["logits"] = seq["input"] * alpha
    if bias is not None:
        seq["logits"] = seq["logits"] + bias

    # --------------------------------------------------
    # 3) pssm = softmax of raw logits (no temp)
    # --------------------------------------------------
    seq["pssm"] = jax.nn.softmax(seq["logits"], axis=-1)

    # --------------------------------------------------
    # 4) 'soft' = temperature-scaled softmax
    # --------------------------------------------------
    temp = opt.get("temp", 1.0)
    seq["soft"] = jax.nn.softmax(seq["logits"] / temp, axis=-1)

    # --------------------------------------------------
    # 5) 'hard' representation
    #    - If gumbel=True => sample from soft distribution
    #    - Else => standard argmax
    # --------------------------------------------------
    gumbel_flag = opt.get("gumbel", False)
    soft_factor = opt.get("soft", 0.0)
    hard_factor = opt.get("hard", 0.0)

    # Ensure gumbel_flag is a scalar boolean
    gumbel_flag = jnp.asarray(gumbel_flag, dtype=bool)

    # Define the condition: gumbel_flag and soft_factor > 0 and key is not None
    # All components need to be scalars
    condition = jnp.logical_and(jnp.logical_and(gumbel_flag, soft_factor > 0), key is not None)

    # Define function to sample using Gumbel-Softmax
    def sample_gumbel():
        # Sample categorical indices from the 'soft' distribution
        # Add a small epsilon to avoid log(0)
        logits = jnp.log(seq["soft"] + 1e-20)
        cat_sample = jax.random.categorical(key, logits, axis=-1)
        # Convert to one-hot
        x_hard = jax.nn.one_hot(cat_sample, seq["soft"].shape[-1])
        # Apply straight-through estimator
        return jax.lax.stop_gradient(x_hard - seq["soft"]) + seq["soft"]

    # Define function to use argmax
    def use_argmax():
        argmax_idx = jnp.argmax(seq["soft"], axis=-1)
        x_hard = jax.nn.one_hot(argmax_idx, seq["soft"].shape[-1])
        # Apply straight-through estimator
        return jax.lax.stop_gradient(x_hard - seq["soft"]) + seq["soft"]

    # Use jax.lax.cond to choose between sampling and argmax
    seq["hard"] = jax.lax.cond(
        condition,
        sample_gumbel,
        use_argmax
    )

    # --------------------------------------------------
    # 6) 'pseudo' = final blend
    # --------------------------------------------------
    # Blend 'soft' and 'logits'
    seq["pseudo"] = soft_factor * seq["soft"] + (1 - soft_factor) * seq["input"]
    # Blend 'hard' and the above 'pseudo'
    seq["pseudo"] = hard_factor * seq["hard"] + (1 - hard_factor) * seq["pseudo"]

    return seq
