import random
import jax
import numpy as np
import jax.numpy as jnp
import sys, gc

def clear_mem():
  # clear vram (GPU)
  backend = jax.lib.xla_bridge.get_backend()
  if hasattr(backend,'live_buffers'):
    for buf in backend.live_buffers():
      buf.delete()

  # TODO: clear ram (CPU)
  gc.collect()
  
def update_dict(D, *args, **kwargs):
  '''robust function for updating dictionary'''
  def set_dict(d, x, override=False):
    for k,v in x.items():
      if v is not None:
        if k in d:
          if isinstance(v, dict):
            set_dict(d[k], x[k], override=override)
          elif override or d[k] is None:
            d[k] = v
          elif isinstance(d[k],(np.ndarray,jnp.ndarray)):
            d[k] = np.asarray(v)
          elif isinstance(d[k], dict):
            d[k] = jax.tree_map(lambda x: type(x)(v), d[k])
          else:
            d[k] = type(d[k])(v)
        else:
          print(f"ERROR: '{k}' not found in {list(d.keys())}")  
  override = kwargs.pop("override", False)
  while len(args) > 0 and isinstance(args[0],str):
    D,args = D[args[0]],args[1:]
  for a in args:
    if isinstance(a, dict): set_dict(D, a, override=override)
  set_dict(D, kwargs, override=override)

def copy_dict(x):
  '''deepcopy dictionary'''
  return jax.tree_map(lambda y:y, x)

def to_float(x):
  '''Recursively convert numeric values to float, but leave strings and bytes unchanged.'''
  # Avoid iterating over strings and bytes
  if isinstance(x, (str, bytes)):
      return x
  # If the object can be converted to a list (for example, a NumPy array), do so.
  if hasattr(x, "tolist"):
      x = x.tolist()
  # If a dictionary, recursively process its items.
  if isinstance(x, dict):
      return {k: to_float(v) for k, v in x.items()}
  # If an iterable (but not a string), apply recursively.
  elif hasattr(x, "__iter__"):
      return [to_float(y) for y in x]
  # Otherwise, try to convert to a float.
  else:
      return float(x)
def dict_to_str(x, filt=None, keys=None, ok=None, print_str=None, f=2):
    '''convert dictionary to string for print out'''  
    if keys is None: keys = []
    if filt is None: filt = {}
    if print_str is None: print_str = ""
    if ok is None: ok = []

    # Gather keys not already listed
    for k in x.keys():
        if k not in keys:
            keys.append(k)

    # Separate keys into categories
    normal_keys = []
    target_keys = []
    offtarget_keys = []
    for k in keys:
        if k.startswith("t_"):
            target_keys.append(k)
        elif k.startswith("o_"):
            offtarget_keys.append(k)
        else:
            normal_keys.append(k)

    def format_value(k, v):
        if isinstance(v, float):
            if int(v) == v:
                return f" {k} {int(v)}"
            else:
                return f" {k} {v:.{f}f}"
        else:
            return f" {k} {v}"

    # Build output lines
    # Line 1: normal keys
    line1 = ""
    for k in normal_keys:
        if k in x and (filt.get(k, True) or k in ok):
            line1 += format_value(k, x[k])
    print_str += line1.strip() + "\n"

    # Line 2: target keys
    line2 = ""
    for k in target_keys:
        if k in x and (filt.get(k, True) or k in ok):
            line2 += format_value(k, x[k])
    print_str += line2.strip() + "\n"

    # Line 3: offtarget keys
    line3 = ""
    for k in offtarget_keys:
        if k in x and (filt.get(k, True) or k in ok):
            line3 += format_value(k, x[k])
    print_str += line3.strip() + "\n"

    return print_str


class Key():
    '''random key generator'''
    def __init__(self, key=None, seed=None):
      if key is None:
        self.seed = random.randint(0,2147483647) if seed is None else seed
        self.key = jax.random.PRNGKey(self.seed) 
      else:
        self.key = key

    def get(self, num=1):
        if num > 1:
            self.key, *sub_keys = jax.random.split(self.key, num=(num+1))
        else:
            self.key, sub_key = jax.random.split(self.key)
            return sub_key

def softmax(x, axis=-1):
  x = x - x.max(axis,keepdims=True)
  x = np.exp(x)
  return x / x.sum(axis,keepdims=True)

def categorical(p):
  return (p.cumsum(-1) >= np.random.uniform(size=p.shape[:-1])[..., None]).argmax(-1)

def to_list(xs):
  if not isinstance(xs,list): xs = [xs]
  return [x for x in xs if x is not None]

def copy_missing(a,b):
  for i,v in a.items():
    if i not in b:
      b[i] = v
    elif isinstance(v,dict):
      copy_missing(v,b[i])