import copy
import hashlib
import os
import math
import jax
import jax.numpy as jnp
import numpy as np
import optax
from dataclasses import dataclass
from typing import Any, List, Optional, Iterable, Dict, Tuple
from abc import ABC, abstractmethod
from colabdesign.af.alphafold.common import residue_constants
from colabdesign.shared.utils import copy_dict, update_dict, Key, dict_to_str, to_float, softmax, categorical, to_list, copy_missing
from jax.flatten_util import ravel_pytree


# %%

# Pre‑projection stabilizer (centralize + Fisher + EMA gains)

Gradient = Any  # PyTree of arrays


class GradProjectionStrategy(ABC):
    """
    Base interface for gradient projection strategies.
    
    Provides configurable algorithms for projecting auxiliary gradients
    relative to a reference gradient in multi-objective optimization.
    
    Use: projected_aux = strategy(ref_grad, aux_grad)
    """
    name: str = "base"

    def __init__(self, match_norm: bool = False, eps: float = 1e-8) -> None:
        self.match_norm = match_norm
        self.eps = eps

    def __call__(self, ref: Gradient, aux: Gradient) -> Gradient:
        if self.match_norm:
            aux = self._match_norm(ref, aux)
        return self.project(ref, aux)

    @abstractmethod
    def project(self, ref: Gradient, aux: Gradient) -> Gradient:
        """Return projected version of aux w.r.t. ref."""
        ...

    @staticmethod
    def _ravel(x) -> Tuple[jnp.ndarray, callable]:
        flat, unravel = ravel_pytree(x)
        return flat, unravel

    def _match_norm(self, ref: Gradient, aux: Gradient) -> Gradient:
        rf, _ = self._ravel(ref)
        af, unravel = self._ravel(aux)
        n_r = jnp.linalg.norm(rf)
        n_a = jnp.linalg.norm(af)
        scale = jnp.where(n_a > self.eps, n_r / n_a, 1.0)
        return jax.tree_map(lambda t: t * scale, aux)



class PCGradProjection(GradProjectionStrategy):
    """
    If <ga, gm> < 0: ga <- ga - ( <ga,gm> / ||gm||^2 ) * gm
    else keep ga.
    """
    name = "pcgrad"
    def project(self, ref: Gradient, aux: Gradient) -> Gradient:
        gm, unravel_aux = self._ravel(ref)
        ga, _ = self._ravel(aux)
        inner = jnp.dot(ga, gm)
        denom = jnp.dot(gm, gm)
        coeff = jnp.where(denom > self.eps, inner / denom, 0.0)
        ga_new = jnp.where(inner < 0.0, ga - coeff * gm, ga)
        return unravel_aux(ga_new)

##################################################
# 6) OOP Task Representation & Gradient Combination
##################################################

@dataclass(frozen=True)
class Task:
    """One context gradient and its fixed context scale."""
    name: str
    grad: Any
    scaler: float
    role: str
    loss: float


class TaskBuilder:
    """Build context tasks from unscaled sequence gradients."""

    @staticmethod
    def _extract_seq_grad(gradient):
        if isinstance(gradient, dict) and "seq" in gradient:
            return gradient["seq"]
        return gradient

    def build_tasks(
        self,
        losses,
        gradients,
        roles,
        *,
        task_names=None,
        scale_magnitudes=None,
    ):
        count = len(losses)
        if not (
            len(gradients) == len(roles) == count
            and (task_names is None or len(task_names) == count)
        ):
            raise ValueError(
                "build_tasks: lengths of losses/gradients/roles/task_names must match"
            )
        if task_names is None:
            task_names = [f"task_{index}" for index in range(count)]
        if scale_magnitudes is None or len(scale_magnitudes) != count:
            raise ValueError(
                "build_tasks: scale_magnitudes must match the number of tasks"
            )

        return [
            Task(
                name=task_names[index],
                grad=self._extract_seq_grad(gradients[index]),
                scaler=float(scale_magnitudes[index]),
                role=roles[index],
                loss=float(losses[index]),
            )
            for index in range(count)
        ]


class GradientCombiner:
    """Sum target gradients and PCGrad-project signed off-target gradients."""

    def __init__(self, *, log_contributions: bool = True):
        self.offtarget_project_strategy = PCGradProjection(
            match_norm=False,
            eps=1e-8,
        )
        self.log_contributions = log_contributions

    @staticmethod
    def _extract_seq_grad(gradient):
        if isinstance(gradient, dict) and "seq" in gradient:
            return gradient["seq"]
        return gradient

    @staticmethod
    def _l2_norm(gradient) -> float:
        flat, _ = ravel_pytree(gradient)
        return float(jnp.linalg.norm(flat))

    @staticmethod
    def _cosine_sim(first, second) -> float:
        first_flat, _ = ravel_pytree(first)
        second_flat, _ = ravel_pytree(second)
        denominator = jnp.maximum(
            jnp.linalg.norm(first_flat) * jnp.linalg.norm(second_flat),
            1e-8,
        )
        return float(jnp.dot(first_flat, second_flat) / denominator)

    def combine_gradients(self, tasks: List[Task]) -> Any:
        targets = [task for task in tasks if task.role == "target"]
        others = [task for task in tasks if task.role != "target"]
        if not targets:
            raise ValueError("Need at least one target task")

        target_gradients = [
            self._extract_seq_grad(task.grad) * float(task.scaler)
            for task in targets
        ]
        consensus = jax.tree_map(lambda value: value * 0.0, target_gradients[0])
        for gradient in target_gradients:
            consensus = jax.tree_map(
                lambda total, value: total + value,
                consensus,
                gradient,
            )

        contributions = []
        for task, gradient in zip(targets, target_gradients):
            original_norm = self._l2_norm(self._extract_seq_grad(task.grad))
            gradient_norm = self._l2_norm(gradient)
            contributions.append(
                (task.name, task.scaler, original_norm, gradient_norm, gradient_norm)
            )

        combined = consensus
        for task in others:
            gradient = self._extract_seq_grad(task.grad) * float(task.scaler)
            original_norm = self._l2_norm(self._extract_seq_grad(task.grad))
            if task.role == "internal":
                projected = gradient
            else:
                cosine_before = self._cosine_sim(consensus, gradient)
                projected = self.offtarget_project_strategy(consensus, gradient)
                if self.log_contributions:
                    cosine_after = self._cosine_sim(consensus, projected)
                    print(
                        f"[PROJ] name={task.name} role={task.role} method=pcgrad "
                        f"cos_pre={cosine_before:+.3f} cos_post={cosine_after:+.3f} "
                        f"delta={cosine_after - cosine_before:+.3f}"
                    )

            projected_norm = self._l2_norm(projected)
            combined = jax.tree_map(
                lambda total, value: total + value,
                combined,
                projected,
            )
            contributions.append(
                (task.name, task.scaler, original_norm, projected_norm, projected_norm)
            )

        if self.log_contributions:
            total_norm = sum(item[-1] for item in contributions) or 1.0
            print(
                f"[GRAD] total={total_norm:.4f} | "
                f"targets={len(targets)} | others={len(others)}"
            )
            for name, scaler, original_norm, projected_norm, contribution in contributions:
                percentage = 100.0 * contribution / total_norm
                print(
                    f"[GRAD] task={name} scaler={scaler:.3f} "
                    f"orig={original_norm:.3f} proj={projected_norm:.3f} "
                    f"contrib={contribution:.3f} ({percentage:.1f}%)"
                )

        return combined


def compute_weighted_loss_from_tasks(tasks):
    """
    Compute weighted total loss from a list of tasks.
    NOTE: Role-based sign is NOT applied here. If you need repulsion,
    return a negative loss (or a signed gradient) from the task itself,
    or pass a negative weight explicitly.
    """
    total_loss = 0.0

    for t in tasks:
        total_loss += t.loss * t.scaler
    return total_loss


####################################################
# AF_DESIGN - design functions
####################################################
# \
# \_af_design
# |\
# | \_restart
#  \
#   \_design
#    \_step
#     \_run
#      \_recycle
#       \_single
#
####################################################

class _af_design:
    def restart(self, seed=None, opt=None, weights=None,
                seq=None, mode=None, keep_history=False, reset_opt=True, **kwargs):
        """
        Restart the optimization
        """
        if reset_opt and not keep_history:
            copy_missing(self.opt, self._opt)
            self.opt = copy_dict(self._opt)
            if hasattr(self, "aux"):
                del self.aux

        if not keep_history:
            # Initialize trajectory
            self._tmp = {
                "traj": {"seq": [], "xyz": [], "plddt": [], "pae": []},
                "log": [],
                "best": {}
            }

        # Update options/settings (if defined)
        self.set_opt(opt)
        self.set_weights(weights)

        # Initialize sequence
        self.set_seed(seed)
        self.set_seq(seq=seq, mode=mode, **kwargs)

        # Reset optimizer
        self._k = 0
        self.set_optimizer()
        
    def restart_offtarget(self, seed=None, opt=None, weights=None,
                          seq=None, mode=None, keep_history=False, reset_opt=True, **kwargs):
        """
        Restart the off-target optimization
        """
        if reset_opt and not keep_history:
            if hasattr(self, '_offtarget_opt'):
                self._offtarget_opt = copy.deepcopy(self._default_offtarget_opt)
            else:
                self._offtarget_opt = copy.deepcopy(self.opt)
                self._default_offtarget_opt = copy.deepcopy(self.opt)
            if hasattr(self, 'aux'):
                if 'aux_offtarget' in self.aux:
                    del self.aux['aux_offtarget']

        if not keep_history:
            self._offtarget_tmp = {
                "traj": {"seq": [], "xyz": [], "plddt": [], "pae": []},
                "log": [],
                "best": {}
            }

        if opt is not None:
            self.set_offtarget_opt(opt)
        if weights is not None:
            self.set_offtarget_weights(weights)

        # Initialize off-target sequence
        if seed is not None:
            self.set_seed(seed)
        self.set_offtarget_seq(seq=seq, mode=mode, **kwargs)

        self._offtarget_k = 0
        self.set_offtarget_optimizer()

    def set_offtarget_opt(self, opt):
        if not hasattr(self, '_offtarget_opt'):
            self._offtarget_opt = copy.deepcopy(self.opt)
        update_dict(self._offtarget_opt, opt)

    def set_offtarget_weights(self, *args, **kwargs):
        if not hasattr(self, '_offtarget_opt'):
            self._offtarget_opt = copy.deepcopy(self.opt)
        if kwargs.pop("set_defaults", False):
            update_dict(self._default_offtarget_opt["weights"], *args, **kwargs)
        update_dict(self._offtarget_opt["weights"], *args, **kwargs)

    def set_offtarget_optimizer(self, optimizer=None, learning_rate=None, 
                                norm_seq_grad=None, **kwargs):
        """
        Set/reset optimizer for the off-target
        """
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

        o = optimizers[optimizer](self._offtarget_opt["learning_rate"], **kwargs)
        self._offtarget_state = o.init(self._offtarget_params)

        def update_grad(state, grad, params):
            updates, state = o.update(grad, state, params)
            grad = jax.tree_map(lambda x: -x, updates)
            return state, grad

        self._offtarget_optimizer = jax.jit(update_grad)

    def _init_oop_components(self):
        self._task_builder = TaskBuilder()
        self._grad_combiner = GradientCombiner(log_contributions=True)
        print("[GRAD] fixed policy: target sum, off-target PCGrad, unit context scaling")

    def initialize_offtarget_sequence(self, mode=None, **kwargs):
        shape = (self._num, self._offtarget_len, self._args.get("alphabet_size", 20))
        if mode == "random":
            x = np.random.rand(*shape)
        elif mode == "zeros":
            x = np.zeros(shape)
        else:
            x = np.zeros(shape)
        return x

    def _get_model_nums(self, num_models=None, sample_models=None, models=None):
        if num_models is None:
            num_models = self.opt["num_models"]
        if sample_models is None:
            sample_models = self.opt["sample_models"]

        ns_name = self._model_names
        ns = list(range(len(ns_name)))
        if models is not None:
            models = models if isinstance(models, list) else [models]
            ns = [ns[n if isinstance(n, int) else ns_name.index(n)]
                  for n in models]

        m = min(num_models, len(ns))
        if sample_models and m != len(ns):
            model_nums = np.random.choice(ns, (m,), replace=False)
        else:
            model_nums = ns[:m]
        return model_nums

    def run(self, num_recycles=None, num_models=None, sample_models=None, models=None,
            backprop=True, callback=None, model_nums=None, return_aux=False):
        """
        run(...) executes one or more forward/backward passes of the model(s),
        accumulating target and off-target gradients via _single(), and storing results in self.aux.
        """
        print("Running ColabDesign...")
        self._init_oop_components()  # Initialize gradient combiner components

        for fn in self._callbacks["design"]["pre"]:
            fn(self)

        if model_nums is None:
            model_nums = self._get_model_nums(num_models, sample_models, models)
        assert len(model_nums) > 0, "ERROR: no model params defined"

        # Share only the sequence-sampling key across ensemble models/recycles
        # when stochastic sequence sampling is active. Model-side randomness
        # still comes from each _single() call.
        seq_key = copy.deepcopy(self.key()) if self.opt.get("gumbel", False) else None

        auxs = []
        for n in model_nums:
            p = self._model_params[n]
            auxs.append(
                self._recycle(
                    p,
                    num_recycles=num_recycles,
                    backprop=backprop,
                    seq_key=seq_key,
                )
            )

        # Aggregate results across models.
        auxs = jax.tree_map(lambda *x: np.stack(x), *auxs)
        self._assert_ensemble_binder_sequence_consistent(auxs, model_nums)

        def avg_or_first(x):
            # For floating-point arrays, take the mean along the first axis.
            if np.issubdtype(x.dtype, np.floating):
                return x.mean(0)
            else:
                return x[0]

        self.aux = jax.tree_map(avg_or_first, auxs)
        self.aux["atom_positions"] = auxs["atom_positions"][0]
        self.aux["all"] = auxs

        for fn in (self._callbacks["design"]["post"] + to_list(callback)):
            fn(self)

        # Create or merge with existing log structure
        if "log" not in self.aux or not self.aux["log"]:
            self.aux["log"] = {}

        # --- Merge hierarchical task structure from losses, if available ---
        if "losses" in self.aux and "tasks" in self.aux["losses"]:
            if "tasks" not in self.aux["log"]:
                self.aux["log"]["tasks"] = copy.deepcopy(self.aux["losses"]["tasks"])
            else:
                for task, data in self.aux["losses"]["tasks"].items():
                    if task not in self.aux["log"]["tasks"]:
                        self.aux["log"]["tasks"][task] = data
                    else:
                        self.aux["log"]["tasks"][task].update(data)
        # (If losses does not have a "tasks" key, we keep self.aux["log"]["tasks"] as built by _single)

        # Dynamically copy top-level keys from losses (all keys in aux["losses"] except "tasks")
        if "losses" in self.aux:
            for k, v in self.aux["losses"].items():
                if k != "tasks" and k not in self.aux["log"]:
                    self.aux["log"][k] = v

        for k in ["loss", "i_ptm", "ptm"]:
            self.aux["log"][k] = self.aux[k]

        # Update design flags in a generic way (avoid hardcoding extra keys).
        for k in ["hard", "soft", "temp"]:
            if k in self.opt:
                self.aux["log"][k] = self.opt[k]

        # (Optional) Compute sequence recovery if applicable.
        if self.protocol in ["fixbb", "partial"] or (self.protocol == "binder" and self._args.get("redesign", False)):
            if self.protocol == "partial":
                aatype = self.aux["aatype"][..., self.opt["pos"]]
            else:
                aatype = self.aux["seq"]["pseudo"].argmax(-1)
            mask = self._wt_aatype != -1
            true = self._wt_aatype[mask]
            pred = aatype[..., mask]
            self.aux["log"]["seqid"] = (true == pred).mean()

        # Convert all values in the log to float where appropriate.
        self.aux["log"] = to_float(self.aux["log"])

        # Add recycles and model information.
        self.aux["log"].update({
            "recycles": int(self.aux.get("num_recycles", 0)),
            "models": model_nums
        })

        # Include legacy keys (those starting with "t_" or "o_") without overwriting existing ones.
        for k in self.aux:
            if k.startswith("t_") or k.startswith("o_"):
                val = self.aux[k]
                if (isinstance(val, (int, float))) or (isinstance(val, np.ndarray) and val.ndim == 0):
                    self.aux["log"].setdefault(k, float(val))

        if return_aux:
            return self.aux




    def _initialize_prev_inputs(self):
        a = self._args

        # main target
        L = self._inputs["residue_index"].shape[0]
        use_existing_prev_main = ("prev" in self._inputs and not a["clear_prev"])
        if not use_existing_prev_main:
            prev_target = {
                'prev_msa_first_row': np.zeros([L, 256]),
                'prev_pair': np.zeros([L, L, 128]),
                'prev_pos': np.zeros([L, 37, 3]),
            }
            if a["use_dgram"]:
                prev_target["prev_dgram"] = np.zeros([L, L, 64])
            if a["use_initial_guess"] and "batch" in self._inputs:
                prev_target["prev_pos"] = self._inputs["batch"]["all_atom_positions"]
            elif a["use_initial_atom_pos"] and "batch" in self._inputs:
                self._inputs["initial_atom_pos"] = self._inputs["batch"]["all_atom_positions"]
        else:
            prev_target = self._inputs["prev"]

        self._inputs["prev"] = prev_target

        # off-targets (if we have multiple)
        if hasattr(self, "_offtargets") and self._offtargets:
            for offdict in self._offtargets:
                off_inputs = offdict["inputs"]
                try:
                    L_off = int(off_inputs["residue_index"].shape[0])
                except Exception:
                    L_off = int(off_inputs.get("aatype", np.zeros((0,))).shape[0])

                off_args = offdict.get("_args", {})
                clear_prev_off = off_args.get("clear_prev", a["clear_prev"])
                use_dgram_off = off_args.get("use_dgram", a["use_dgram"])
                use_initial_guess_off = off_args.get("use_initial_guess", a["use_initial_guess"])
                use_initial_atom_pos_off = off_args.get("use_initial_atom_pos", a["use_initial_atom_pos"])

                use_existing_prev_off = ("prev" in off_inputs and not clear_prev_off)
                if not use_existing_prev_off:
                    off_prev = {
                        'prev_msa_first_row': np.zeros([L_off, 256]),
                        'prev_pair': np.zeros([L_off, L_off, 128]),
                        'prev_pos': np.zeros([L_off, 37, 3]),
                    }
                    if use_dgram_off:
                        off_prev["prev_dgram"] = np.zeros([L_off, L_off, 64])
                    if use_initial_guess_off and "batch" in off_inputs:
                        off_prev["prev_pos"] = off_inputs["batch"]["all_atom_positions"]
                    elif use_initial_atom_pos_off and "batch" in off_inputs:
                        off_inputs["initial_atom_pos"] = off_inputs["batch"]["all_atom_positions"]
                    off_inputs["prev"] = off_prev
        elif hasattr(self, "_offtarget_inputs") and isinstance(self._offtarget_inputs, dict) and self._offtarget_inputs:
            use_existing_prev_off = ("prev" in self._offtarget_inputs and not a["clear_prev"])
            if not use_existing_prev_off:
                prev_offtarget = {
                    'prev_msa_first_row': np.zeros([L, 256]),
                    'prev_pair': np.zeros([L, L, 128]),
                    'prev_pos': np.zeros([L, 37, 3]),
                }
                if a["use_dgram"]:
                    prev_offtarget["prev_dgram"] = np.zeros([L, L, 64])
                if a["use_initial_guess"] and "batch" in self._offtarget_inputs:
                    prev_offtarget["prev_pos"] = self._offtarget_inputs["batch"]["all_atom_positions"]
                elif a["use_initial_atom_pos"] and "batch" in self._offtarget_inputs:
                    self._offtarget_inputs["initial_atom_pos"] = self._offtarget_inputs["batch"]["all_atom_positions"]
            else:
                prev_offtarget = self._offtarget_inputs.get("prev", {})

            self._offtarget_inputs["prev"] = prev_offtarget

    def _store_prev_from_aux(self, aux):
        a = self._args

        if aux.get("target_prev") is not None:
            self._inputs["prev"] = aux["target_prev"]

        if hasattr(self, "_offtargets") and self._offtargets:
            for i, offdict in enumerate(self._offtargets):
                keyname = f"offtarget_prev_{i}"
                if keyname in aux and aux[keyname] is not None:
                    offdict["inputs"]["prev"] = aux[keyname]
        elif hasattr(self, "_offtarget_inputs") and isinstance(self._offtarget_inputs, dict) and self._offtarget_inputs:
            if aux.get("offtarget_prev") is not None:
                self._offtarget_inputs["prev"] = aux["offtarget_prev"]

        if a["use_initial_atom_pos"]:
            if aux.get("target_prev") and "prev_pos" in aux["target_prev"]:
                self._inputs["initial_atom_pos"] = aux["target_prev"]["prev_pos"]
            if hasattr(self, "_offtargets") and self._offtargets:
                for i, offdict in enumerate(self._offtargets):
                    keyname = f"offtarget_prev_{i}"
                    if keyname in aux and aux[keyname] is not None:
                        if "prev_pos" in aux[keyname]:
                            offdict["inputs"]["initial_atom_pos"] = aux[keyname]["prev_pos"]
            elif hasattr(self, "_offtarget_inputs") and isinstance(self._offtarget_inputs, dict) and self._offtarget_inputs:
                if aux.get("offtarget_prev") and "prev_pos" in aux["offtarget_prev"]:
                    self._offtarget_inputs["initial_atom_pos"] = aux["offtarget_prev"]["prev_pos"]

    def _merge_dual_prev(self, target_prev, aux_prev):
        alpha = float(self._args.get("dual_prev_alpha", 0.5))
        merge_prev_pos = bool(self._args.get("dual_merge_prev_pos", True))
        bL = int(self._binder_len)

        merged_target_prev = dict(target_prev)
        merged_aux_prev = dict(aux_prev)

        merged_pair = (
            alpha * merged_target_prev["prev_pair"][-bL:, -bL:, :]
            + (1.0 - alpha) * merged_aux_prev["prev_pair"][-bL:, -bL:, :]
        )
        merged_msa = (
            alpha * merged_target_prev["prev_msa_first_row"][-bL:, :]
            + (1.0 - alpha) * merged_aux_prev["prev_msa_first_row"][-bL:, :]
        )

        merged_target_prev["prev_pair"] = merged_target_prev["prev_pair"].at[-bL:, -bL:, :].set(merged_pair)
        merged_aux_prev["prev_pair"] = merged_aux_prev["prev_pair"].at[-bL:, -bL:, :].set(merged_pair)

        merged_target_prev["prev_msa_first_row"] = merged_target_prev["prev_msa_first_row"].at[-bL:, :].set(merged_msa)
        merged_aux_prev["prev_msa_first_row"] = merged_aux_prev["prev_msa_first_row"].at[-bL:, :].set(merged_msa)

        if merge_prev_pos and "prev_pos" in merged_target_prev and "prev_pos" in merged_aux_prev:
            merged_pos = (
                alpha * merged_target_prev["prev_pos"][-bL:, :, :]
                + (1.0 - alpha) * merged_aux_prev["prev_pos"][-bL:, :, :]
            )
            merged_target_prev["prev_pos"] = merged_target_prev["prev_pos"].at[-bL:, :, :].set(merged_pos)
            merged_aux_prev["prev_pos"] = merged_aux_prev["prev_pos"].at[-bL:, :, :].set(merged_pos)

        merged_target_prev = jax.tree_map(jax.lax.stop_gradient, merged_target_prev)
        merged_aux_prev = jax.tree_map(jax.lax.stop_gradient, merged_aux_prev)
        return merged_target_prev, merged_aux_prev

    def _recycle_dual_prev_merge(self, model_params, num_recycles=None, backprop=True, seq_key=None):
        if num_recycles is None:
            num_recycles = self.opt["num_recycles"]
        if int(num_recycles) != 1:
            raise ValueError("enable_dual_prev_merge requires exactly 2 total passes (num_recycles == 1).")
        if not (hasattr(self, "_offtargets") and len(self._offtargets) == 1):
            raise ValueError("enable_dual_prev_merge requires exactly one auxiliary target/offtarget.")

        self._initialize_prev_inputs()

        aux0 = self._single(model_params, backprop=False, seq_key=seq_key)

        target_prev0 = aux0.get("target_prev")
        aux_prev0 = aux0.get("offtarget_prev_0")
        if target_prev0 is None or aux_prev0 is None:
            raise ValueError("Dual prev merge expected target_prev and offtarget_prev_0 after recycle 0.")

        merged_target_prev, merged_aux_prev = self._merge_dual_prev(target_prev0, aux_prev0)

        self._inputs["prev"] = merged_target_prev
        self._offtargets[0]["inputs"]["prev"] = merged_aux_prev

        merge_prev_pos = bool(self._args.get("dual_merge_prev_pos", True))

        if self._args.get("use_initial_atom_pos", False) and merge_prev_pos:
            if "prev_pos" in merged_target_prev:
                self._inputs["initial_atom_pos"] = np.asarray(merged_target_prev["prev_pos"], dtype=np.float32)
            if "prev_pos" in merged_aux_prev:
                self._offtargets[0]["inputs"]["initial_atom_pos"] = np.asarray(merged_aux_prev["prev_pos"], dtype=np.float32)

        aux1 = self._single(model_params, backprop=backprop, seq_key=seq_key)
        self._store_prev_from_aux(aux1)
        aux1["num_recycles"] = 1
        return aux1

    def _recycle(self, model_params, num_recycles=None, backprop=True, seq_key=None):
        a = self._args
        mode = a["recycle_mode"]
        if num_recycles is None:
            num_recycles = self.opt["num_recycles"]
        if a.get("enable_dual_prev_merge", False):
            return self._recycle_dual_prev_merge(
                model_params, num_recycles=num_recycles, backprop=backprop, seq_key=seq_key
            )

        self._initialize_prev_inputs()

        cycles = (num_recycles + 1)
        mask = [0] * cycles
        if mode == "sample":
            mask[np.random.randint(0, cycles)] = 1
        elif mode == "average":
            mask = [1/cycles] * cycles
        elif mode == "last":
            mask[-1] = 1
        elif mode == "first":
            mask[0] = 1

        grad = []
        for m in mask:
            pass_backprop = bool(backprop and m != 0)
            aux = self._single(
                model_params,
                backprop=pass_backprop,
                seq_key=seq_key,
            )
            if pass_backprop:
                grad.append(jax.tree_map(lambda x: x*m, aux["grad"]))

            self._store_prev_from_aux(aux)

        if backprop:
            aux["grad"] = jax.tree_map(lambda *x: np.stack(x).sum(0), *grad)
        aux["num_recycles"] = num_recycles
        return aux


    # %%
    # (removed) pre‑mix norm control helpers: not used

    def _single(self, model_params, backprop=True, seq_key=None):
        """
        Clean, in-class orchestrator:
        1) Run target & offtarget passes
        2) Build tasks
        3) Combine gradients
        4) Compute scalar total loss
        5) Assemble aux and return
        """
        # 1) forward passes — compute shared key once so target & offtargets
        #    receive the same PRNG key for their respective AF forward passes.
        shared_key = copy.deepcopy(self.key())
        tgt_loss, tgt_aux, tgt_grad = self._run_target(
            model_params, backprop=backprop, key=shared_key, seq_key=seq_key
        )

        off_results = self._run_offtargets(
            model_params, backprop=backprop, key=shared_key, seq_key=seq_key
        )
        self._debug_assert_target_offtarget_sequences(tgt_aux, off_results)
        # (removed) pre-mix unit/clip control: unused

        scale_magnitudes = [
            float(self._args.get("gradient_weight", 1.0)),
            *[
                float(self._offtargets[result["idx"]].get("gradient_weight", 1.0))
                for result in off_results
            ],
        ]
        tasks = self._build_tasks_inline(
            tgt_loss,
            tgt_grad,
            off_results,
            scale_magnitudes=scale_magnitudes,
        )
        if backprop:
            combined_grad = self._grad_combiner.combine_gradients(tasks)
        else:
            combined_grad = jax.tree_map(lambda x: np.zeros_like(x), tgt_grad)

        total_loss = compute_weighted_loss_from_tasks(tasks)

        # 6) assemble aux (unchanged)
        aux = self._assemble_aux(combined_grad, total_loss, tgt_aux, off_results)
        return aux

    def _sequence_debug_enabled(self):
        return os.environ.get("ODIN_MULTI_SEQ_DEBUG", "").lower() in {"1", "true", "yes", "on"}

    @staticmethod
    def _sequence_debug_digest(value):
        arr = np.asarray(value)
        if arr.ndim > 0 and arr.shape[-1] in (20, 21, 22):
            arr = arr.argmax(-1)
        arr = np.ascontiguousarray(arr)
        digest = hashlib.sha1(arr.view(np.uint8)).hexdigest()[:12]
        head = arr.reshape(-1)[:12].tolist()
        return f"shape={arr.shape} sha1={digest} head={head}"

    def _debug_assert_target_offtarget_sequences(self, tgt_aux, off_results):
        if not self._sequence_debug_enabled():
            return
        target_seq = tgt_aux.get("seq", {}).get("hard")
        if target_seq is None:
            return

        target = np.asarray(target_seq).argmax(-1)
        for off_result in off_results:
            off_seq = off_result["aux"].get("seq", {}).get("hard")
            if off_seq is None:
                continue
            off = np.asarray(off_seq).argmax(-1)
            np.testing.assert_array_equal(
                off,
                target,
                err_msg=(
                    "Target and off-target received different binder hard sequences "
                    f"for off-target index {off_result['idx']}"
                ),
            )

        print(
            "[SEQ_DEBUG] target/offtarget hard seq match "
            f"step={self._k} hard={self.opt.get('hard')} soft={self.opt.get('soft')} "
            f"temp={self.opt.get('temp')} gumbel={self.opt.get('gumbel', False)} "
            f"{self._sequence_debug_digest(target)}",
            flush=True,
        )

    def _assert_ensemble_binder_sequence_consistent(self, auxs, model_nums):
        if self.protocol != "binder" or len(model_nums) <= 1:
            return

        binder_len = int(self._params["seq"].shape[1])

        if "seq" in auxs and "hard" in auxs["seq"]:
            seq_hard = np.asarray(auxs["seq"]["hard"]).argmax(-1)
            np.testing.assert_array_equal(
                seq_hard,
                np.broadcast_to(seq_hard[0], seq_hard.shape),
                err_msg="Ensemble models produced different binder hard sequences",
            )
            if self._sequence_debug_enabled():
                print(
                    "[SEQ_DEBUG] ensemble seq hard consistent "
                    f"models={model_nums} {self._sequence_debug_digest(seq_hard)}",
                    flush=True,
                )

        if "aatype" not in auxs:
            return

        binder_aatype = np.asarray(auxs["aatype"])[:, -binder_len:]
        np.testing.assert_array_equal(
            binder_aatype,
            np.broadcast_to(binder_aatype[0], binder_aatype.shape),
            err_msg="Ensemble models received different binder sequences",
        )
        if self._sequence_debug_enabled():
            print(
                "[SEQ_DEBUG] ensemble binder aatype consistent "
                f"models={model_nums} {self._sequence_debug_digest(binder_aatype)}",
                flush=True,
            )


    # =========================
    # internal helpers (in-class)
    # =========================
    def _run_target(self, model_params, backprop: bool, key=None, seq_key=None):
        """Run main target forward/backward and return (loss, aux, grad)."""
        self._inputs["opt"] = self.opt
        if key is None:
            key = copy.deepcopy(self.key())
        flags = [self._params, model_params, self._inputs, key, seq_key]
        if backprop:
            (loss, aux), grad = self._model["grad_fn"](*flags)
        else:
            loss, aux = self._model["fn"](*flags)
            grad = jax.tree_map(np.zeros_like, self._params)

        aux["loss"] = loss
        aux["grad"] = grad
        return float(loss), aux, grad

    def _run_offtargets(self, model_params, backprop: bool, key=None, seq_key=None):
        """Run all offtarget passes; returns list of dicts with idx, loss, aux, grad.

        All configured off-targets participate with their constant context scale.
        """
        results = []
        if not (hasattr(self, "_offtargets") and self._offtargets):
            return results

        if key is None:
            key = copy.deepcopy(self.key())
        for i, offdict in enumerate(self._offtargets):
            backprop_off = backprop
            off_model = offdict["_offtarget_model"]
            off_inputs = offdict["inputs"]
            off_inputs["opt"] = offdict["opt_offtarget"]
            # Keep the off-target binder bias aligned with the main design bias.
            # _get_seq_offtarget() already knows how to consume binder-length,
            # full-complex, or vector bias forms, so avoid rewriting valid
            # binder-length bias into a zero full-length matrix.
            if "bias" in self._inputs:
                off_inputs["bias"] = self._inputs["bias"]
            # Do NOT overwrite off-target prev with target prev; let AF build defaults if missing
            # Validate any existing prev to match the off-target length; otherwise drop it
            try:
                L_off = int(off_inputs["residue_index"].shape[0])
            except Exception:
                L_off = int(off_inputs.get("aatype", np.zeros((0,))).shape[0])
            if "prev" in off_inputs:
                prev = off_inputs["prev"]
                bad_shape = False
                try:
                    if prev.get("prev_msa_first_row", np.zeros((0,))).shape[0] != L_off:
                        bad_shape = True
                    if prev.get("prev_pair", np.zeros((0,0,0))).shape[:2] != (L_off, L_off):
                        bad_shape = True
                    if prev.get("prev_pos", np.zeros((0,0,0))).shape[0] != L_off:
                        bad_shape = True
                    if "prev_dgram" in prev and prev["prev_dgram"].shape[:2] != (L_off, L_off):
                        bad_shape = True
                except Exception:
                    bad_shape = True
                if bad_shape:
                    off_inputs.pop("prev", None)

            # ensure animate() expectations
            if "batch" not in off_inputs:
                off_inputs["batch"] = {}
            if "all_atom_positions" not in off_inputs["batch"]:
                off_inputs["batch"]["all_atom_positions"] = np.zeros([L_off, 37, 3], dtype=np.float32)

            # expose for animate/log
            # Keep reference for animate/log without sharing across different off-targets
            self._offtarget_inputs = off_inputs

            flags = [self._params, model_params, off_inputs, key, seq_key]
            if backprop_off:
                (loss_off, aux_off), grad_off = off_model["grad_fn"](*flags)
            else:
                loss_off, aux_off = off_model["fn"](*flags)
                grad_off = jax.tree_map(np.zeros_like, self._params)

            aux_off["loss"] = loss_off
            aux_off["grad"] = grad_off

            results.append({
                "idx": i,
                "loss": float(loss_off),
                "aux": aux_off,
                "grad": grad_off
            })
        return results


    def _build_tasks_inline(self, tgt_loss, tgt_grad, off_results, scale_magnitudes=None):
        """
        Minimal task assembly from unscaled gradients.
        - Collect roles, losses, gradients, names for target + offtargets
        - Pass scale_magnitudes through for TaskBuilder to apply exactly once
        """
        # Target + offtargets (unscaled grads)
        task_roles = [self.role] + [self._offtargets[o["idx"]]["role"] 
                                    for o in off_results]
        task_losses = [tgt_loss] + [o["loss"] for o in off_results]
        task_gradients = [tgt_grad["seq"]] + [o["grad"]["seq"] 
                                             for o in off_results]
        task_names = [self.role] + [self._offtargets[o["idx"]]["role"] 
                                   for o in off_results]


        # Sanity: scale_magnitudes must align with the assembled tasks
        if (scale_magnitudes is None or 
            len(scale_magnitudes) != len(task_losses)):
            raise ValueError(
                f"scale_magnitudes length "
                f"({0 if scale_magnitudes is None else len(scale_magnitudes)}) "
                f"does not match number of tasks ({len(task_losses)})"
            )

        # (removed) pre‑projection stabilizer: unused

        # Delegate scaling to TaskBuilder (single application)
        tasks = self._task_builder.build_tasks(
            losses=task_losses,
            gradients=task_gradients,      # unscaled
            roles=task_roles,
            task_names=task_names,
            scale_magnitudes=scale_magnitudes
        )
        return tasks

    def _assemble_aux(self, combined_grad, total_loss, tgt_aux, off_results):
        """Assemble final aux (prev states, logs, metrics) exactly like before."""
        aux = {}
        # gradient under {"seq": ...}
        if isinstance(combined_grad, dict) and "seq" in combined_grad:
            final_grad = {"seq": combined_grad["seq"]}
        else:
            final_grad = {"seq": combined_grad}
        aux["grad"] = final_grad
        aux["loss"] = float(total_loss)

        # prev states
        aux["target_prev"] = tgt_aux.get("prev", None)
        if hasattr(self, "_offtargets") and self._offtargets:
            for i, o in enumerate(off_results):
                aux[f"offtarget_prev_{i}"] = o["aux"].get("prev", None)
        else:
            aux["offtarget_prev"] = None

        # key data from target
        for key_name in ["atom_positions", "plddt", "ptm", "i_ptm", "pae", "aatype", "residue_index", "atom_mask"]:
            if key_name in tgt_aux:
                val = tgt_aux[key_name]
                aux[key_name] = float(val) if key_name in ["ptm", "i_ptm"] else val

        # offtarget summaries
        packed = []
        for o in off_results:
            aux_o = o["aux"]
            sub = {
                "loss": aux_o["loss"],
                "prev": aux_o.get("prev", None),
                "atom_positions": aux_o.get("atom_positions", None),
                "plddt": aux_o.get("plddt", None),
                "ptm": aux_o.get("ptm", None),
                "i_ptm": aux_o.get("i_ptm", None),
                "pae": aux_o.get("pae", None),
            }
            packed.append(sub)
        aux["offtargets"] = packed

        # logs
        aux["log"] = {"tasks": {}, "loss": float(total_loss)}

        # main target log task
        aux["log"]["tasks"][self.name] = {"_role": self.role, "loss": float(tgt_aux["loss"])}
        if "losses" in tgt_aux:
            for metric, value in tgt_aux["losses"].items():
                aux["log"]["tasks"][self.name][metric] = float(value)
        for key_name in ["plddt", "ptm", "i_ptm"]:
            if key_name in tgt_aux:
                if key_name == "plddt":
                    aux["log"]["tasks"][self.name][key_name] = float(np.mean(tgt_aux[key_name]))
                else:
                    aux["log"]["tasks"][self.name][key_name] = float(tgt_aux[key_name])
        # Include scaler info if present
        if "_scaler_info" in tgt_aux:
            sc = tgt_aux["_scaler_info"]
            aux["log"]["tasks"][self.name]["scaler_combined"] = float(sc.get("combined", 0.0))
            aux["log"]["tasks"][self.name]["scaler_factor"] = float(sc.get("factor", 1.0))
            aux["log"]["tasks"][self.name]["scaler_scale_loss"] = float(sc.get("scale_loss", 0.0))
            pens = sc.get("penalties", {})
            for k, v in pens.items():
                aux["log"]["tasks"][self.name][f"sc_{k}_pen"] = float(v)

        # (removed) pre‑projection debug injection

        # offtarget log tasks
        if hasattr(self, "_offtargets") and self._offtargets:
            for o in off_results:
                task_name = self._offtargets[o["idx"]]["name"]
                role = self._offtargets[o["idx"]]["role"]
                aux["log"]["tasks"][task_name] = {"_role": role, "loss": float(o["loss"])}
                if "losses" in o["aux"]:
                    for metric, value in o["aux"]["losses"].items():
                        aux["log"]["tasks"][task_name][metric] = float(value)
                for key_name in ["plddt", "ptm", "i_ptm"]:
                    if key_name in o["aux"]:
                        if key_name == "plddt":
                            aux["log"]["tasks"][task_name][key_name] = float(np.mean(o["aux"][key_name]))
                        else:
                            aux["log"]["tasks"][task_name][key_name] = float(o["aux"][key_name])
                # Include scaler info if present
                if "_scaler_info" in o["aux"]:
                    sc = o["aux"]["_scaler_info"]
                    aux["log"]["tasks"][task_name]["scaler_combined"] = float(sc.get("combined", 0.0))
                    aux["log"]["tasks"][task_name]["scaler_factor"] = float(sc.get("factor", 1.0))
                    aux["log"]["tasks"][task_name]["scaler_scale_loss"] = float(sc.get("scale_loss", 0.0))
                    pens = sc.get("penalties", {})
                    for k, v in pens.items():
                        aux["log"]["tasks"][task_name][f"sc_{k}_pen"] = float(v)

                # (removed) pre‑projection debug injection for offtargets

        # copy top-level loss keys (except nested "tasks")
        if "losses" in tgt_aux:
            aux["losses"] = tgt_aux["losses"]
            for k, v in tgt_aux["losses"].items():
                if k != "tasks" and k not in aux["log"]:
                    aux["log"][k] = v

        # sequence info (unchanged)
        if "seq" in tgt_aux and isinstance(tgt_aux["seq"], dict):
            aux["seq"] = tgt_aux["seq"]
        else:
            raise ValueError("aux_target['seq'] is not a dictionary.")

        return aux

    def step(self, lr_scale=1.0, num_recycles=None,
            num_models=None, sample_models=None, models=None, backprop=True,
            callback=None, save_best=False, verbose=1, stage=None):
        """
        One step of gradient descent
        """
        self.run(num_recycles=num_recycles, num_models=num_models, 
                sample_models=sample_models, models=models, 
                backprop=backprop, callback=callback)

        if self.opt["norm_seq_grad"]:
            self._norm_seq_grad()

        self._state, self.aux["grad"] = self._optimizer(
            self._state, self.aux["grad"], self._params
        )

        lr = self.opt["learning_rate"] * lr_scale
        self._params = jax.tree_map(
            lambda x, g: x - lr * g, self._params, self.aux["grad"]
        )

        self._save_results(save_best=save_best, verbose=verbose, stage=stage)
        self._k += 1

    def _print_log(self, step_str=None, aux=None):
        """
        Print logs in multi-line format with column alignment per metric.
        line 1 => top-level global info
        subsequent lines => each task from the hierarchical structure
        """
        if aux is None:
            aux = self.aux
        log_dict = aux["log"]  # dictionary with hierarchical task structure

        # 1) Print some global keys on one line
        global_keys = ["models", "recycles", "hard", "soft", "temp", "loss", "plddt", "ptm", "i_ptm"]

        out = ""
        if step_str is not None:
            out += f"{step_str} "
        for k in global_keys:
            if k in log_dict:
                v = log_dict[k]
                if isinstance(v, float):
                    out += f"{k} {v:.2f} "
                else:
                    out += f"{k} {v} "
        print(out.strip())  # Print the "global" line

        # -----------------------
        # 2) Collect all tasks
        tasks_data = []  # list of (prefix, name, metrics_dict)

        # Newer hierarchical structure:
        if "tasks" in log_dict:
            for name, task_data in log_dict["tasks"].items():
                role = task_data.get("_role", "unknown")
                prefix = "t" if role == "target" else "o"
                # filter out the _role key
                metrics = {m_name: m_val
                        for m_name, m_val in task_data.items()
                        if m_name != "_role"}
                tasks_data.append((prefix, name, metrics))
        else:
            # Fallback to old format
            task_map = {}  # (prefix, idx, name) -> {metric_name -> metric_val}
            for k, v in log_dict.items():
                parts = k.split("_")
                if len(parts) < 4:
                    continue
                prefix = parts[0]
                if prefix not in ("t", "o"):
                    continue

                idx = parts[1]
                name = parts[2]
                metric = "_".join(parts[3:])
                task_map.setdefault((prefix, idx, name), {})[metric] = v

            sorted_keys = sorted(task_map.keys(), key=lambda x: (x[0], int(x[1])))
            for (prefix, idx, name) in sorted_keys:
                # e.g. prefix="t", idx="0", name="myTask"
                # but we'll combine idx+name for printing as "0_myTask"
                tasks_data.append((prefix, f"{idx}_{name}", task_map[(prefix, idx, name)]))

        if not tasks_data:
            return  # No tasks to print

        # -----------------------
        # 3) Figure out widths for columns
        #    We want columns for each metric. 
        #    Each metric has a "name" column and a "value" column. 
        #    We'll also align the "prefix_name" at the start.

        # 3a) collect all metric names
        all_metrics = set()
        for prefix, name, metrics in tasks_data:
            all_metrics.update(metrics.keys())

        # 3b) find max length of "prefix_name"
        max_prefix_len = 0
        for prefix, name, _ in tasks_data:
            prefix_name = prefix + "_" + name
            max_prefix_len = max(max_prefix_len, len(prefix_name))

        # 3c) for each metric, find:
        #     - the maximum length of that metric's name (usually just len(metric_name))
        #     - the maximum length of that metric's *value* (across tasks).
        metric_name_len = {}
        metric_val_len = {}

        for m in all_metrics:
            # Name length is straightforward
            metric_name_len[m] = len(m)

            # Value length depends on the biggest string needed for any value of that metric
            max_len = 0
            for prefix, name, metrics in tasks_data:
                if m in metrics:
                    val = metrics[m]
                    val_str = f"{val:.2f}" if isinstance(val, float) else str(val)
                    max_len = max(max_len, len(val_str))
            metric_val_len[m] = max_len

        # -----------------------
        # 4) Print each task in a row:
        #    prefix_name:  then for each metric in sorted order, 
        #    "metricName metricVal" (with spacing).

        # Sort metrics so columns are in a consistent order across tasks
        sorted_metrics = sorted(all_metrics)

        for prefix, name, metrics in tasks_data:
            prefix_name = prefix + "_" + name
            # This line begins with "prefix_name:<width>:", then we add metric columns
            line_str = f"{prefix_name:<{max_prefix_len}}:"

            # For each metric in sorted_metrics:
            for m in sorted_metrics:
                # metric name (left-justified by metric_name_len[m])
                m_str = f"{m:<{metric_name_len[m]}}"

                # metric value, or blank if not present
                if m in metrics:
                    val = metrics[m]
                    val_str = f"{val:.2f}" if isinstance(val, float) else str(val)
                else:
                    val_str = ""  # if this task doesn't have that metric
                # right-justify the value in metric_val_len[m] 
                v_str = f"{val_str:>{metric_val_len[m]}}"

                # 1 space between name/value, then 2 spaces before next pair
                line_str += f" {m_str} {v_str}  "

            print(line_str)


    @staticmethod
    def _append_trajectory_frame(trajectory, frame, max_frames):
        """Append one aligned, host-backed trajectory frame."""
        frame_keys = ("seq", "xyz", "plddt", "pae", "ptm", "i_ptm", "iteration", "stage")
        present_lengths = {
            key: len(trajectory[key])
            for key in frame_keys
            if key in trajectory and isinstance(trajectory[key], list)
        }
        current_length = max(present_lengths.values(), default=0)
        inconsistent = {
            key: length for key, length in present_lengths.items()
            if length != current_length
        }
        if inconsistent:
            raise ValueError(f"Unaligned trajectory before append: {inconsistent}")
        for key in frame_keys:
            trajectory.setdefault(key, [None] * current_length)

        try:
            limit = int(max_frames)
        except (TypeError, ValueError, OverflowError):
            limit = 0
        if limit > 0 and current_length >= limit:
            for key in frame_keys:
                trajectory[key].pop(0)

        for key in frame_keys:
            value = frame.get(key)
            if value is not None and key in {"seq", "xyz", "plddt", "pae"}:
                value = np.array(value, copy=True)
            elif value is not None and key in {"ptm", "i_ptm"}:
                value = float(np.asarray(value))
            elif value is not None and key == "iteration":
                value = int(value)
            elif value is not None and key == "stage":
                value = str(value)
            trajectory[key].append(value)

    def _save_results(self, aux=None, save_best=False,
                best_metric=None, metric_higher_better=False,
                verbose=True, stage=None):
        if aux is None:
            aux = self.aux

        # 1) append to main log
        self._tmp["log"].append(aux["log"])

        # 2) store main target trajectory
        if (self._k % self._args["traj_iter"]) == 0:
            target_traj = {
                "seq": aux["seq"]["pseudo"],
                "xyz": (aux["atom_positions"][:, 1, :] 
                        if aux["atom_positions"] is not None else None),
                "plddt": aux.get("plddt"),
                "pae": aux.get("pae"),
                "ptm": aux.get("ptm"),
                "i_ptm": aux.get("i_ptm"),
                "iteration": self._k,
                "stage": stage or "unknown",
            }
            self._append_trajectory_frame(
                self._tmp["traj"], target_traj, self._args["traj_max"]
            )

        # 3) store multi-offtarget data in each offdict["_tmp"]
        if hasattr(self, "_offtargets") and self._offtargets:
            # aux["offtargets"] is a list (same length as self._offtargets)
            # each element => {"loss", "atom_positions", "plddt", "pae", ...}
            for i, offdict in enumerate(self._offtargets):
                # Make sure _tmp/traj sub-keys exist on the offdict
                if "_tmp" not in offdict:
                    offdict["_tmp"] = {"traj": {"seq":[], "xyz":[], "plddt":[], "pae":[]}, "log":[]}
                if "traj" not in offdict["_tmp"]:
                    offdict["_tmp"]["traj"] = {"seq":[], "xyz":[], "plddt":[], "pae":[]}

                # only store frames every X steps
                if (self._k % self._args["traj_iter"]) == 0:
                    off_data = aux["offtargets"][i]
                    xyz = off_data.get("atom_positions", None)
                    if xyz is not None:
                        # usually shape [N,2,37,3] => take [:,1] for the backbone
                        xyz = xyz[:, 1, :]
                    offtarget_traj = {
                        "seq": aux["seq"]["pseudo"],
                        "xyz": xyz,
                        "plddt": off_data.get("plddt", None),
                        "pae": off_data.get("pae", None),
                        "ptm": off_data.get("ptm", None),
                        "i_ptm": off_data.get("i_ptm", None),
                        "iteration": self._k,
                        "stage": stage or "unknown",
                    }
                    self._append_trajectory_frame(
                        offdict["_tmp"]["traj"],
                        offtarget_traj,
                        self._args["traj_max"],
                    )
        else:
            # single off-target scenario or none
            pass
        # 4) track "best" design
        if save_best:
            if best_metric is None:
                best_metric = self._args["best_metric"]
            metric = float(aux["log"][best_metric])
            if best_metric in ["plddt", "ptm", "i_ptm", "seqid", "composite"] or metric_higher_better:
                metric = -metric
            if "metric" not in self._tmp["best"] or metric < self._tmp["best"]["metric"]:
                self._tmp["best"]["aux"] = copy_dict(aux)
                self._tmp["best"]["metric"] = metric
        
        print(f'{self._k}: {self._tmp["best"]["metric"]}')
              
        if "tasks" in aux["log"]:
            task_items = list(aux["log"]["tasks"].items())

            struct_names = [key for (key, _) in task_items]
            # transform "target"→"t", "offtarget"→"o"
            roles_short = []
            roles = []
            for (_, data) in task_items:
                role_str = data.get("_role", "target")  # fallback
                roles.append(role_str)
                roles_short.append("t" if role_str == "target" else "o")
            
            self._tmp["tasks_struct_names"] = struct_names
            self._tmp["tasks_roles"] = roles
            self._tmp["tasks_roles_short"] = roles_short
        # 5) print logs
        if verbose and ((self._k + 1) % verbose) == 0:
            self._print_log(f"{self._k+1}", aux=aux)



    def predict(self, seq=None, bias=None,
                num_models=None, num_recycles=None, models=None, sample_models=False,
                dropout=False, hard=True, soft=False, temp=1,
                return_aux=False, verbose=True, seed=None, **kwargs):
        """Predict structure for input sequence (if provided)."""
        def load_settings():
            if "save" in self._tmp:
                [self.opt, self._args, self._params,
                self._inputs, self._offtarget_inputs] = self._tmp.pop("save")

        def save_settings():
            load_settings()
            off_inputs = getattr(self, "_offtarget_inputs", {})
            self._tmp["save"] = [
                copy_dict(x) for x in [
                    self.opt, self._args, self._params,
                    self._inputs, off_inputs
                ]
            ]

        save_settings()
        if seed is not None:
            self.set_seed(seed)

        if seq is not None:
            self.set_seq(seq=seq, bias=bias)
            self.set_seq_offtarget(seq=seq, bias=bias)

        self.set_opt(hard=hard, soft=soft, temp=temp, dropout=dropout, pssm_hard=True)
        self.set_args(shuffle_first=False)
        if hasattr(self, '_offtarget_opt'):
            self._offtarget_opt['hard'] = hard
            self._offtarget_opt['soft'] = soft
            self._offtarget_opt['temp'] = temp
            self._offtarget_opt['dropout'] = dropout
            self._offtarget_opt['pssm_hard'] = True

        self.run(num_recycles=num_recycles, num_models=num_models,
                sample_models=sample_models, models=models,
                backprop=False, **kwargs)
        if verbose:
            self._print_log("predict")

        load_settings()
        if return_aux:
            return self.aux

    # ----------------------------------------------
    # Example design functions...
    # ----------------------------------------------
    def design(self, iters=100,
            soft=0.0, e_soft=None,
            temp=1.0, e_temp=None,
            hard=0.0, e_hard=None,
            step=1.0, e_step=None,
            dropout=True, opt=None, weights=None,
            num_recycles=None, ramp_recycles=False,
            num_models=None, sample_models=None, models=None,
            backprop=True, callback=None, save_best=False, verbose=1,
            off_target=False, gumbel=False, stage=None):
        """
        Example multi-iteration design logic.
        """
        self.set_opt(opt, dropout=dropout)
        self.opt["gumbel"] = gumbel
        self.opt["dropout"] = dropout

        # If multiple off-targets, set them likewise
        if hasattr(self, "_offtargets"):
            for offdict in self._offtargets:
                offdict["opt_offtarget"]["gumbel"] = gumbel
                offdict["opt_offtarget"]["dropout"] = dropout
        elif hasattr(self, '_offtarget_opt'):
            self._offtarget_opt["gumbel"] = gumbel
            self._offtarget_opt["dropout"] = dropout

        self.set_weights(weights)
        if weights is not None:
            if hasattr(self, "_offtargets"):
                for offdict in self._offtargets:
                    update_dict(offdict["opt_offtarget"]["weights"], weights)
            elif hasattr(self, '_offtarget_opt'):
                update_dict(self._offtarget_opt["weights"], weights)

        # Schedules
        m = {
            "soft": [soft, e_soft],
            "temp": [temp, e_temp],
            "hard": [hard, e_hard],
            "step": [step, e_step]
        }
        m = {k: [s, (s if e is None else e)] for k, (s, e) in m.items()}
        if ramp_recycles and num_recycles is None:
            num_recycles = self.opt.get("num_recycles", 0)
            m["num_recycles"] = [0, num_recycles]

        for i in range(iters):
            if off_target:
                update_keys = ["soft", "hard"]
            else:
                update_keys = list(m.keys())

            frac = (i + 1) / iters
            for k in update_keys:
                s, e = m[k]
                if k == "temp":
                    new_val = e + (s - e) * (1 - frac) ** 2
                    self.set_opt({k: new_val})
                    if hasattr(self, "_offtargets"):
                        for offdict in self._offtargets:
                            offdict["opt_offtarget"][k] = new_val
                    elif hasattr(self, '_offtarget_opt'):
                        self._offtarget_opt[k] = new_val
                elif k == "step":
                    new_val = s + (e - s) * frac
                    step_val = new_val
                elif k == "num_recycles":
                    new_val = round(s + (e - s) * frac)
                    num_recycles = new_val
                else:
                    new_val = s + (e - s) * frac
                    self.set_opt({k: new_val})
                    if hasattr(self, "_offtargets"):
                        for offdict in self._offtargets:
                            offdict["opt_offtarget"][k] = new_val
                    elif hasattr(self, '_offtarget_opt'):
                        self._offtarget_opt[k] = new_val

            # ensure off-target "soft"/"hard" are synced
            if hasattr(self, "_offtargets"):
                for offdict in self._offtargets:
                    offdict["opt_offtarget"]["soft"] = self.opt["soft"]
                    offdict["opt_offtarget"]["hard"] = self.opt["hard"]
            elif hasattr(self, '_offtarget_opt'):
                self._offtarget_opt["soft"] = self.opt["soft"]
                self._offtarget_opt["hard"] = self.opt["hard"]

            # LR scale
            step_val = step_val if "step" in m else 1.0
            lr_scale = step_val * (
                (1 - self.opt.get("soft", 0.0)) +
                self.opt.get("soft", 0.0) * self.opt.get("temp", 1.0)
            )

            # Step
            self.step(
                lr_scale=lr_scale,
                num_recycles=num_recycles if ramp_recycles else None,
                num_models=num_models,
                sample_models=sample_models,
                models=models,
                backprop=backprop,
                callback=callback,
                save_best=save_best,
                verbose=verbose,
                stage=stage,
            )

    def design_logits(self, iters=100, **kwargs):
        """Optimize logits."""
        kwargs.setdefault("stage", "logits")
        self.design(iters, **kwargs)

    def design_soft(self, iters=100, temp=1, **kwargs):
        """Optimize softmax(logits/temp)."""
        kwargs.setdefault("stage", "soft")
        self.design(iters, soft=1, temp=temp, **kwargs)

    def design_hard(self, iters=100, **kwargs):
        """Optimize argmax(logits)."""
        kwargs.setdefault("stage", "hard")
        self.design(iters, soft=1, hard=1, **kwargs)

    def design_gumbel(self, iters=100, temp=1, **kwargs):
        """Optimize with Gumbel-Softmax."""
        kwargs.setdefault("stage", "ste")
        self.design(iters, soft=1, hard=1, temp=temp, gumbel=True, **kwargs)
    # ---------------------------------------------------------------------------------
    # experimental
    #---------------------------------------------------------------------------------
    def design_3stage(self, soft_iters=300, temp_iters=100, hard_iters=10,
                    ramp_recycles=True, **kwargs):
        '''three stage design (logits→soft→hard)'''

        verbose = kwargs.get("verbose", 1)

        # stage 1: logits -> softmax(logits/1.0)
        if soft_iters > 0:
            if verbose:
                print("Stage 1: running (logits → soft)")
            self.design_logits(soft_iters, e_soft=1,
                            ramp_recycles=ramp_recycles, **kwargs)
            self._tmp["seq_logits"] = self.aux["seq"]["logits"]

        # stage 2: softmax(logits/1.0) -> softmax(logits/0.01)
        if temp_iters > 0:
            if verbose:
                print("Stage 2: running (soft → hard)")
            self.design_soft(temp_iters, e_temp=1e-2, **kwargs)

        # stage 3:
        if hard_iters > 0:
            if verbose:
                print("Stage 3: running (hard)")
            kwargs["dropout"] = False
            kwargs["save_best"] = True
            kwargs["num_models"] = len(self._model_names)
            self.design_hard(hard_iters, temp=1e-2, **kwargs)

    def _mutate(self, seq, plddt=None, logits=None, mutation_rate=1):
        '''mutate random position'''
        seq = np.array(seq)
        N, L = seq.shape

        # fix some positions
        i_prob = np.ones(L) if plddt is None else np.maximum(1-plddt, 0)
        i_prob[np.isnan(i_prob)] = 0
        if "fix_pos" in self.opt:
            if "pos" in self.opt:
                p = self.opt["pos"][self.opt["fix_pos"]]
                seq[..., p] = self._wt_aatype_sub
            else:
                p = self.opt["fix_pos"]
                seq[..., p] = self._wt_aatype[..., p]
            i_prob[p] = 0

        for m in range(mutation_rate):
            # sample position
            # https://www.biorxiv.org/content/10.1101/2021.08.24.457549v1
            i = np.random.choice(np.arange(L), p=i_prob/i_prob.sum())

            # sample amino acid
            logits = np.array(0 if logits is None else logits)
            if logits.ndim == 3:
                logits = logits[:, i]
            elif logits.ndim == 2:
                logits = logits[i]
            a_logits = logits - \
                np.eye(self._args["alphabet_size"])[seq[:, i]] * 1e8
            a = categorical(softmax(a_logits))

            # return mutant
            seq[:, i] = a

        return seq

    def design_semigreedy(self, iters=100, tries=10, dropout=False,
                        save_best=True, seq_logits=None, e_tries=None, **kwargs):
        '''semigreedy search'''
        if e_tries is None:
            e_tries = tries

        # get starting sequence
        if hasattr(self, "aux"):
            seq = self.aux["seq"]["logits"].argmax(-1)
        else:
            seq = (self._params["seq"] + self._inputs["bias"]).argmax(-1)

        # bias sampling towards the defined bias
        if seq_logits is None:
            seq_logits = 0

        model_flags = {k: kwargs.pop(k, None) for k in [
            "num_models", "sample_models", "models"]}
        verbose = kwargs.pop("verbose", 1)

        # get current plddt
        aux = self.predict(seq, return_aux=True,
                        verbose=False, **model_flags, **kwargs)
        plddt = self.aux["plddt"]
        plddt = plddt[self._target_len:] if self.protocol == "binder" else plddt[:self._len]

        # optimize!
        if verbose:
            print("Running semigreedy optimization...")

        for i in range(iters):
            buff = []
            model_nums = self._get_model_nums(**model_flags)
            num_tries = (tries+(e_tries-tries)*((i+1)/iters))
            for t in range(int(num_tries)):
                mut_seq = self._mutate(seq=seq, plddt=plddt,
                                    logits=seq_logits + self._inputs["bias"])
                aux = self.predict(seq=mut_seq, return_aux=True,
                                model_nums=model_nums, verbose=False, **kwargs)
                buff.append({"aux": aux, "seq": np.array(mut_seq)})

            # accept best
            losses = [x["aux"]["loss"] for x in buff]
            best = buff[np.argmin(losses)]
            self.aux, seq = best["aux"], jnp.array(best["seq"])
            self.set_seq(seq=seq, bias=self._inputs["bias"])
            # Guard bias copy: only copy if lengths match the current off-target
            try:
                if hasattr(self, "_offtarget_inputs") and isinstance(self._offtarget_inputs, dict):
                    b = self._inputs.get("bias", None)
                    if b is not None:
                        try:
                            L_off = int(self._offtarget_inputs.get("residue_index", np.zeros((0,))).shape[0])
                        except Exception:
                            L_off = int(self._offtarget_inputs.get("aatype", np.zeros((0,))).shape[0])
                        if isinstance(b, np.ndarray) and b.shape[0] == L_off:
                            self._offtarget_inputs['bias'] = b
            except Exception:
                pass
            self._save_results(save_best=save_best, verbose=verbose, stage="greedy")

            # update plddt
            plddt = best["aux"]["plddt"]
            plddt = plddt[self._target_len:] if self.protocol == "binder" else plddt[:self._len]
            self._k += 1

    def design_pssm_semigreedy(self, soft_iters=300, hard_iters=32, tries=10, e_tries=None,
                            ramp_recycles=True, ramp_models=True, **kwargs):

        verbose = kwargs.get("verbose", 1)

        # stage 1: logits -> softmax(logits)
        if soft_iters > 0:
            self.design_3stage(
                soft_iters, 0, 0, ramp_recycles=ramp_recycles, **kwargs)
            self._tmp["seq_logits"] = kwargs["seq_logits"] = self.aux["seq"]["logits"]

        # stage 2: semi_greedy
        if hard_iters > 0:
            kwargs["dropout"] = False
            if ramp_models:
                num_models = len(kwargs.get("models", self._model_names))
                iters = hard_iters
                for m in range(num_models):
                    if verbose and m > 0:
                        print(f'Increasing number of models to {m+1}.')

                    kwargs["num_models"] = m + 1
                    kwargs["save_best"] = (m + 1) == num_models
                    self.design_semigreedy(
                        iters, tries=tries, e_tries=e_tries, **kwargs)
                    if m < 2:
                        iters = iters // 2
            else:
                self.design_semigreedy(
                    hard_iters, tries=tries, e_tries=e_tries, **kwargs)

    # ---------------------------------------------------------------------------------
    # experimental optimizers (not extensively evaluated)
    # ---------------------------------------------------------------------------------

    def _design_mcmc(self, steps=1000, half_life=200, T_init=0.01, mutation_rate=1,
                    seq_logits=None, save_best=True, **kwargs):
        '''
        MCMC with simulated annealing
        ----------------------------------------
        steps = number for steps for the MCMC trajectory
        half_life = half-life for the temperature decay during simulated annealing
        T_init = starting temperature for simulated annealing. Temperature is decayed exponentially
        mutation_rate = number of mutations at each MCMC step
        '''

        # code borrowed from: github.com/bwicky/oligomer_hallucination

        # gather settings
        verbose = kwargs.pop("verbose", 1)
        model_flags = {k: kwargs.pop(k, None) for k in [
            "num_models", "sample_models", "models"]}

        # initialize
        plddt, best_loss, current_loss = None, np.inf, np.inf
        current_seq = (self._params["seq"] + self._inputs["bias"]).argmax(-1)
        if seq_logits is None:
            seq_logits = 0

        # run!
        if verbose:
            print("Running MCMC with simulated annealing...")
        for i in range(steps):

            # update temperature
            T = T_init * (np.exp(np.log(0.5) / half_life) ** i)

            # mutate sequence
            if i == 0:
                mut_seq = current_seq
            else:
                mut_seq = self._mutate(seq=current_seq, plddt=plddt,
                                    logits=seq_logits +
                                    self._inputs["bias"],
                                    mutation_rate=mutation_rate)

            # get loss
            model_nums = self._get_model_nums(**model_flags)
            aux = self.predict(seq=mut_seq, return_aux=True,
                            verbose=False, model_nums=model_nums, **kwargs)
            loss = aux["log"]["loss"]

            # decide
            delta = loss - current_loss
            if i == 0 or delta < 0 or np.random.uniform() < np.exp(-delta / T):

                # accept
                (current_seq, current_loss) = (mut_seq, loss)

                plddt = aux["all"]["plddt"].mean(0)
                plddt = plddt[self._target_len:] if self.protocol == "binder" else plddt[:self._len]

                if loss < best_loss:
                    (best_loss, self._k) = (loss, i)
                    self.set_seq(seq=current_seq, bias=self._inputs["bias"])
                    self._save_results(save_best=save_best, verbose=verbose, stage="mcmc")
