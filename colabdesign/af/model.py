import jax
import jax.numpy as jnp
import numpy as np
from inspect import signature

from colabdesign.af.alphafold.model import data, config, model, all_atom

from colabdesign.shared.model import design_model
from colabdesign.shared.utils import Key

from colabdesign.af.prep import _af_prep
from colabdesign.af.loss import _af_loss, get_plddt, get_pae, get_ptm
from colabdesign.af.loss import get_contact_map, get_seq_ent_loss, get_mlm_loss
from colabdesign.af.utils import _af_utils
from colabdesign.af.design import _af_design
from colabdesign.af.inputs import _af_inputs, update_seq, update_aatype
from colabdesign.shared.utils import copy_dict
################################################################
# MK_DESIGN_MODEL - initialize model, and put it all together
################################################################

class mk_af_model(design_model, _af_inputs, _af_loss, _af_prep, _af_design, _af_utils):
    def __init__(self,
                 protocol="fixbb",
                 use_multimer=False,
                 use_templates=False,
                 debug=False,
                 data_dir=".",
                 **kwargs):
        assert protocol in ["fixbb", "hallucination", "binder", "partial"]
        self.protocol = protocol
        self._num = kwargs.pop("num_seq", 1)
        self._args = {
            "use_templates": use_templates,
            "use_multimer": use_multimer,
            "use_bfloat16": True,
            "recycle_mode": "last",
            "use_mlm": False,
            "realign": True,
            "debug": debug,
            "repeat": False,
            "homooligomer": False,
            "copies": 1,
            "optimizer": "sgd",
            "best_metric": "loss",
            "traj_iter": 1,
            "traj_max": 10000,
            "clear_prev": True,
            "use_dgram": False,
            "shuffle_first": True,
            "use_remat": True,
            "alphabet_size": 20,
            "use_initial_guess": False,
            "use_initial_atom_pos": False
        }
        if self.protocol == "binder":
            self._args["use_templates"] = True

        self.opt = {
            "dropout": True,
            "pssm_hard": False,
            "learning_rate": 0.1,
            "norm_seq_grad": True,
            "num_recycles": 0,
            "num_models": 1,
            "sample_models": True,
            "temp": 1.0,
            "soft": 0.0,
            "hard": 0.0,
            "alpha": 2.0,
            "con": {
                "num": 2,
                "cutoff": 14.0,
                "binary": False,
                "seqsep": 9,
                "num_pos": float("inf")
            },
            "i_con": {
                "num": 1,
                "cutoff": 21.6875,
                "binary": False,
                "num_pos": float("inf")
            },
            "template": {"rm_ic": False},
            "weights": {
                "seq_ent": 0.0,
                "plddt": 0.0,
                "pae": 0.0,
                "exp_res": 0.0,
                "helix": 0.0,
            },
            "fape_cutoff": 10.0
        }
        self._params = {}
        self._inputs = {}
        # Ensure this attribute exists even when no offtargets are configured
        self._offtarget_inputs = {}
        self._tmp = {
            "traj": {"seq": [], "xyz": [], "plddt": [], "pae": []},
            "log": [],
            "best": {}
        }

        # handle user overrides
        if "initial_guess" in kwargs:
            kwargs["use_initial_guess"] = kwargs.pop("initial_guess")
        model_names = kwargs.pop("model_names", None)
        # override self._args / self.opt from kwargs
        for k in list(kwargs.keys()):
            if k in self._args:
                self._args[k] = kwargs.pop(k)
            if k in self.opt:
                self.opt[k] = kwargs.pop(k)

        # gather user callbacks
        self._callbacks = {
            "model": {
                "pre":  kwargs.pop("pre_callback", None),
                "post": kwargs.pop("post_callback", None),
                "loss": kwargs.pop("loss_callback", None),
            },
            "design": {
                "pre":  kwargs.pop("pre_design_callback", None),
                "post": kwargs.pop("post_design_callback", None),
            }
        }
        for m, n in self._callbacks.items():
            for cbk, val in n.items():
                if val is None:
                    val = []
                if not isinstance(val, list):
                    val = [val]
                self._callbacks[m][cbk] = val

        if self._args["use_mlm"]:
            self.opt["mlm_dropout"] = 0.15
            self.opt["weights"]["mlm"] = 0.1

        assert len(kwargs) == 0, f"ERROR: the following inputs were not set: {kwargs}"

        #############################
        # configure alphafold
        #############################
        if self._args["use_multimer"]:
            self._cfg = config.model_config("model_1_multimer")
            self.opt["pssm_hard"] = True
        else:
            if use_templates:
                self._cfg = config.model_config("model_1_ptm")
            else:
                self._cfg = config.model_config("model_3_ptm")

        if self._args["recycle_mode"] in ["average", "first", "last", "sample"]:
            num_recycles = 0
        else:
            num_recycles = self.opt["num_recycles"]
        self._cfg.model.num_recycle = num_recycles
        self._cfg.model.global_config.use_remat = self._args["use_remat"]
        self._cfg.model.global_config.use_dgram = self._args["use_dgram"]
        self._cfg.model.global_config.bfloat16 = self._args["use_bfloat16"]

        # load model params
        if model_names is None:
            model_names = []
            if self._args["use_multimer"]:
                model_names += [f"model_{k}_multimer_v3" for k in [1, 2, 3, 4, 5]]
            else:
                if self._args["use_templates"]:
                    model_names += [f"model_{k}_ptm" for k in [1, 2]]
                else:
                    model_names += [f"model_{k}_ptm" for k in [1, 2, 3, 4, 5]]

        self._model_params, self._model_names = [], []
        for model_name in model_names:
            params = data.get_model_haiku_params(model_name=model_name, data_dir=data_dir, fuse=True)
            if params is not None:
                if not self._args["use_multimer"] and not self._args["use_templates"]:
                    # remove template modules
                    params = {k: v for k, v in params.items() if "template" not in k}
                self._model_params.append(params)
                self._model_names.append(model_name)
            else:
                print(f"WARNING: '{model_name}' not found")

        #####################################
        # set protocol-specific functions
        #####################################
        self.opt_offtarget = copy_dict(self.opt)

        idx = ["fixbb", "hallucination", "binder", "partial"].index(self.protocol)
        self.prep_inputs = [self._prep_fixbb, 
                            self._prep_hallucination, 
                            self._prep_binder, 
                            self._prep_partial][idx]
        self._get_loss = [self._loss_fixbb, 
                          self._loss_hallucination, 
                          self._loss_binder, 
                          self._loss_partial][idx]

        # If "binder", define multi-offtarget placeholders
        if self.protocol == "binder":
            # We'll store off-target info in a list
            self._offtargets = []  # each item is a dict
            self.prep_offtarget_inputs = self._prep_binder_offtargets
            self._get_offtarget_loss = self._loss_binder_offtarget
    def _get_model(self, cfg, callback=None):

        a = self._args
        runner = model.RunModel(cfg,
                                recycle_mode=a["recycle_mode"],
                                use_multimer=a["use_multimer"])

        # setup function to get gradients
        def _model(params, model_params, inputs, key, seq_key=None):
            inputs["params"] = params
            opt = inputs["opt"]

            aux = {}
            
            key = Key(key=key).get

            #######################################################################
            # INPUTS
            #######################################################################
            # get sequence. A caller-provided seq_key is shared across
            # ensemble models/recycles so stochastic sequence sampling stays
            # identical. Still consume the local sequence subkey so downstream
            # model-side randomness keeps the same key cadence as before.
            seq_sample_key = key()
            seq = self._get_seq(inputs, aux, seq_key if seq_key is not None else seq_sample_key)

            # update sequence features
            pssm = jnp.where(opt["pssm_hard"], seq["hard"], seq["pseudo"])
            if a["use_mlm"]:
                shape = seq["pseudo"].shape[:2]
                mlm = jax.random.bernoulli(key(), opt["mlm_dropout"], shape)
                update_seq(seq["pseudo"], inputs, seq_pssm=pssm, mlm=mlm)
            else:
                update_seq(seq["pseudo"], inputs, seq_pssm=pssm)

            # update amino acid sidechain identity
            update_aatype(seq["pseudo"][0].argmax(-1), inputs)

            # define masks
            inputs["msa_mask"] = jnp.where(
                inputs["seq_mask"], inputs["msa_mask"], 0)

            inputs["seq"] = aux["seq"]

            # update template features
            inputs["mask_template_interchain"] = opt["template"]["rm_ic"]
            if a["use_templates"]:
                self._update_template(inputs, key())

            # set dropout
            inputs["use_dropout"] = opt["dropout"]

            if "batch" not in inputs:
                inputs["batch"] = None

            # pre callback
            for fn in self._callbacks["model"]["pre"]:
                fn_args = {"inputs": inputs, "opt": opt, "aux": aux,
                           "seq": seq, "key": key(), "params": params}
                sub_args = {k: fn_args.get(k, None)
                            for k in signature(fn).parameters}
                fn(**sub_args)

            #######################################################################
            # OUTPUTS
            #######################################################################
            outputs = runner.apply(model_params, key(), inputs)

            # add aux outputs
            aux.update({"atom_positions": outputs["structure_module"]["final_atom_positions"],
                        "atom_mask":      outputs["structure_module"]["final_atom_mask"],
                        "residue_index":  inputs["residue_index"],
                        "aatype":         inputs["aatype"],
                        "plddt":          get_plddt(outputs),
                        "pae":            get_pae(outputs),
                        "ptm":            get_ptm(inputs, outputs),
                        "i_ptm":          get_ptm(inputs, outputs, interface=True),
                        "cmap":           get_contact_map(outputs, opt["con"]["cutoff"]),
                        "i_cmap":         get_contact_map(outputs, opt["i_con"]["cutoff"]),
                        "prev":           outputs["prev"]})

            #######################################################################
            # LOSS
            #######################################################################
            aux["losses"] = {}

            # add protocol specific losses
            self._get_loss(inputs=inputs, outputs=outputs, aux=aux)

            # sequence entropy loss
            aux["losses"].update(get_seq_ent_loss(inputs))

            # experimental masked-language-modeling
            if a["use_mlm"]:
                aux["mlm"] = outputs["masked_msa"]["logits"]
                mask = jnp.where(inputs["seq_mask"], mlm, 0)
                aux["losses"].update(get_mlm_loss(
                    outputs, mask=mask, truth=seq["pssm"]))

            # run user defined callbacks
            for c in ["loss", "post"]:
                for fn in self._callbacks["model"][c]:
                    fn_args = {"inputs": inputs, "outputs": outputs, "opt": opt,
                               "aux": aux, "seq": seq, "key": key(), "params": params}
                    sub_args = {k: fn_args.get(k, None)
                                for k in signature(fn).parameters}
                    if c == "loss":
                        aux["losses"].update(fn(**sub_args))
                    if c == "post":
                        fn(**sub_args)

            # save for debugging
            if a["debug"]:
                aux["debug"] = {"inputs": inputs, "outputs": outputs}

            # weighted loss
            w = opt["weights"]
            loss = sum([v * w[k] if k in w else v for k,
                       v in aux["losses"].items()])
            return loss, aux

        return {"grad_fn": jax.jit(jax.value_and_grad(_model, has_aux=True, argnums=0)),
                "fn": jax.jit(_model), "runner": runner}

    def _get_model_offtargets(self):
        """
        Build a simple Python-level aggregator function that loops over
        self._offtargets, calling a jitted function for each off-target.
        
        We'll return {"multi_fn": aggregator}, so you can do something like:
            off_model = self._get_model_offtargets()
            results = off_model["multi_fn"](params, model_params, key)
        """
        def multi_offtarget_forward_and_grad(params, model_params, key):
            results_list = []

            # Plain Python loop over each off‐target
            for i, offdict in enumerate(self._offtargets):
                # Use the per-offtarget length from inputs rather than a shared/global value
                try:
                    off_len = int(offdict["inputs"]["residue_index"].shape[0])
                except Exception:
                    off_len = int(offdict["inputs"].get("offtarget_len", 0))

                # local function that calls `_single_offtarget_forward(...)`
                # passing off_len in as a normal argument
                def single_offtarget_forward(params_, model_params_, offdict_, off_len_, key_):
                    # get references
                    off_inputs = offdict_["inputs"]
                    off_opt    = offdict_["opt_offtarget"]
                    # call your single-offtarget function
                    return self._single_offtarget_forward(
                        params_, model_params_,
                        off_inputs, off_opt, off_len_,
                        key_
                    )

                # jitted + grad version
                single_grad_fn = jax.jit(
                    jax.value_and_grad(
                        single_offtarget_forward,
                        has_aux=True,
                        argnums=0
                    )
                )

                # call it
                (loss_val, aux_dict), grads = single_grad_fn(params, model_params, offdict, off_len, key)
                results_list.append(((loss_val, aux_dict), grads))

            return results_list

        # Return that aggregator
        return {"multi_fn": multi_offtarget_forward_and_grad}

    def _get_model_offtarget(self, cfg):
        """
        Minimal single-offtarget "model" so that
        _prep_offtarget_model can call self._get_model_offtarget(self._cfg).
        """
        runner = model.RunModel(cfg,
                                recycle_mode=self._args["recycle_mode"],
                                use_multimer=self._args["use_multimer"])

        def _model_offtarget(params, model_params, off_inputs, key, seq_key=None):
            # off_opt = off_inputs.get("opt", self.opt_offtarget)
            # off_len = off_inputs["offtarget_len_static"]

            return self._single_offtarget_forward(
                params, model_params, off_inputs, key, seq_key=seq_key
            )

        grad_fn = jax.jit(jax.value_and_grad(_model_offtarget, has_aux=True, argnums=0))
        return {"fn": jax.jit(_model_offtarget), "grad_fn": grad_fn, "runner": runner}

    def _single_offtarget_forward(self,
                                params, model_params,
                                off_inputs,
                                key,
                                seq_key=None):
        """
        Single off-target forward pass, receiving a normal Python int off_len
        (already extracted in the Python loop).
        """
        # 1) build runner
        runner = model.RunModel(
            self._cfg,
            recycle_mode=self._args["recycle_mode"],
            use_multimer=self._args["use_multimer"]
        )
        aux = {}
        key = Key(key=key).get

        # 2) fill in fields
        off_inputs["params"] = params
        off_opt = off_inputs["opt"]
        # Derive lengths from static shapes to avoid tracer -> int() issues under JIT
        # binder length equals the learnable binder logits length
        binder_len = int(params["seq"].shape[1])
        # total sequence length for this off-target comes from batch aatype
        if "batch" in off_inputs and "aatype" in off_inputs["batch"]:
            total_len = int(off_inputs["batch"]["aatype"].shape[0])
        elif "residue_index" in off_inputs:
            total_len = int(off_inputs["residue_index"].shape[0])
        else:
            total_len = binder_len
        off_len = max(total_len - binder_len, 0)
        # 3) If you want to do the slicing logic for the off-target region, do it here:
        #    e.g. adjusting the batch array if needed
        #    for example:
        # off_inputs["batch"]["aatype"] = off_inputs["batch"]["aatype"][:off_len]

        # 4) get sequence from self._get_seq_offtarget
        #    NOTE: remove any reference to self._offtarget_lens[off_idx] from inside that function!
        # A caller-provided seq_key is shared with the target pass and other
        # ensemble models so all tasks evaluate the same binder sequence. Still
        # consume the local sequence subkey so downstream model-side randomness
        # keeps the same key cadence as before.
        seq_sample_key = key()
        seq = self._get_seq_offtarget(
            off_inputs, aux, off_len, seq_key if seq_key is not None else seq_sample_key
        )

        # 5) do your usual steps (pssm, mlm, update_aatype, etc.)
        pssm = jnp.where(off_opt["pssm_hard"], seq["hard"], seq["pseudo"])
        if self._args["use_mlm"]:
            shape = seq["pseudo"].shape[:2]
            mlm = jax.random.bernoulli(key(), off_opt["mlm_dropout"], shape)
            update_seq(seq["pseudo"], off_inputs, seq_pssm=pssm, mlm=mlm)
        else:
            update_seq(seq["pseudo"], off_inputs, seq_pssm=pssm)

        update_aatype(seq["pseudo"][0].argmax(-1), off_inputs)
        off_inputs["msa_mask"] = jnp.where(off_inputs["seq_mask"], off_inputs["msa_mask"], 0)
        off_inputs["seq"] = aux["seq"]

        # -------------- Run "pre" callbacks (same as in _model) --------------
        for fn in self._callbacks["model"]["pre"]:
            fn_args = {
                "inputs": off_inputs, 
                "opt":    off_opt, 
                "aux":    aux,
                "seq":    seq, 
                "key":    key(), 
                "params": params
            }
            sub_args = {
                k: fn_args.get(k, None) for k in signature(fn).parameters
            }
            fn(**sub_args)
        # 6) template if needed
        off_inputs["mask_template_interchain"] = off_opt["template"]["rm_ic"]
        if self._args["use_templates"]:
            self._update_template_offtarget(off_inputs, key())

        off_inputs["use_dropout"] = off_opt["dropout"]
        if "batch" not in off_inputs:
            off_inputs["batch"] = None

        # 7) run the model
        outputs = runner.apply(model_params, key(), off_inputs)

        # 8) gather
        aux.update({
            "atom_positions": outputs["structure_module"]["final_atom_positions"],
            "atom_mask":      outputs["structure_module"]["final_atom_mask"],
            "residue_index":  off_inputs["residue_index"],
            "aatype":         off_inputs["aatype"],
            "plddt":          get_plddt(outputs),
            "pae":            get_pae(outputs),
            "ptm":            get_ptm(off_inputs, outputs),
            "i_ptm":          get_ptm(off_inputs, outputs, interface=True),
            "cmap":           get_contact_map(outputs, off_opt["con"]["cutoff"]),
            "i_cmap":         get_contact_map(outputs, off_opt["i_con"]["cutoff"]),
            "prev":           outputs["prev"]
        })
        aux["losses"] = {}

        # 9) off-target–specific losses
        self._loss_binder_offtarget(inputs=off_inputs, outputs=outputs, aux=aux)

        # 10) sequence entropy / mlm
        aux["losses"].update(get_seq_ent_loss(off_inputs))
        if self._args["use_mlm"]:
            aux["mlm"] = outputs["masked_msa"]["logits"]
            mask = jnp.where(off_inputs["seq_mask"], mlm, 0)
            aux["losses"].update(get_mlm_loss(outputs, mask=mask, truth=seq["pssm"]))

        # user callbacks
        for c in ["loss", "post"]:
            for fn in self._callbacks["model"][c]:
                fn_args = {
                    "inputs":  off_inputs,
                    "outputs": outputs,
                    "opt":     off_opt,
                    "aux":     aux,
                    "seq":     seq,
                    "key":     key(),
                    "params":  params
                }
                sub_args = {
                    k: fn_args.get(k, None)
                    for k in signature(fn).parameters
                }
                if c == "loss":
                    aux["losses"].update(fn(**sub_args))
                else:
                    fn(**sub_args)

        # 11) final weighted loss
        w = off_opt["weights"]
        loss_val = sum(
            (v * w[k] if k in w else v)
            for k, v in aux["losses"].items()
        )
        return loss_val, aux
