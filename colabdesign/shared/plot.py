# import matplotlib
import numpy as np
from scipy.special import expit as sigmoid
from colabdesign.shared.protein import _np_kabsch, alphabet_list

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patheffects
from matplotlib import animation
from matplotlib.gridspec import GridSpec 
from matplotlib import collections as mcoll
try:
  import py3Dmol
except:
  print("py3Dmol not installed")
  
pymol_color_list = ["#33ff33","#00ffff","#ff33cc","#ffff00","#ff9999","#e5e5e5","#7f7fff","#ff7f00",
                    "#7fff7f","#199999","#ff007f","#ffdd5e","#8c3f99","#b2b2b2","#007fff","#c4b200",
                    "#8cb266","#00bfbf","#b27f7f","#fcd1a5","#ff7f7f","#ffbfdd","#7fffff","#ffff7f",
                    "#00ff7f","#337fcc","#d8337f","#bfff3f","#ff7fff","#d8d8ff","#3fffbf","#b78c4c",
                    "#339933","#66b2b2","#ba8c84","#84bf00","#b24c66","#7f7f7f","#3f3fa5","#a5512b"]

jalview_color_list = {"Clustal":           ["#80a0f0","#f01505","#00ff00","#c048c0","#f08080","#00ff00","#c048c0","#f09048","#15a4a4","#80a0f0","#80a0f0","#f01505","#80a0f0","#80a0f0","#ffff00","#00ff00","#00ff00","#80a0f0","#15a4a4","#80a0f0"],
                      "Zappo":             ["#ffafaf","#6464ff","#00ff00","#ff0000","#ffff00","#00ff00","#ff0000","#ff00ff","#6464ff","#ffafaf","#ffafaf","#6464ff","#ffafaf","#ffc800","#ff00ff","#00ff00","#00ff00","#ffc800","#ffc800","#ffafaf"],
                      "Taylor":            ["#ccff00","#0000ff","#cc00ff","#ff0000","#ffff00","#ff00cc","#ff0066","#ff9900","#0066ff","#66ff00","#33ff00","#6600ff","#00ff00","#00ff66","#ffcc00","#ff3300","#ff6600","#00ccff","#00ffcc","#99ff00"],
                      "Hydrophobicity":    ["#ad0052","#0000ff","#0c00f3","#0c00f3","#c2003d","#0c00f3","#0c00f3","#6a0095","#1500ea","#ff0000","#ea0015","#0000ff","#b0004f","#cb0034","#4600b9","#5e00a1","#61009e","#5b00a4","#4f00b0","#f60009","#0c00f3","#680097","#0c00f3"],
                      "Helix Propensity":  ["#e718e7","#6f906f","#1be41b","#778877","#23dc23","#926d92","#ff00ff","#00ff00","#758a75","#8a758a","#ae51ae","#a05fa0","#ef10ef","#986798","#00ff00","#36c936","#47b847","#8a758a","#21de21","#857a85","#49b649","#758a75","#c936c9"],
                      "Strand Propensity": ["#5858a7","#6b6b94","#64649b","#2121de","#9d9d62","#8c8c73","#0000ff","#4949b6","#60609f","#ecec13","#b2b24d","#4747b8","#82827d","#c2c23d","#2323dc","#4949b6","#9d9d62","#c0c03f","#d3d32c","#ffff00","#4343bc","#797986","#4747b8"],
                      "Turn Propensity":   ["#2cd3d3","#708f8f","#ff0000","#e81717","#a85757","#3fc0c0","#778888","#ff0000","#708f8f","#00ffff","#1ce3e3","#7e8181","#1ee1e1","#1ee1e1","#f60909","#e11e1e","#738c8c","#738c8c","#9d6262","#07f8f8","#f30c0c","#7c8383","#5ba4a4"],
                      "Buried Index":      ["#00a35c","#00fc03","#00eb14","#00eb14","#0000ff","#00f10e","#00f10e","#009d62","#00d52a","#0054ab","#007b84","#00ff00","#009768","#008778","#00e01f","#00d52a","#00db24","#00a857","#00e619","#005fa0","#00eb14","#00b649","#00f10e"]}

pymol_cmap = matplotlib.colors.ListedColormap(pymol_color_list)
    
def show_pdb(pdb_str, show_sidechains=False, show_mainchains=False,
             color="pLDDT", chains=None, Ls=None, vmin=50, vmax=90,
             color_HP=False, size=(800,480), hbondCutoff=4.0,
             animate=False):
  
  if chains is None:
    chains = 1 if Ls is None else len(Ls)

  view = py3Dmol.view(js='https://3dmol.org/build/3Dmol.js', width=size[0], height=size[1])
  if animate:
    view.addModelsAsFrames(pdb_str,'pdb',{'hbondCutoff':hbondCutoff})
  else:
    view.addModel(pdb_str,'pdb',{'hbondCutoff':hbondCutoff})
  if color == "pLDDT":
    view.setStyle({'cartoon': {'colorscheme': {'prop':'b','gradient': 'roygb','min':vmin,'max':vmax}}})
  elif color == "rainbow":
    view.setStyle({'cartoon': {'color':'spectrum'}})
  elif color == "chain":
    for n,chain,color in zip(range(chains),alphabet_list,pymol_color_list):
       view.setStyle({'chain':chain},{'cartoon': {'color':color}})
  if show_sidechains:
    BB = ['C','O','N']
    HP = ["ALA","GLY","VAL","ILE","LEU","PHE","MET","PRO","TRP","CYS","TYR"]
    if color_HP:
      view.addStyle({'and':[{'resn':HP},{'atom':BB,'invert':True}]},
                    {'stick':{'colorscheme':"yellowCarbon",'radius':0.3}})
      view.addStyle({'and':[{'resn':HP,'invert':True},{'atom':BB,'invert':True}]},
                    {'stick':{'colorscheme':"whiteCarbon",'radius':0.3}})
      view.addStyle({'and':[{'resn':"GLY"},{'atom':'CA'}]},
                    {'sphere':{'colorscheme':"yellowCarbon",'radius':0.3}})
      view.addStyle({'and':[{'resn':"PRO"},{'atom':['C','O'],'invert':True}]},
                    {'stick':{'colorscheme':"yellowCarbon",'radius':0.3}})
    else:
      view.addStyle({'and':[{'resn':["GLY","PRO"],'invert':True},{'atom':BB,'invert':True}]},
                    {'stick':{'colorscheme':f"WhiteCarbon",'radius':0.3}})
      view.addStyle({'and':[{'resn':"GLY"},{'atom':'CA'}]},
                    {'sphere':{'colorscheme':f"WhiteCarbon",'radius':0.3}})
      view.addStyle({'and':[{'resn':"PRO"},{'atom':['C','O'],'invert':True}]},
                    {'stick':{'colorscheme':f"WhiteCarbon",'radius':0.3}})  
  if show_mainchains:
    BB = ['C','O','N','CA']
    view.addStyle({'atom':BB},{'stick':{'colorscheme':f"WhiteCarbon",'radius':0.3}})
  view.zoomTo()
  if animate: view.animate()
  return view

def plot_pseudo_3D(xyz, c=None, ax=None, chainbreak=5, Ls=None,
                   cmap="gist_rainbow", line_w=2.0,
                   cmin=None, cmax=None, zmin=None, zmax=None,
                   shadow=0.95):

  def rescale(a, amin=None, amax=None):
    a = np.copy(a)
    if amin is None: amin = a.min()
    if amax is None: amax = a.max()
    a[a < amin] = amin
    a[a > amax] = amax
    return (a - amin)/(amax - amin)

  # make segments and colors for each segment
  xyz = np.asarray(xyz)
  if Ls is None:
    seg = np.concatenate([xyz[:,None],np.roll(xyz,1,0)[:,None]],axis=1)
    c_seg = np.arange(len(seg))[::-1] if c is None else (c + np.roll(c,1,0))/2
  else:
    Ln = 0
    seg = []
    c_seg = []
    for L in Ls:
      sub_xyz = xyz[Ln:Ln+L]
      seg.append(np.concatenate([sub_xyz[:,None],np.roll(sub_xyz,1,0)[:,None]],axis=1))
      if c is not None:
        sub_c = c[Ln:Ln+L]
        c_seg.append((sub_c + np.roll(sub_c,1,0))/2)
      Ln += L
    seg = np.concatenate(seg,0)
    c_seg = np.arange(len(seg))[::-1] if c is None else np.concatenate(c_seg,0)
  
  # set colors
  c_seg = rescale(c_seg,cmin,cmax)  
  if isinstance(cmap, str):
    if cmap == "gist_rainbow": 
      c_seg *= 0.75
    colors = matplotlib.cm.get_cmap(cmap)(c_seg)
  else:
    colors = cmap(c_seg)
  
  # remove segments that aren't connected
  seg_len = np.sqrt(np.square(seg[:,0] - seg[:,1]).sum(-1))
  if chainbreak is not None:
    idx = seg_len < chainbreak
    seg = seg[idx]
    seg_len = seg_len[idx]
    colors = colors[idx]

  seg_mid = seg.mean(1)
  seg_xy = seg[...,:2]
  seg_z = seg[...,2].mean(-1)
  order = seg_z.argsort()

  # add shade/tint based on z-dimension
  z = rescale(seg_z,zmin,zmax)[:,None]

  # add shadow (make lines darker if they are behind other lines)
  seg_len_cutoff = (seg_len[:,None] + seg_len[None,:]) / 2
  seg_mid_z = seg_mid[:,2]
  seg_mid_dist = np.sqrt(np.square(seg_mid[:,None] - seg_mid[None,:]).sum(-1))
  shadow_mask = sigmoid(seg_len_cutoff * 2.0 - seg_mid_dist) * (seg_mid_z[:,None] < seg_mid_z[None,:])
  np.fill_diagonal(shadow_mask,0.0)
  shadow_mask = shadow ** shadow_mask.sum(-1,keepdims=True)

  seg_mid_xz = seg_mid[:,:2]
  seg_mid_xydist = np.sqrt(np.square(seg_mid_xz[:,None] - seg_mid_xz[None,:]).sum(-1))
  tint_mask = sigmoid(seg_len_cutoff/2 - seg_mid_xydist) * (seg_mid_z[:,None] < seg_mid_z[None,:])
  np.fill_diagonal(tint_mask,0.0)
  tint_mask = 1 - tint_mask.max(-1,keepdims=True)

  colors[:,:3] = colors[:,:3] + (1 - colors[:,:3]) * (0.50 * z + 0.50 * tint_mask) / 3
  colors[:,:3] = colors[:,:3] * (0.20 + 0.25 * z + 0.55 * shadow_mask)

  set_lim = False
  if ax is None:
    fig, ax = plt.subplots()
    fig.set_figwidth(5)
    fig.set_figheight(5)
    set_lim = True
  else:
    fig = ax.get_figure()
    if ax.get_xlim() == (0,1):
      set_lim = True
      
  if set_lim:
    xy_min = xyz[:,:2].min() - line_w
    xy_max = xyz[:,:2].max() + line_w
    ax.set_xlim(xy_min,xy_max)
    ax.set_ylim(xy_min,xy_max)

  ax.set_aspect('equal')
    
  # determine linewidths
  width = fig.bbox_inches.width * ax.get_position().width
  linewidths = line_w * 72 * width / np.diff(ax.get_xlim())

  lines = mcoll.LineCollection(seg_xy[order], colors=colors[order], linewidths=linewidths,
                               path_effects=[matplotlib.patheffects.Stroke(capstyle="round")])
  
  return ax.add_collection(lines)

def plot_ticks(ax, Ls, Ln=None, add_yticks=False):
  if Ln is None: Ln = sum(Ls)
  L_prev = 0
  for L_i in Ls[:-1]:
    L = L_prev + L_i
    L_prev += L_i
    ax.plot([0,Ln],[L,L],color="black")
    ax.plot([L,L],[0,Ln],color="black")
  
  if add_yticks:
    ticks = np.cumsum([0]+Ls)
    ticks = (ticks[1:] + ticks[:-1])/2
    ax.yticks(ticks,alphabet_list[:len(ticks)])
def make_animation(seq, con=None, xyz=None, plddt=None, pae=None,
                    losses=None, pos_ref=None, line_w=2.0,
                    dpi=100, interval=60, color_msa="Taylor",
                    length=None, align_xyz=True, color_by="plddt",
                    # New arguments for multiple targets
                    extra_xyz=None,        # list of [xyz frames] for each extra target
                    extra_plddt=None,      # list of [plddt frames]
                    extra_pae=None,        # list of [pae frames]
                    extra_pos_refs=None,   # list of reference coords (same shape as extra_xyz[i][0])
                    struct_names=None,
                    roles=None,
                    **kwargs):
    """
    Create an animation of main + multiple target structures or contact maps.

    seq:    list of arrays representing the main target sequence(s) over frames
    con:    list of NxN arrays (predicted contact map) for the main target
    xyz:    list of Nx3 arrays (cartesian coords) for the main target
    plddt:  list of Nx1 arrays (pLDDT per residue) for the main target
    pae:    list of NxN arrays (predicted aligned error) for the main target

    pos_ref: reference coords for the main target
    length:  either integer or list, used for chain boundaries or alignment
    align_xyz: if True, will perform Kabsch alignment of each frame to pos_ref

    extra_xyz, extra_plddt, extra_pae, extra_pos_refs:
        each should be a list of length (# of extra targets).
        example:
            extra_xyz[i] is the list of frames for the i-th extra target
            extra_pos_refs[i] is the reference coords for the i-th extra target

    color_by: "plddt", "chain", "rainbow"
    interval: ms per frame in the animation

    Returns: an HTML5 video via ani.to_html5_video().
    """

    # Make sure these are lists even if None
    if extra_xyz       is None: extra_xyz       = []
    if extra_plddt     is None: extra_plddt     = []
    if extra_pae       is None: extra_pae       = []
    if extra_pos_refs  is None: extra_pos_refs  = []

    # local helper that calls your existing Kabsch
    def nankabsch(a, b, return_v=False, use_jax=False):
        ok = np.isfinite(a).all(axis=1) & np.isfinite(b).all(axis=1)
        a, b = a[ok], b[ok]
        return _np_kabsch(a, b, return_v=return_v, use_jax=use_jax)

    # -------------------------------
    # (A) Align main target (if xyz)
    # -------------------------------
    pos, pos_ref_full = None, None
    if xyz is not None and len(xyz) > 0:
        # If no pos_ref given, use the last frame
        if pos_ref is None:
            pos_ref = xyz[-1]

        # figure out length(s)
        if length is None:
            L = len(pos_ref)
            Ls = None
        elif isinstance(length, list):
            L = length[0]
            Ls = length
        else:
            L = length
            Ls = None

        # align frames to pos_ref if needed
        if align_xyz:
            pos_ref_trim = pos_ref[:L]
            mu = np.nanmean(pos_ref_trim, axis=0)
            pos_ref_trim -= mu

            aligned = []
            for frame_xyz in xyz:
                f_mu = np.nanmean(frame_xyz[:L], axis=0)
                aln = nankabsch(frame_xyz[:L] - f_mu, pos_ref_trim, use_jax=False)
                aligned.append((frame_xyz - f_mu) @ aln)
            pos = np.array(aligned)

            # final rotation for best view
            pos_mean = np.concatenate(pos, axis=0)
            m = np.nanmean(pos_mean, 0)
            rot_mtx = nankabsch(pos_mean - m, pos_mean - m, return_v=True, use_jax=False)
            pos = (pos - m) @ rot_mtx
            pos_ref_full = ((pos_ref - mu) - m) @ rot_mtx
        else:
            # no alignment, just unify
            pos_mean = np.concatenate(xyz, axis=0)
            m = np.nanmean(pos_mean, 0)
            rot_mtx = nankabsch(pos_mean - m, pos_mean - m, return_v=True, use_jax=False)
            pos = [(frame_xyz - m) @ rot_mtx for frame_xyz in xyz]
            pos_ref_full = (pos_ref - m) @ rot_mtx

    # ----------------------------------------------------
    # (B) Align each "extra" target in a loop
    # ----------------------------------------------------
    extra_positions = []      # list of arrays (aligned frames) for each target
    extra_positions_refs = [] # list of final reference coords
    n_extras = len(extra_xyz)

    for i in range(n_extras):
        # handle case with empty data
        if extra_xyz[i] is None or len(extra_xyz[i]) == 0:
            extra_positions.append(None)
            extra_positions_refs.append(None)
            continue

        # pick a reference
        cur_ref = extra_pos_refs[i] if i < len(extra_pos_refs) else extra_xyz[i][-1]
        if cur_ref is None:
            extra_positions.append(None)
            extra_positions_refs.append(None)
            continue

        # figure out length for this extra target
        if length is None:
            L_off = len(cur_ref)
            Ls_off = None
        elif isinstance(length, list):
            if i < len(length):
                L_off = length[i]
            else:
                L_off = length[0]
            Ls_off = length
        else:
            L_off = length
            Ls_off = None

        # alignment
        if align_xyz:
            ref_trim = cur_ref[:L_off]
            mu_off = np.nanmean(ref_trim, axis=0)
            ref_trim -= mu_off

            new_pos = []
            for frame_xyz in extra_xyz[i]:
                f_mu = np.nanmean(frame_xyz[:L_off], axis=0)
                aln = nankabsch(frame_xyz[:L_off] - f_mu, ref_trim, use_jax=False)
                new_pos.append((frame_xyz - f_mu) @ aln)
            new_pos = np.array(new_pos)

            # final rotation
            off_pos_mean = np.concatenate(new_pos, axis=0)
            m_off = np.nanmean(off_pos_mean, 0)
            rot_off = nankabsch(off_pos_mean - m_off, off_pos_mean - m_off,
                                return_v=True, use_jax=False)
            new_pos = (new_pos - m_off) @ rot_off
            ref_full = ((cur_ref - mu_off) - m_off) @ rot_off
        else:
            # no alignment
            off_pos_mean = np.concatenate(extra_xyz[i], axis=0)
            m_off = np.nanmean(off_pos_mean, 0)
            rot_off = nankabsch(off_pos_mean - m_off, off_pos_mean - m_off,
                                return_v=True, use_jax=False)
            new_pos = [(frame_xyz - m_off) @ rot_off for frame_xyz in extra_xyz[i]]
            ref_full = (cur_ref - m_off) @ rot_off

        extra_positions.append(new_pos)
        extra_positions_refs.append(ref_full)

    # -------------------------------------------------------------
    # (C) Figure out how many columns for the subplot grid
    # -------------------------------------------------------------
    has_pae = (pae is not None and len(pae) > 0)
    # each extra target can have up to 3 columns: xyz, seq, pae
    # main can have 2 or 3 columns (2 if no PAE, 3 if PAE)
    base_cols = 3 if has_pae else 2
    # for each extra target, we also assume 3 columns:
    #  - (xyz, seq, pae) if that target has a non-empty pae
    #  - if there's no pae, we still allocate 2 columns for (xyz, seq)
    # but simpler to just do 3 columns each time to keep uniform
    extra_cols = 0
    for i in range(n_extras):
        # if we truly have no data, skip
        if extra_positions[i] is None: 
            continue
        # you might want to check if extra_pae[i] is None to do 2 or 3 columns
        # but simpler just to do 3 for all
        extra_cols += 3

    total_cols = base_cols + extra_cols
    fig = plt.figure()
    gs = GridSpec(4, total_cols, figure=fig)
    fig.set_figwidth(4 + 3*n_extras)
    fig.set_figheight(6)
    fig.set_dpi(dpi)
    fig.subplots_adjust(top=0.95, bottom=0.1, right=0.95, left=0.05, hspace=0, wspace=0)

    # main subplots
    col_idx = 0
    if has_pae:
        ax1 = fig.add_subplot(gs[:3, col_idx:(col_idx+2)])
        ax2 = fig.add_subplot(gs[3:, col_idx:(col_idx+2)])
        ax3 = fig.add_subplot(gs[:3, (col_idx+2):(col_idx+3)])
        col_idx += 3
    else:
        ax1 = fig.add_subplot(gs[:3, col_idx:(col_idx+2)])
        ax2 = fig.add_subplot(gs[3:, col_idx:(col_idx+2)])
        ax3 = None
        col_idx += 2

    # extra targets subplots
    extra_axes = []
    for i in range(n_extras):
        if extra_positions[i] is None:
            extra_axes.append((None,None,None))
            continue
        # xyz + seq
        ax_xyz = fig.add_subplot(gs[:3, col_idx:(col_idx+2)])
        ax_seq = fig.add_subplot(gs[3:, col_idx:(col_idx+2)])
        col_idx += 2
        # pae
        ax_pae = None
        # if you want to check whether that target has real PAE data:
        if extra_pae[i] is not None and len(extra_pae[i]) > 0:
            ax_pae = fig.add_subplot(gs[:3, col_idx:(col_idx+1)])
            col_idx += 1
        extra_axes.append((ax_xyz, ax_seq, ax_pae))

    # set titles for the main
    if xyz is not None:
        if struct_names is not None and roles is not None:
           ax1.set_title(f"{roles[0]}: {struct_names[0]}")
        else:
            ax1.set_title("Main: N→C" if plddt is not None else "Main structure")
    else:
        ax1.set_title("Predicted contact map")
    if has_pae and ax3 is not None:
        ax3.set_title("pAE")
        ax3.set_xticks([])
        ax3.set_yticks([])

    ax2.set_xlabel("positions")
    ax2.set_yticks([])
    if seq is not None and len(seq) > 0 and seq[0].shape[0] > 1:
        ax2.set_ylabel("sequences")
    else:
        ax2.set_ylabel("amino acids")

    # set titles for each extra target
    for i,(ax_xyz, ax_seq, ax_pae) in enumerate(extra_axes):
        if ax_xyz is None: 
            continue
        if struct_names is not None and roles is not None:
            ax_xyz.set_title(f"{roles[i+1]}: {struct_names[i+1]}")
        else:
            ax_xyz.set_title(f"Target {i+1}")
        ax_seq.set_xlabel("positions")
        ax_seq.set_yticks([])
        ax_seq.set_ylabel(f"{struct_names[i+1]} amino acids")
        if ax_pae is not None:
            ax_pae.set_title(f"T{i+1} pAE")
            ax_pae.set_xticks([])
            ax_pae.set_yticks([])

    # Adjust axis limits for main xyz
    if xyz is not None and pos is not None and len(pos) > 0:
        main_ok = pos_ref_full[np.isfinite(pos_ref_full).all(1)]
        pos_ok = [p[np.isfinite(p).all(1)] for p in pos]
        x_min = min(x.min(0)[0] for x in pos_ok)
        x_max = max(x.max(0)[0] for x in pos_ok)
        y_min = min(x.min(0)[1] for x in pos_ok)
        y_max = max(x.max(0)[1] for x in pos_ok)
        # expand
        x_min -= 5; x_max += 5; y_min -= 5; y_max += 5
        # keep square
        dx = x_max - x_min
        dy = y_max - y_min
        if dx > dy:
            diff = (dx - dy)/2
            y_min -= diff
            y_max += diff
        else:
            diff = (dy - dx)/2
            x_min -= diff
            x_max += diff
        ax1.set_xlim(x_min, x_max)
        ax1.set_ylim(y_min, y_max)
        ax1.set_xticks([])
        ax1.set_yticks([])

    # Adjust axis limits for each extra target
    for i,(ax_xyz, ax_seq, ax_pae) in enumerate(extra_axes):
        if ax_xyz is None: continue
        p = extra_positions[i]
        if p is None or len(p) == 0: continue
        ref_full = extra_positions_refs[i]
        ref_ok = ref_full[np.isfinite(ref_full).all(1)]
        p_ok = [f[np.isfinite(f).all(1)] for f in p]
        ox_min = min(x.min(0)[0] for x in p_ok)
        ox_max = max(x.max(0)[0] for x in p_ok)
        oy_min = min(x.min(0)[1] for x in p_ok)
        oy_max = max(x.max(0)[1] for x in p_ok)
        ox_min -= 5; ox_max += 5; oy_min -= 5; oy_max += 5
        odx = ox_max - ox_min
        ody = oy_max - oy_min
        if odx > ody:
            diff = (odx - ody)/2
            oy_min -= diff
            oy_max += diff
        else:
            diff = (ody - odx)/2
            ox_min -= diff
            ox_max += diff
        ax_xyz.set_xlim(ox_min, ox_max)
        ax_xyz.set_ylim(oy_min, oy_max)
        ax_xyz.set_xticks([])
        ax_xyz.set_yticks([])

    # for coloring sequences
    cmap = matplotlib.colors.ListedColormap(jalview_color_list[color_msa])
    vmax = len(jalview_color_list[color_msa]) - 1

    ims = []
    n_frames = len(seq) if seq is not None else 0

    # build frames
    for k in range(n_frames):
        frame_artists = []

        # (1) main target structure or contact map
        if xyz is not None and pos is not None:
            c = None
            if color_by == "plddt" and plddt is not None and len(plddt) > k:
                c = plddt[k]
                frame_artists.append(plot_pseudo_3D(pos[k], c=c, Ls=Ls, cmin=0.5, cmax=0.9, ax=ax1, line_w=line_w))
            elif color_by == "chain" and length is not None:
                chain_color = []
                if isinstance(length, list):
                    offset = 0
                    for i_chain, Lc in enumerate(length):
                        chain_color.extend([i_chain]*Lc)
                        offset += Lc
                else:
                    chain_color = np.arange(length)
                frame_artists.append(plot_pseudo_3D(pos[k], c=np.array(chain_color), Ls=Ls, cmin=0, cmax=39,
                                                    ax=ax1, line_w=line_w))
            else:
                # rainbow
                c = np.arange(pos[k].shape[0])[::-1]
                frame_artists.append(plot_pseudo_3D(pos[k], c=c, Ls=Ls, cmin=0, cmax=pos[k].shape[0],
                                                    ax=ax1, line_w=line_w))
        else:
            # fallback to contact map
            if con is not None and len(con) > k:
                im = ax1.imshow(con[k], animated=True, cmap="Greys",vmin=0, vmax=1,
                                extent=(0, con[k].shape[0], con[k].shape[0], 0))
                frame_artists.append(im)

        # (2) main seq
        if seq is not None and len(seq) > k:
            if seq[k].shape[0] == 1:
                # pseudo for single sequence
                im_seq = ax2.imshow(seq[k][0].T, animated=True, cmap="bwr_r", vmin=-1, vmax=1)
                frame_artists.append(im_seq)
            else:
                # argmax-based coloring
                im_seq = ax2.imshow(seq[k].argmax(-1), animated=True, cmap=cmap, vmin=0, vmax=vmax, interpolation="none")
                frame_artists.append(im_seq)

        # (3) main pae if present
        if has_pae and len(pae) > k and pae[k] is not None and ax3 is not None:
            im_pae = ax3.imshow(pae[k], animated=True, cmap="bwr", vmin=0, vmax=30,
                                extent=(0, pae[k].shape[0], pae[k].shape[0], 0))
            frame_artists.append(im_pae)

        # (4) each extra target
        for i in range(n_extras):
            ax_xyz, ax_seq, ax_pae = extra_axes[i]
            if ax_xyz is None: 
                continue
            # get aligned positions for frame k
            cur_positions = extra_positions[i]
            if cur_positions is None or len(cur_positions) <= k:
                continue

            # plot the structure
            c2 = None
            if extra_plddt[i] is not None and len(extra_plddt[i]) > k:
                c2 = extra_plddt[i][k]
                frame_artists.append(plot_pseudo_3D(cur_positions[k], c=c2, Ls=None,
                                                    cmin=0.5, cmax=0.9, ax=ax_xyz, line_w=line_w))
            else:
                # fallback rainbow
                Lc_off = cur_positions[k].shape[0]
                c2 = np.arange(Lc_off)[::-1]
                frame_artists.append(plot_pseudo_3D(cur_positions[k], c=c2, Ls=None,
                                                    cmin=0, cmax=Lc_off, ax=ax_xyz, line_w=line_w))

            # If you want a separate "extra_seq[i]" to show, pass it in similarly
            # For demonstration, we'll reuse main seq or just skip it:
            if seq is not None and len(seq) > k:
                if seq[k].shape[0] == 1:
                    im_es = ax_seq.imshow(seq[k][0].T, animated=True, cmap="bwr_r", vmin=-1, vmax=1)
                    frame_artists.append(im_es)
                else:
                    im_es = ax_seq.imshow(seq[k].argmax(-1), animated=True, cmap=cmap,
                                        vmin=0, vmax=vmax, interpolation="none")
                    frame_artists.append(im_es)

            # extra pae
            if extra_pae[i] is not None and len(extra_pae[i]) > k and ax_pae is not None:
                e_pae = extra_pae[i][k]
                im_ep = ax_pae.imshow(e_pae, animated=True, cmap="bwr", vmin=0, vmax=30,
                                    extent=(0, e_pae.shape[0], e_pae.shape[0], 0))
                frame_artists.append(im_ep)

        ims.append(frame_artists)

    # if length is given, add boundary ticks
    if length is not None:
        # main
        Ls_main = length if isinstance(length, list) else [length,None]
        if con is not None and len(con) > 0:
            plot_ticks(ax1, Ls_main, con[0].shape[0])
        if has_pae and ax3 is not None and len(pae) > 0:
            plot_ticks(ax3, Ls_main, pae[0].shape[0])

        # similarly for each extra target
        # if you want to show chain boundaries on the structure or PAE
        # you'd do a similar call to plot_ticks on each ax_pae

    ani = animation.ArtistAnimation(fig, ims, blit=False, interval=interval)
    plt.close(fig)
    return ani.to_html5_video()