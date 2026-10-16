import os
import glob
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit
from scipy import stats

def get_ca_coords(pdb_path):
    coords = {}
    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith('ATOM') and line[12:16].strip() == 'CA':
                res_num = int(line[22:26].strip())
                x = float(line[30:38].strip())
                y = float(line[38:46].strip())
                z = float(line[46:54].strip())
                coords[res_num] = np.array([x, y, z])
    return coords

def parse_double_mut_positions(mut_str):
    parts = mut_str.split(':')
    if len(parts) != 2:
        return None, None
    m1 = re.search(r'\d+', parts[0])
    m2 = re.search(r'\d+', parts[1])
    if m1 and m2:
        return int(m1.group(0)), int(m2.group(0))
    return None, None

def main():
    data_path = '/home/sareeves/software/esm-msr-release/data/tsuboyama/Tsuboyama2023_Dataset2_Dataset3_20230416.csv'
    pdb_dir = '/home/sareeves/software/esm-msr-release/data/tsuboyama/AlphaFold_model_PDBs'
    out_fig_dir = 'results/figures'
    out_tab_dir = 'results/tables'
    brain_fig_dir = '/home/sareeves/.gemini/antigravity/brain/73260da3-d1b9-4bbc-ba76-78297e940788/figures'
    os.makedirs(out_fig_dir, exist_ok=True)
    os.makedirs(out_tab_dir, exist_ok=True)
    os.makedirs(brain_fig_dir, exist_ok=True)

    paired_summary_path = os.path.join(out_tab_dir, 'paired_doubles_summary.csv')
    if os.path.exists(paired_summary_path):
        print("Loading precomputed paired doubles summary...")
        paired_df = pd.read_csv(paired_summary_path)
    else:
        print("Precomputed summary not found, loading full dataset...")
        cols = ['mut_type', 'WT_name', 'deltaG', 'deltaG_95CI', 'dG_ML', 'ddG_ML', 'pair_name']
        df = pd.read_csv(data_path, usecols=cols, low_memory=False)

        is_wt = df['mut_type'] == 'wt'
        is_double = df['pair_name'].notna() & df['mut_type'].str.contains(':', na=False)
        is_single = (~is_wt) & (~is_double)

        wt_stabs = df[is_wt].groupby('WT_name')['deltaG'].mean().to_dict()
        df_singles = df[is_single].copy()
        single_dict = df_singles.set_index(['WT_name', 'mut_type'])['deltaG'].to_dict()

        df_doubles = df[is_double].copy()
        pdb_cache = {}
        for pdb_file in glob.glob(os.path.join(pdb_dir, '*.pdb')):
            wt_key = os.path.basename(pdb_file)
            pdb_cache[wt_key] = get_ca_coords(pdb_file)

        records = []
        for idx, row in df_doubles.iterrows():
            wt = row['WT_name']
            mut = row['mut_type']
            obs_dG = row['deltaG']
            if wt not in wt_stabs:
                continue
            wt_dG = wt_stabs[wt]
            parts = mut.split(':')
            if len(parts) != 2:
                continue
            m1, m2 = parts[0], parts[1]
            dG1 = single_dict.get((wt, m1), None)
            dG2 = single_dict.get((wt, m2), None)
            if dG1 is None or dG2 is None:
                continue
            exp_dG = dG1 + dG2 - wt_dG
            dist = np.nan
            if wt in pdb_cache:
                p1, p2 = parse_double_mut_positions(mut)
                coords = pdb_cache[wt]
                if p1 in coords and p2 in coords:
                    dist = np.linalg.norm(coords[p1] - coords[p2])
            records.append({
                'WT_name': wt,
                'mut_type': mut,
                'dG_obs': obs_dG,
                'dG_exp': exp_dG,
                'wt_dG': wt_dG,
                'dist_ca': dist,
                'dG_95CI': row['deltaG_95CI']
            })
        paired_df = pd.DataFrame(records)
        paired_df.to_csv(paired_summary_path, index=False)

    # Filter strictly distant pairs (dist_ca > 15 Å) to isolate pure assay response
    distant_df = paired_df[(paired_df['dist_ca'] > 15.0) & (paired_df['dG_obs'] > -5.0) & (paired_df['dG_exp'] > -6.0) & (paired_df['dG_exp'] < 6.0)].copy()
    print(f"Distant non-interacting pairs (>15 Å): {len(distant_df)} variants.")

    x = distant_df['dG_exp'].values
    y = distant_df['dG_obs'].values

    # Mathematical Models to compare:
    # 1. Hard floor (Piecewise linear)
    def hard_floor(x, x_floor):
        return np.maximum(x, x_floor)

    # 2. Thermodynamic Softplus with physical floor at y_floor
    # Parametrized as: y = y_floor + tau * ln(1 + exp((x - x_floor)/tau))
    def softplus_floor(x, y_floor, x_floor, tau):
        z = (x - x_floor) / np.clip(tau, 1e-4, 10.0)
        return y_floor + tau * np.log1p(np.exp(np.clip(z, -30, 30)))

    # Fit Model 1: Hard floor via grid search
    grid_floors = np.linspace(-2.0, 1.0, 301)
    best_hf = None
    best_hf_rss = np.inf
    for f in grid_floors:
        pred = hard_floor(x, f)
        rss = np.sum((y - pred)**2)
        if rss < best_hf_rss:
            best_hf_rss = rss
            best_hf = f
    print(f"Hard Floor best threshold: {best_hf:.3f} kcal/mol, RSS: {best_hf_rss:.1f}")

    # Fit Model 2: Thermodynamic Softplus
    # Physical initial guess: floor near 0, transition at 0, tau ~ RT (0.59)
    popt_sp, _ = curve_fit(
        softplus_floor, x, y, 
        p0=[0.0, 0.0, 0.6], 
        bounds=([-2.0, -2.0, 0.05], [1.0, 2.0, 3.0])
    )
    pred_sp = softplus_floor(x, *popt_sp)
    rss_sp = np.sum((y - pred_sp)**2)
    print(f"Softplus best fit: y_floor={popt_sp[0]:.3f}, x_floor={popt_sp[1]:.3f}, tau={popt_sp[2]:.3f} kcal/mol, RSS: {rss_sp:.1f}")

    # Fit Model 3: Algebraic soft floor
    def algebraic_floor(x, y_floor, tau):
        diff = np.maximum(x - y_floor, 0.0)
        return y_floor + diff**2 / np.sqrt(diff**2 + tau**2)

    popt_alg, _ = curve_fit(algebraic_floor, x, y, p0=[0.0, 0.6], bounds=([-2.0, 0.01], [1.0, 3.0]))
    pred_alg = algebraic_floor(x, *popt_alg)
    rss_alg = np.sum((y - pred_alg)**2)

    # Model Evaluation Metrics
    n_pts = len(x)
    def calc_metrics(rss, k):
        rmse = np.sqrt(rss / n_pts)
        aic = n_pts * np.log(rss / n_pts) + 2 * k
        bic = n_pts * np.log(rss / n_pts) + k * np.log(n_pts)
        r2 = 1.0 - rss / np.sum((y - np.mean(y))**2)
        return {'RMSE': rmse, 'AIC': aic, 'BIC': bic, 'R2': r2}

    m_hf = calc_metrics(best_hf_rss, 1)
    m_sp = calc_metrics(rss_sp, 3)
    m_alg = calc_metrics(rss_alg, 2)

    comp_df = pd.DataFrame([
        {'Model': 'Hard Floor (Piecewise Linear)', 'Parameters': f'x_floor={best_hf:.3f}', **m_hf},
        {'Model': 'Thermodynamic Softplus', 'Parameters': f'y_fl={popt_sp[0]:.2f}, x_fl={popt_sp[1]:.2f}, tau={popt_sp[2]:.2f}', **m_sp},
        {'Model': 'Algebraic Soft Floor', 'Parameters': f'y_fl={popt_alg[0]:.2f}, tau={popt_alg[1]:.2f}', **m_alg}
    ])
    comp_df.to_csv(os.path.join(out_tab_dir, 'floor_model_comparison.csv'), index=False)
    print("\nModel Comparison Table:\n", comp_df)

    # =========================================================================
    # CREATE CLEAN, PRISTINE PUBLICATION FIGURE 3
    # =========================================================================
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    x_dense = np.linspace(-4.5, 4.5, 300)

    # -------------------------------------------------------------------------
    # Panel A: Scatter of Distant Pairs and Fitted Model Curves
    # -------------------------------------------------------------------------
    ax = axes[0, 0]
    hb = ax.hexbin(x, y, gridsize=70, cmap='Blues', mincnt=1, bins='log', alpha=0.85)
    cbar = plt.colorbar(hb, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Variant Count (log scale)', fontsize=10)
    
    ax.plot(x_dense, x_dense, color='gray', linestyle=':', lw=1.5, label='Identity line ($y = x$)')
    ax.plot(x_dense, hard_floor(x_dense, best_hf), color='#e41a1c', lw=2.5, 
            label=f'Hard Floor ($x_{{\\text{{floor}}}} = {best_hf:.2f}$)')
    ax.plot(x_dense, softplus_floor(x_dense, *popt_sp), color='#2ca02c', lw=3.0, 
            label=f'Softplus ($\\tau = {popt_sp[2]:.2f},\\, y_{{\\text{{floor}}}} = {popt_sp[0]:.2f}$)')
    
    ax.axvline(x=0.0, color='blue', linestyle=':', alpha=0.6)
    ax.axhline(y=0.0, color='blue', linestyle=':', alpha=0.6)
    ax.set_xlim(-4.0, 4.0)
    ax.set_ylim(-3.0, 4.5)
    ax.set_xlabel(r'Additive Expected Stability $\Delta G_{\text{expected}}$ (kcal/mol)', fontsize=11, fontweight='bold')
    ax.set_ylabel(r'Observed Stability $\Delta G_{\text{observed}}$ (kcal/mol)', fontsize=11, fontweight='bold')
    ax.set_title(r'(A) Empirical Assay Readout Curve: Distant Pairs ($d > 15$ Å)', fontsize=12, fontweight='bold')
    ax.legend(frameon=True, facecolor='white', loc='upper left', fontsize=9.5)

    # -------------------------------------------------------------------------
    # Panel B: Zoom on Transition Zone [-1.5, +1.5]
    # -------------------------------------------------------------------------
    ax = axes[0, 1]
    ax.hexbin(x, y, gridsize=45, cmap='Purples', mincnt=1, bins='log', alpha=0.85)
    ax.plot(x_dense, x_dense, color='gray', linestyle=':', lw=1.5, label='$y = x$')
    ax.plot(x_dense, hard_floor(x_dense, best_hf), color='#e41a1c', lw=2.5, label='Hard Floor')
    ax.plot(x_dense, softplus_floor(x_dense, *popt_sp), color='#2ca02c', lw=3.0, label='Softplus Floor')
    ax.axvspan(-0.5, 0.0, color='orange', alpha=0.18, label='Pile-up Zone $[-0.5, 0]$')
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_xlabel(r'$\Delta G_{\text{expected}}$ (kcal/mol)', fontsize=11, fontweight='bold')
    ax.set_ylabel(r'$\Delta G_{\text{observed}}$ (kcal/mol)', fontsize=11, fontweight='bold')
    ax.set_title('(B) Curvature in Transition Zone Confirms Soft Floor', fontsize=12, fontweight='bold')
    ax.legend(frameon=True, facecolor='white', loc='upper left', fontsize=9.5)

    # -------------------------------------------------------------------------
    # Panel C: Residual Analysis (Hard Floor vs Softplus)
    # -------------------------------------------------------------------------
    ax = axes[1, 0]
    res_hf = y - hard_floor(x, best_hf)
    res_sp = y - pred_sp
    
    res_bins = np.linspace(-3.0, 3.0, 31)
    res_bin_centers = 0.5 * (res_bins[:-1] + res_bins[1:])
    mean_res_hf = []
    mean_res_sp = []
    for b_l, b_h in zip(res_bins[:-1], res_bins[1:]):
        m = (x >= b_l) & (x < b_h)
        if m.sum() > 20:
            mean_res_hf.append(np.mean(res_hf[m]))
            mean_res_sp.append(np.mean(res_sp[m]))
        else:
            mean_res_hf.append(np.nan)
            mean_res_sp.append(np.nan)

    ax.plot(res_bin_centers, mean_res_hf, 'o-', color='#e41a1c', lw=2.2, label='Hard Floor Residual Bias')
    ax.plot(res_bin_centers, mean_res_sp, 's-', color='#2ca02c', lw=2.2, label='Softplus Residual Bias')
    ax.axhline(y=0.0, color='black', linestyle='--', lw=1.2)
    ax.axvline(x=0.0, color='blue', linestyle=':', lw=1.2)
    ax.set_xlim(-3.0, 3.0)
    ax.set_ylim(-0.6, 0.6)
    ax.set_xlabel(r'$\Delta G_{\text{expected}}$ (kcal/mol)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Mean Model Residual (kcal/mol)', fontsize=11, fontweight='bold')
    ax.set_title('(C) Residual Bias: Hard Floor Suffers Systematic Distortion', fontsize=12, fontweight='bold')
    ax.legend(frameon=True, facecolor='white', loc='lower right', fontsize=10)

    # -------------------------------------------------------------------------
    # Panel D: Goodness-of-Fit & Biophysical Softness Scale
    # -------------------------------------------------------------------------
    ax = axes[1, 1]
    
    # We display a dual comparison: Model R^2 and RMSE
    models = ['Hard Floor', 'Algebraic', 'Softplus']
    r2_vals = [m_hf['R2'], m_alg['R2'], m_sp['R2']]
    rmse_vals = [m_hf['RMSE'], m_alg['RMSE'], m_sp['RMSE']]
    
    x_pos = np.arange(len(models))
    w = 0.35
    
    b1 = ax.bar(x_pos - w/2, r2_vals, width=w, color='#377eb8', alpha=0.85, label=r'Variance Explained ($R^2$)')
    ax2 = ax.twinx()
    b2 = ax2.bar(x_pos + w/2, rmse_vals, width=w, color='#e41a1c', alpha=0.85, label='Root Mean Squared Error (RMSE)')
    
    # Value labels properly bounded within limits
    for bar in b1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 0.015, f'{h:.3f}', 
                ha='center', va='bottom', fontsize=10, fontweight='bold', color='#377eb8')
    for bar in b2:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., h + 0.02, f'{h:.3f}', 
                 ha='center', va='bottom', fontsize=10, fontweight='bold', color='#e41a1c')

    ax.set_xticks(x_pos)
    ax.set_xticklabels(models, fontsize=11, fontweight='bold')
    ax.set_ylabel(r'Goodness-of-Fit $R^2$', fontsize=11, fontweight='bold', color='#377eb8')
    ax2.set_ylabel('RMSE (kcal/mol)', fontsize=11, fontweight='bold', color='#e41a1c')
    ax.set_ylim(0, 0.95)
    ax2.set_ylim(0, 1.65)
    ax2.grid(False) # avoid overlapping grids
    
    # Annotation box for physical thermal scale RT (placed cleanly in open upper region)
    ax.text(0.50, 0.83, 
            f"Thermodynamic Model Support:\n"
            f"• Fitted Softplus $\\tau = {popt_sp[2]:.2f}$ kcal/mol\n"
            f"• Thermal Energy $RT_{{25^\\circ\\text{{C}}}} = 0.59$ kcal/mol\n"
            f"• $\\Delta\\text{{AIC}} = {m_sp['AIC'] - m_hf['AIC']:.0f}$ (Decisive Soft Floor)",
            transform=ax.transAxes, ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#ffffea', edgecolor='#bbbbbb', alpha=0.95),
            fontsize=9.5)
    
    ax.set_title('(D) Quantitative Model Performance & Thermal Energy Match', fontsize=12, fontweight='bold')

    # Use constrained/tight layout cleanly with subplots_adjust to guarantee no clipped labels
    plt.tight_layout()
    fig_path3 = os.path.join(out_fig_dir, 'fig3_mathematical_floor_models.png')
    fig_path3_v2 = os.path.join(out_fig_dir, 'fig3_mathematical_floor_models_v2.png')
    fig_path3_brain = os.path.join(brain_fig_dir, 'fig3_mathematical_floor_models.png')
    fig_path3_brain_v2 = os.path.join(brain_fig_dir, 'fig3_mathematical_floor_models_v2.png')
    plt.savefig(fig_path3, dpi=300, bbox_inches='tight')
    plt.savefig(fig_path3_v2, dpi=300, bbox_inches='tight')
    plt.savefig(fig_path3_brain, dpi=300, bbox_inches='tight')
    plt.savefig(fig_path3_brain_v2, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved pristine Figure 3 to {fig_path3_brain} and {fig_path3_brain_v2}")

if __name__ == '__main__':
    main()
