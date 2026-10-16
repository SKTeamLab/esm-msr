import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def main():
    data_path = '/home/sareeves/software/esm-msr-release/data/tsuboyama/Tsuboyama2023_Dataset2_Dataset3_20230416.csv'
    out_fig_dir = 'results/figures'
    out_tab_dir = 'results/tables'
    brain_fig_dir = '/home/sareeves/.gemini/antigravity/brain/73260da3-d1b9-4bbc-ba76-78297e940788/figures'
    os.makedirs(out_fig_dir, exist_ok=True)
    os.makedirs(out_tab_dir, exist_ok=True)
    os.makedirs(brain_fig_dir, exist_ok=True)

    print("Loading data for per-library floor analysis...")
    cols = ['name', 'mut_type', 'WT_name', 'WT_cluster', 'deltaG', 'deltaG_95CI', 
            'deltaG_t', 'deltaG_c', 'log10_K50_t', 'log10_K50unfolded_t',
            'log10_K50_c', 'log10_K50unfolded_c', 'dG_ML', 'ddG_ML', 'pair_name']
    df = pd.read_csv(data_path, usecols=cols, low_memory=False)

    # WT stabilities
    wt_rows = df[df['mut_type'] == 'wt']
    wt_stab = wt_rows.groupby('WT_name')['deltaG'].mean().rename('wt_deltaG')

    # Aggregate per library
    def compute_library_metrics(group):
        n = len(group)
        dG = group['deltaG'].dropna()
        ci = group['deltaG_95CI'].dropna()
        if len(dG) < 100:
            return None
        
        # Stability percentiles
        p01 = np.percentile(dG, 1)
        p05 = np.percentile(dG, 5)
        p10 = np.percentile(dG, 10)
        p25 = np.percentile(dG, 25)
        p50 = np.percentile(dG, 50)
        
        # Fractions in sub-floor brackets
        frac_lt_neg1 = (dG < -1.0).mean()
        frac_neg1_to_neg05 = ((dG >= -1.0) & (dG < -0.5)).mean()
        frac_neg05_to_0 = ((dG >= -0.5) & (dG < 0.0)).mean()
        frac_0_to_05 = ((dG >= 0.0) & (dG < 0.5)).mean()
        frac_lt_0 = (dG < 0.0).mean()
        
        # Unfolded K50 baselines
        unfolded_t = group['log10_K50unfolded_t'].median()
        unfolded_c = group['log10_K50unfolded_c'].median()
        
        # CI explosion threshold: lowest dG where rolling median CI <= 0.5 kcal/mol
        # Sort by dG and compute rolling median CI
        sorted_indices = np.argsort(group['deltaG'].values)
        sorted_dG = group['deltaG'].values[sorted_indices]
        sorted_ci = group['deltaG_95CI'].values[sorted_indices]
        
        # Bin in windows of 0.2 kcal/mol from -2 to 3
        test_bins = np.linspace(-2.0, 3.0, 26)
        crit_dG = np.nan
        for b_low, b_high in zip(test_bins[:-1], test_bins[1:]):
            mask = (sorted_dG >= b_low) & (sorted_dG < b_high)
            if mask.sum() >= 10:
                med_ci = np.median(sorted_ci[mask])
                if med_ci <= 0.5 and np.isnan(crit_dG):
                    crit_dG = (b_low + b_high) / 2.0
                    break

        return pd.Series({
            'total_variants': n,
            'p01_dG': p01,
            'p05_dG': p05,
            'p10_dG': p10,
            'p25_dG': p25,
            'median_dG': p50,
            'frac_lt_neg1': frac_lt_neg1,
            'frac_neg1_to_neg05': frac_neg1_to_neg05,
            'frac_neg05_to_0': frac_neg05_to_0,
            'frac_0_to_05': frac_0_to_05,
            'frac_lt_0': frac_lt_0,
            'unfolded_t_median': unfolded_t,
            'unfolded_c_median': unfolded_c,
            'crit_dG_ci05': crit_dG
        })

    lib_stats = df.groupby('WT_name').apply(compute_library_metrics).dropna(how='all')
    lib_stats = lib_stats.join(wt_stab, how='inner')
    lib_stats.to_csv(os.path.join(out_tab_dir, 'per_library_floor_metrics.csv'))
    print(f"Computed metrics for {len(lib_stats)} libraries.")
    print("Library statistics summary:\n", lib_stats[['wt_deltaG', 'p05_dG', 'frac_neg05_to_0', 'frac_lt_0', 'crit_dG_ci05']].describe())

    # Correlations
    corr_wt_p05 = stats.spearmanr(lib_stats['wt_deltaG'], lib_stats['p05_dG'])
    corr_unfold_t_p05 = stats.spearmanr(lib_stats['unfolded_t_median'], lib_stats['p05_dG'])
    corr_wt_frac0 = stats.spearmanr(lib_stats['wt_deltaG'], lib_stats['frac_lt_0'])
    print(f"Spearman r(WT stability, 5th percentile dG): {corr_wt_p05.statistic:.3f} (p={corr_wt_p05.pvalue:.2e})")
    print(f"Spearman r(Unfolded K50 trypsin, 5th percentile dG): {corr_unfold_t_p05.statistic:.3f} (p={corr_unfold_t_p05.pvalue:.2e})")

    # Select 6 diverse, high-count libraries to illustrate floor heterogeneity
    # We want libraries with different WT stability, floor penetration, and pile-up
    rep_wts = ['3DKM.pdb', '1Y0M.pdb', '2KGT.pdb', '1QP2.pdb', '2LGW.pdb', '2MXD.pdb']
    rep_labels = {
        '3DKM.pdb': '3DKM (Extensive Sub-Floor, WT=2.2)',
        '1Y0M.pdb': '1Y0M (Sharp [-0.5, 0] Pile-Up, WT=2.8)',
        '2KGT.pdb': '2KGT (Intermediate Soft Floor, WT=3.1)',
        '1QP2.pdb': '1QP2 (Strong Negative Tail, WT=2.7)',
        '2LGW.pdb': '2LGW (Resistant Floor, WT=3.9)',
        '2MXD.pdb': '2MXD (High Stability, WT=4.6)'
    }

    # Create Publication Figure 2: Per-Library Floor Heterogeneity
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Panel A: Distribution of library 5th percentiles (shows floor varies across libraries!)
    ax = axes[0, 0]
    sns.histplot(lib_stats['p05_dG'], bins=30, kde=True, color='#2b5c8f', ax=ax)
    ax.axvline(x=0.0, color='blue', linestyle=':', lw=1.5, label=r'$\Delta G = 0$ kcal/mol')
    ax.axvline(x=-0.5, color='orange', linestyle='--', lw=1.5, label=r'$\Delta G = -0.5$ kcal/mol')
    ax.axvline(x=-1.0, color='black', linestyle='--', lw=1.5, label=r'$\Delta G = -1.0$ (dG_ML cutoff)')
    ax.set_xlabel(r'Library 5th Percentile $\Delta G$ (kcal/mol)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Libraries (Domains)', fontsize=12, fontweight='bold')
    ax.set_title('(A) Effective Floor Varies Widely Across Libraries', fontsize=13, fontweight='bold')
    ax.legend(frameon=True, facecolor='white', loc='upper right')

    # Panel B: Fraction of variants in [-0.5, 0] Pile-Up vs WT stability
    ax = axes[0, 1]
    scatter = ax.scatter(lib_stats['wt_deltaG'], lib_stats['frac_neg05_to_0'] * 100, 
                         c=lib_stats['unfolded_t_median'], cmap='coolwarm', s=45, alpha=0.8, edgecolors='none')
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label(r'Median $\log_{10} K_{50,\text{unfolded}}$ (Trypsin)', fontsize=10)
    ax.set_xlabel(r'Wild-Type Stability $\Delta G_{\text{WT}}$ (kcal/mol)', fontsize=12, fontweight='bold')
    ax.set_ylabel(r'% of Variants in $[-0.5, 0.0]$ Pile-Up Zone', fontsize=12, fontweight='bold')
    ax.set_title(r'(B) Pile-Up Intensity vs $\Delta G_{\text{WT}}$ and Unfolded Baseline', fontsize=13, fontweight='bold')

    # Highlight the 6 representative libraries
    for wt in rep_wts:
        if wt in lib_stats.index:
            row = lib_stats.loc[wt]
            ax.scatter(row['wt_deltaG'], row['frac_neg05_to_0'] * 100, color='black', s=100, marker='*', zorder=5)
            ax.text(row['wt_deltaG'] + 0.1, row['frac_neg05_to_0'] * 100 + 0.5, wt.split('.')[0], 
                    fontsize=9, fontweight='bold', zorder=6)

    # Panel C: ECDF of deltaG for the 6 representative libraries
    ax = axes[1, 0]
    colors = sns.color_palette('tab10', len(rep_wts))
    for i, wt in enumerate(rep_wts):
        sub_dG = df.loc[df['WT_name'] == wt, 'deltaG'].dropna().sort_values()
        y = np.linspace(0, 1, len(sub_dG))
        ax.plot(sub_dG, y, label=rep_labels.get(wt, wt), color=colors[i], lw=2.2)
    ax.axvline(x=0.0, color='blue', linestyle=':', lw=1.2, label='dG = 0')
    ax.axvline(x=-1.0, color='black', linestyle='--', lw=1.2, label='dG = -1 cutoff')
    ax.axvspan(-0.5, 0.0, color='orange', alpha=0.12, label='Pile-up [-0.5, 0]')
    ax.set_xlim(-2.5, 5.0)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel(r'Measured Stability $\Delta G$ (kcal/mol)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Empirical Cumulative Distribution (ECDF)', fontsize=12, fontweight='bold')
    ax.set_title('(C) Contrasting Library CDFs: Floor Shape & Penetration', fontsize=13, fontweight='bold')
    ax.legend(frameon=True, facecolor='white', loc='lower right', fontsize=8.5)

    # Panel D: Density curves near floor for the 6 representative libraries
    ax = axes[1, 1]
    for i, wt in enumerate(rep_wts):
        sub_dG = df.loc[(df['WT_name'] == wt) & (df['deltaG'] > -3) & (df['deltaG'] < 4), 'deltaG'].dropna()
        sns.kdeplot(sub_dG, ax=ax, label=wt.split('.')[0], color=colors[i], lw=2.2, bw_adjust=0.7)
    ax.axvline(x=0.0, color='blue', linestyle=':', lw=1.2)
    ax.axvline(x=-1.0, color='black', linestyle='--', lw=1.2)
    ax.axvspan(-0.5, 0.0, color='orange', alpha=0.15)
    ax.set_xlim(-2.0, 3.0)
    ax.set_xlabel(r'Measured Stability $\Delta G$ (kcal/mol)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Density', fontsize=12, fontweight='bold')
    ax.set_title(r'(D) Floor Density Profiles: Local Pile-Up at $\Delta G \sim 0$', fontsize=13, fontweight='bold')
    ax.legend(frameon=True, facecolor='white', loc='upper right', fontsize=9)

    plt.tight_layout()
    fig_path2 = os.path.join(out_fig_dir, 'fig2_per_library_floor_variation.png')
    fig_path2_brain = os.path.join(brain_fig_dir, 'fig2_per_library_floor_variation.png')
    plt.savefig(fig_path2, dpi=300)
    plt.savefig(fig_path2_brain, dpi=300)
    plt.close()
    print(f"Saved Figure 2 to {fig_path2} and {fig_path2_brain}")

if __name__ == '__main__':
    main()
