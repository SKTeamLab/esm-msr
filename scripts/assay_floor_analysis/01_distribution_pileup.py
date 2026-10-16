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

    print("Loading data...")
    cols = ['name', 'mut_type', 'WT_name', 'WT_cluster', 'deltaG', 'deltaG_95CI', 
            'deltaG_t', 'deltaG_c', 'log10_K50_t', 'log10_K50unfolded_t',
            'log10_K50_c', 'log10_K50unfolded_c', 'dG_ML', 'ddG_ML', 'pair_name']
    df = pd.read_csv(data_path, usecols=cols, low_memory=False)
    print(f"Loaded {len(df)} rows.")

    # Categorize variants
    is_wt = df['mut_type'] == 'wt'
    is_double = df['pair_name'].notna() & df['mut_type'].str.contains(':', na=False)
    is_single = (~is_wt) & (~is_double)
    
    df['variant_class'] = 'other'
    df.loc[is_wt, 'variant_class'] = 'WT'
    df.loc[is_single, 'variant_class'] = 'Single'
    df.loc[is_double, 'variant_class'] = 'Double'

    # 1. Binned counts analysis around floor (-2 to +2 in fine bins)
    fine_bins = np.linspace(-2.5, 2.5, 101)
    df_window = df[(df['deltaG'] >= -2.5) & (df['deltaG'] <= 2.5)].copy()
    
    # Save fine bin counts
    hist_all, bin_edges = np.histogram(df['deltaG'], bins=fine_bins)
    hist_singles, _ = np.histogram(df.loc[is_single, 'deltaG'], bins=fine_bins)
    hist_doubles, _ = np.histogram(df.loc[is_double, 'deltaG'], bins=fine_bins)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    bin_df = pd.DataFrame({
        'bin_center': bin_centers,
        'count_all': hist_all,
        'count_singles': hist_singles,
        'count_doubles': hist_doubles
    })
    bin_df.to_csv(os.path.join(out_tab_dir, 'floor_fine_histogram.csv'), index=False)

    # Calculate key stability brackets
    brackets = [
        ('<-1.0 (censored in dG_ML)', df['deltaG'] < -1.0),
        ('-1.0 to -0.5', (df['deltaG'] >= -1.0) & (df['deltaG'] < -0.5)),
        ('-0.5 to 0.0 (pile-up zone)', (df['deltaG'] >= -0.5) & (df['deltaG'] < 0.0)),
        ('0.0 to 0.5', (df['deltaG'] >= 0.0) & (df['deltaG'] < 0.5)),
        ('0.5 to 1.0', (df['deltaG'] >= 0.5) & (df['deltaG'] < 1.0)),
        ('> 1.0', df['deltaG'] >= 1.0)
    ]
    bracket_rows = []
    for label, mask in brackets:
        bracket_rows.append({
            'bracket': label,
            'count_all': mask.sum(),
            'pct_all': mask.mean() * 100,
            'count_singles': (mask & is_single).sum(),
            'pct_singles': (mask & is_single).sum() / is_single.sum() * 100,
            'count_doubles': (mask & is_double).sum(),
            'pct_doubles': (mask & is_double).sum() / is_double.sum() * 100,
        })
    bracket_summary = pd.DataFrame(bracket_rows)
    bracket_summary.to_csv(os.path.join(out_tab_dir, 'floor_bracket_summary.csv'), index=False)
    print("Bracket Summary:\n", bracket_summary)

    # 2. CI Explosion Analysis as function of deltaG
    ci_bins = np.linspace(-3.0, 5.0, 81)
    df['dG_bin'] = pd.cut(df['deltaG'], bins=ci_bins)
    ci_stats = df.groupby('dG_bin', observed=False).agg(
        dG_mean=('deltaG', 'mean'),
        ci_median=('deltaG_95CI', 'median'),
        ci_p25=('deltaG_95CI', lambda x: x.quantile(0.25)),
        ci_p75=('deltaG_95CI', lambda x: x.quantile(0.75)),
        ci_p90=('deltaG_95CI', lambda x: x.quantile(0.90)),
        count=('deltaG', 'count')
    ).dropna().reset_index()
    ci_stats.to_csv(os.path.join(out_tab_dir, 'ci_vs_deltaG.csv'), index=False)

    # 3. Create Publication Figure 1: Comprehensive Floor Architecture
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # Panel A: Overall density & censoring cliff
    ax = axes[0, 0]
    sns.kdeplot(df.loc[df['deltaG'] > -4, 'deltaG'], ax=ax, label='All variants (deltaG)', color='#1f77b4', lw=2)
    sns.kdeplot(df.loc[is_single & (df['deltaG'] > -4), 'deltaG'], ax=ax, label='Singles', color='#2ca02c', lw=2)
    sns.kdeplot(df.loc[is_double & (df['deltaG'] > -4), 'deltaG'], ax=ax, label='Doubles', color='#d62728', lw=2)
    ax.axvline(x=-1.0, color='black', linestyle='--', lw=1.5, label='dG_ML cutoff (<-1)')
    ax.axvline(x=0.0, color='gray', linestyle=':', lw=1.5, label='Thermodynamic zero (dG=0)')
    ax.set_xlim(-4, 6)
    ax.set_xlabel(r'Measured Stability $\Delta G$ (kcal/mol)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Density', fontsize=12, fontweight='bold')
    ax.set_title('(A) Global Stability Distribution & Cutoff Thresholds', fontsize=13, fontweight='bold')
    ax.legend(frameon=True, facecolor='white', loc='upper right')

    # Panel B: High-resolution zoom on the Floor Region [-2.0, +2.0]
    ax = axes[0, 1]
    w = bin_centers[1] - bin_centers[0]
    ax.bar(bin_centers - w/4, hist_singles, width=w/2, color='#2ca02c', alpha=0.7, label='Singles')
    ax.bar(bin_centers + w/4, hist_doubles, width=w/2, color='#d62728', alpha=0.7, label='Doubles')
    ax.axvspan(-0.5, 0.0, color='orange', alpha=0.2, label='Pile-up Zone [-0.5, 0]')
    ax.axvline(x=-1.0, color='black', linestyle='--', lw=1.5, label='<-1 Cutoff')
    ax.axvline(x=0.0, color='blue', linestyle=':', lw=1.5, label='Folding Transition (dG=0)')
    ax.set_xlim(-2.0, 2.0)
    ax.set_xlabel(r'$\Delta G$ (kcal/mol)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Variant Count per 0.05 kcal/mol Bin', fontsize=12, fontweight='bold')
    ax.set_title(r'(B) Fine-Grained Distribution Near Floor: Pile-Up at $[-0.5, 0]$', fontsize=13, fontweight='bold')
    ax.legend(frameon=True, facecolor='white', loc='upper left')

    # Panel C: Measurement Uncertainty (95% CI) Explosion Below dG=0
    ax = axes[1, 0]
    ax.plot(ci_stats['dG_mean'], ci_stats['ci_median'], color='#800080', lw=2.5, label='Median 95% CI')
    ax.fill_between(ci_stats['dG_mean'], ci_stats['ci_p25'], ci_stats['ci_p75'], color='#800080', alpha=0.25, label='IQR (25-75th %ile)')
    ax.plot(ci_stats['dG_mean'], ci_stats['ci_p90'], color='#800080', linestyle='--', lw=1.5, label='90th %ile CI')
    ax.axvline(x=0.0, color='blue', linestyle=':', lw=1.5, label=r'$\Delta G = 0$')
    ax.axvline(x=-1.0, color='black', linestyle='--', lw=1.5, label=r'$\Delta G = -1$ cutoff')
    ax.axhline(y=0.5, color='red', linestyle='-.', lw=1.2, label='CI = 0.5 kcal/mol threshold')
    ax.set_xlim(-2.5, 4.5)
    ax.set_ylim(0, 3.5)
    ax.set_xlabel(r'Measured Stability $\Delta G$ (kcal/mol)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Confidence Interval Width (kcal/mol)', fontsize=12, fontweight='bold')
    ax.set_title('(C) Soft Floor: Uncertainty Explosion Below 0 kcal/mol', fontsize=13, fontweight='bold')
    ax.legend(frameon=True, facecolor='white', loc='upper right')

    # Panel D: Protease Sensitivity Ratio log10(K50/K50_unfolded) vs deltaG
    ax = axes[1, 1]
    # Sample 10000 points to keep plot clean
    sample_df = df.sample(n=min(15000, len(df)), random_state=42).copy()
    diff_t = sample_df['log10_K50_t'] - sample_df['log10_K50unfolded_t']
    scatter = ax.scatter(sample_df['deltaG'], diff_t, c=sample_df['deltaG_95CI'].clip(0, 2), 
                         cmap='viridis', s=8, alpha=0.5)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label(r'$\Delta G$ 95% CI (capped at 2.0)', fontsize=10)
    ax.axhline(y=0.0, color='black', linestyle='--', lw=1.2, label=r'$K_{50} = K_{50,U}$ (Digestion Floor)')
    ax.axhline(y=np.log10(2), color='blue', linestyle=':', lw=1.2, label=r'$K_{50} = 2 K_{50,U}$ ($\Delta G = 0$)')
    ax.axvline(x=0.0, color='blue', linestyle=':', lw=1.2)
    ax.axvline(x=-1.0, color='black', linestyle='--', lw=1.2)
    ax.set_xlim(-2.5, 4.5)
    ax.set_ylim(-0.8, 3.0)
    ax.set_xlabel(r'Reported $\Delta G$ (kcal/mol)', fontsize=12, fontweight='bold')
    ax.set_ylabel(r'$\log_{10}(K_{50,\text{trypsin}} / K_{50,\text{unfolded}})$', fontsize=12, fontweight='bold')
    ax.set_title('(D) Physical Digestion Ratio Asymptote & Noise Spread', fontsize=13, fontweight='bold')
    ax.legend(frameon=True, facecolor='white', loc='lower right')

    plt.tight_layout()
    fig_path1 = os.path.join(out_fig_dir, 'fig1_assay_floor_overview.png')
    fig_path1_brain = os.path.join(brain_fig_dir, 'fig1_assay_floor_overview.png')
    plt.savefig(fig_path1, dpi=300)
    plt.savefig(fig_path1_brain, dpi=300)
    plt.close()
    print(f"Saved Figure 1 to {fig_path1} and {fig_path1_brain}")

if __name__ == '__main__':
    main()
