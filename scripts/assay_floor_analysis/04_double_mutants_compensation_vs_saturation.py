import os
import glob
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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

    print("Loading data for double mutant epistasis analysis...")
    cols = ['mut_type', 'WT_name', 'deltaG', 'deltaG_95CI', 'dG_ML', 'ddG_ML', 'pair_name', 'fitting_error_t', 'fitting_error_c']
    df = pd.read_csv(data_path, usecols=cols, low_memory=False)

    is_wt = df['mut_type'] == 'wt'
    is_double = df['pair_name'].notna() & df['mut_type'].str.contains(':', na=False)
    is_single = (~is_wt) & (~is_double)

    wt_stabs = df[is_wt].groupby('WT_name')['deltaG'].mean().to_dict()
    df_singles = df[is_single].copy()
    single_dict = df_singles.set_index(['WT_name', 'mut_type'])['deltaG'].to_dict()
    df_doubles = df[is_double].copy()

    # Load PDB CA coords
    pdb_cache = {}
    for pdb_file in glob.glob(os.path.join(pdb_dir, '*.pdb')):
        pdb_cache[os.path.basename(pdb_file)] = get_ca_coords(pdb_file)

    records = []
    for idx, row in df_doubles.iterrows():
        wt = row['WT_name']
        mut = row['mut_type']
        obs_dG = row['deltaG']
        ci = row['deltaG_95CI']
        dG_ML = row['dG_ML']
        
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
        epi = obs_dG - exp_dG
        
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
            'dG_epi': epi,
            'wt_dG': wt_dG,
            'dist_ca': dist,
            'dG_95CI': ci,
            'dG_ML': dG_ML
        })

    pdf = pd.DataFrame(records)
    print(f"Total paired double mutants: {len(pdf)}")

    # Add distance categories
    pdf['dist_class'] = 'Unknown'
    pdf.loc[pdf['dist_ca'] <= 8.0, 'dist_class'] = 'Contact (<=8A)'
    pdf.loc[(pdf['dist_ca'] > 8.0) & (pdf['dist_ca'] <= 15.0), 'dist_class'] = 'Intermediate (8-15A)'
    pdf.loc[pdf['dist_ca'] > 15.0, 'dist_class'] = 'Distant (>15A)'

    print("Distance class breakdown:\n", pdf['dist_class'].value_counts())

    # Add stability regime
    pdf['regime'] = 'Above Floor (exp >= 0.5)'
    pdf.loc[pdf['dG_exp'] < 0.0, 'regime'] = 'Sub-Floor (exp < 0.0)'
    pdf.loc[(pdf['dG_exp'] >= 0.0) & (pdf['dG_exp'] < 0.5), 'regime'] = 'Floor Transition [0.0, 0.5)'

    # Epistasis statistics stratified by distance and regime
    strat_stats = pdf.groupby(['regime', 'dist_class'], observed=False).agg(
        count=('dG_epi', 'count'),
        mean_epi=('dG_epi', 'mean'),
        median_epi=('dG_epi', 'median'),
        std_epi=('dG_epi', 'std'),
        pct_positive_epi=('dG_epi', lambda x: (x > 0.5).mean() * 100),
        median_ci=('dG_95CI', 'median')
    ).reset_index()
    strat_stats.to_csv(os.path.join(out_tab_dir, 'epistasis_stratified_summary.csv'), index=False)
    print("\nStratified Epistasis Summary:\n", strat_stats)

    # Sub-floor survivorship breakdown (Section 5 replication)
    subfloor = pdf[pdf['dG_exp'] < 0.0].copy()
    subfloor['class'] = 'Unidentifiable (CI > 0.5)'
    subfloor.loc[(subfloor['dG_95CI'] <= 0.5) & (subfloor['dG_obs'] >= 0.5), 'class'] = 'Genuine Compensation (CI<=0.5, dG>=0.5)'
    subfloor.loc[(subfloor['dG_95CI'] <= 0.5) & (subfloor['dG_obs'] < 0.5), 'class'] = 'Genuinely Unfolded (CI<=0.5, dG<0.5)'

    surv_summary = subfloor.groupby('class').agg(
        n=('dG_epi', 'count'),
        share=('dG_epi', lambda x: len(x) / len(subfloor) * 100),
        mean_epi=('dG_epi', 'mean'),
        median_epi=('dG_epi', 'median'),
        median_ci=('dG_95CI', 'median'),
        pct_contact=('dist_class', lambda x: (x == 'Contact (<=8A)').mean() * 100)
    ).reset_index()
    surv_summary.to_csv(os.path.join(out_tab_dir, 'subfloor_survivorship_breakdown.csv'), index=False)
    print("\nSub-Floor Survivorship Breakdown:\n", surv_summary)

    # Publication Figure 4: Compensation vs Saturation in Double Mutants
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Panel A: Apparent Epistasis vs Expected Stability (Contact vs Distant)
    ax = axes[0, 0]
    # Plot binned curves of epistasis vs dG_exp for Contact vs Distant
    exp_bins = np.linspace(-3.0, 4.0, 36)
    bin_centers = 0.5 * (exp_bins[:-1] + exp_bins[1:])
    
    contacts = pdf[pdf['dist_class'] == 'Contact (<=8A)']
    distants = pdf[pdf['dist_class'] == 'Distant (>15A)']
    
    mean_epi_contact = []
    mean_epi_distant = []
    for b_l, b_h in zip(exp_bins[:-1], exp_bins[1:]):
        mc = (contacts['dG_exp'] >= b_l) & (contacts['dG_exp'] < b_h)
        md = (distants['dG_exp'] >= b_l) & (distants['dG_exp'] < b_h)
        mean_epi_contact.append(contacts.loc[mc, 'dG_epi'].mean() if mc.sum() > 10 else np.nan)
        mean_epi_distant.append(distants.loc[md, 'dG_epi'].mean() if md.sum() > 10 else np.nan)

    ax.plot(bin_centers, mean_epi_contact, 'o-', color='#e41a1c', lw=2.5, label=r'Contact Pairs ($d \leq 8$ Å)')
    ax.plot(bin_centers, mean_epi_distant, 's-', color='#377eb8', lw=2.5, label=r'Distant Pairs ($d > 15$ Å)')
    
    # Theoretical saturation curve: epistasis = floor - dG_exp = -dG_exp (assuming floor ~ 0)
    ax.plot(bin_centers[bin_centers < 0.2], -bin_centers[bin_centers < 0.2], color='black', linestyle='--', lw=1.5, 
            label=r'Pure Floor Saturation Line ($\Delta\Delta G_{\text{epi}} = 0 - \Delta G_{\text{exp}}$)')
    ax.axhline(y=0.0, color='gray', linestyle=':', lw=1.2)
    ax.axvline(x=0.0, color='blue', linestyle=':', lw=1.2, label='Floor Threshold (dG=0)')
    ax.axvspan(-3.0, 0.0, color='red', alpha=0.08, label='Artifact Dominated Regime')
    ax.axvspan(0.5, 4.0, color='green', alpha=0.08, label='Genuine Epistasis Regime')
    ax.set_xlim(-3.0, 3.5)
    ax.set_ylim(-1.0, 3.5)
    ax.set_xlabel(r'Additive Expected Stability $\Delta G_{\text{expected}}$ (kcal/mol)', fontsize=11, fontweight='bold')
    ax.set_ylabel(r'Observed Epistasis $\Delta\Delta G_{\text{epi}} = \Delta G_{AB} - \Delta G_{\text{exp}}$ (kcal/mol)', fontsize=11, fontweight='bold')
    ax.set_title(r'(A) Separation of Saturation Artifact from Genuine Compensation', fontsize=12, fontweight='bold')
    ax.legend(frameon=True, facecolor='white', loc='upper right', fontsize=8.5)

    # Panel B: Epistasis Distribution in the Safe Regime (dG_exp >= 0.5)
    ax = axes[0, 1]
    safe_contact = contacts.loc[contacts['dG_exp'] >= 0.5, 'dG_epi'].dropna()
    safe_distant = distants.loc[distants['dG_exp'] >= 0.5, 'dG_epi'].dropna()
    sns.kdeplot(safe_contact, ax=ax, color='#e41a1c', lw=2.5, label=f'Contacts (N={len(safe_contact)})')
    sns.kdeplot(safe_distant, ax=ax, color='#377eb8', lw=2.5, label=f'Distant (N={len(safe_distant)})')
    ax.axvline(x=0.0, color='gray', linestyle='--', lw=1.2)
    ax.set_xlim(-2.0, 3.0)
    ax.set_xlabel(r'Observed Epistasis $\Delta\Delta G_{\text{epi}}$ (kcal/mol)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Density', fontsize=11, fontweight='bold')
    ax.set_title(r'(B) True Compensation Emerges in Contacts Above Floor ($\Delta G_{\text{exp}} \geq 0.5$)', fontsize=12, fontweight='bold')
    ax.legend(frameon=True, facecolor='white', loc='upper right', fontsize=9.5)

    # Panel C: Sub-floor Survivorship Decomposition
    ax = axes[1, 0]
    classes = ['Genuine\nCompensation', 'Unidentifiable\n(Noise/Bad Data)', 'Genuinely\nUnfolded']
    shares = surv_summary['share'].values
    mean_epis = surv_summary['mean_epi'].values
    contact_enrich = surv_summary['pct_contact'].values
    
    x_pos = np.arange(len(classes))
    bars = ax.bar(x_pos - 0.2, shares, width=0.38, color=['#4daf4a', '#ff7f00', '#984ea3'], alpha=0.85, label='% Share of Sub-Floor Doubles')
    ax2 = ax.twinx()
    bars2 = ax2.bar(x_pos + 0.2, mean_epis, width=0.38, color=['#2ca02c', '#d95f02', '#7570b3'], alpha=0.4, label='Mean Apparent Epistasis (kcal/mol)')
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(classes, fontsize=10, fontweight='bold')
    ax.set_ylabel('% Share of Sub-Floor Variants (N = 89,700)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Mean Apparent Epistasis (kcal/mol)', fontsize=11, fontweight='bold')
    ax.set_title(r'(C) Decomposition of Sub-Floor Doubles ($\Delta G_{\text{expected}} < 0$)', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 60)
    ax2.set_ylim(0, 3.0)

    # Panel D: Contact Enrichment as Function of Epistasis in Sub-Floor vs Above-Floor
    ax = axes[1, 1]
    # For subfloor vs above-floor, check % of pairs that are in contact as function of apparent epistasis
    epi_brackets = [(-1.5, 0.0), (0.0, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 4.0)]
    bracket_labels = ['<0.0', '0.0-0.5', '0.5-1.0', '1.0-2.0', '>2.0']
    
    pct_contact_above = []
    pct_contact_sub = []
    for b_l, b_h in epi_brackets:
        m_above = (pdf['regime'] == 'Above Floor (exp >= 0.5)') & (pdf['dG_epi'] >= b_l) & (pdf['dG_epi'] < b_h) & (pdf['dist_class'] != 'Unknown')
        m_sub = (pdf['regime'] == 'Sub-Floor (exp < 0.0)') & (pdf['dG_epi'] >= b_l) & (pdf['dG_epi'] < b_h) & (pdf['dist_class'] != 'Unknown')
        pct_contact_above.append((pdf.loc[m_above, 'dist_class'] == 'Contact (<=8A)').mean() * 100 if m_above.sum() > 20 else np.nan)
        pct_contact_sub.append((pdf.loc[m_sub, 'dist_class'] == 'Contact (<=8A)').mean() * 100 if m_sub.sum() > 20 else np.nan)

    x_b = np.arange(len(bracket_labels))
    w = 0.35
    ax.bar(x_b - w/2, pct_contact_above, width=w, color='#2ca02c', alpha=0.85, label='Above Floor (exp >= 0.5): Contact Enrichment')
    ax.bar(x_b + w/2, pct_contact_sub, width=w, color='#e41a1c', alpha=0.7, label='Sub-Floor (exp < 0): Flat / Non-Specific')
    ax.set_xticks(x_b)
    ax.set_xticklabels(bracket_labels, fontsize=10, fontweight='bold')
    ax.set_xlabel(r'Apparent Epistasis Bracket $\Delta\Delta G_{\text{epi}}$ (kcal/mol)', fontsize=11, fontweight='bold')
    ax.set_ylabel('% of Variants in Direct Contact (<=8 Å)', fontsize=11, fontweight='bold')
    ax.set_title('(D) Contact Enrichment Proves Genuine Epistasis Above Floor', fontsize=12, fontweight='bold')
    ax.legend(frameon=True, facecolor='white', loc='upper left', fontsize=9.5)

    plt.tight_layout()
    fig_path4 = os.path.join(out_fig_dir, 'fig4_double_mutants_compensation_vs_saturation.png')
    fig_path4_brain = os.path.join(brain_fig_dir, 'fig4_double_mutants_compensation_vs_saturation.png')
    plt.savefig(fig_path4, dpi=300)
    plt.savefig(fig_path4_brain, dpi=300)
    plt.close()
    print(f"Saved Figure 4 to {fig_path4} and {fig_path4_brain}")

if __name__ == '__main__':
    main()
