import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Patch
import matplotlib.gridspec as gridspec


def billions_formatter(x, pos):
    return f'£{x * 1e-9:.1f}Bn'

def make_figure_one():
    mpl.rcParams['font.family'] = 'Helvetica'
    df = pd.read_csv('../data/wrangled/df_clean_merged.csv')
    colors = ['#3288bd', '#d53e4f', '#fee08b']
    fig = plt.figure(figsize=(11, 7))
    outer_gs = gridspec.GridSpec(2, 3, figure=fig)
    nested_gs = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer_gs[0, 1])
    ax_top = fig.add_subplot(nested_gs[0])
    ax_bottom = fig.add_subplot(nested_gs[1])
    ax0 = fig.add_subplot(outer_gs[0, 0])
    ax1 = fig.add_subplot(outer_gs[0, 2])
    ax2 = fig.add_subplot(outer_gs[1, 0])
    ax3 = fig.add_subplot(outer_gs[1, 1])
    ax4 = fig.add_subplot(outer_gs[1, 2])

    min_val = df['Q10'].min()
    max_val = df['Q10'].max()
    bins = np.linspace(min_val, max_val, 11)
    sns.histplot(df[df['Q5'] == 'Female']['Q10'], bins=bins, edgecolor='k', alpha=1,
                 color=colors[0], ax=ax_top, stat='density', legend=True)
    sns.histplot(df[df['Q5'] == 'Male']['Q10'], bins=bins, edgecolor='k', alpha=1,
                 color=colors[1], ax=ax_bottom, stat='density', legend=True)

    sns.violinplot(data=df[(df['Q5'] == 'Male') | (df['Q5'] == 'Female')].sort_values('Q4_stage'),
                   palette=colors[0:2], linewidth=0.75, linecolor='k',
                   x="Q4_stage", y="Q10", hue="Q5", ax=ax0, split=True)

    sns.violinplot(data=df[(df['Q5'] == 'Male') | (df['Q5'] == 'Female')],
                   palette=colors[0:2], linewidth=0.75, linecolor='k',
                   x="Q8_stemshape", y="Q10", hue="Q5", ax=ax1, split=True)
    for line in ax0.lines:
        line.set_color('black')
    for line in ax1.lines:
        line.set_color('black')

    df_filtered = df[((df['Q10'].notnull()) & (df['ICS_GPA'].notnull()))]
    sns.regplot(data=df_filtered, x="ICS_GPA", y="Q10", scatter=False, ax=ax2,
                line_kws={'color': 'k',
                          'linewidth': 0.5,
                          'linestyle': '--'}, ci=99)
    if ax2.collections:
        ci_poly = ax2.collections[0]
        ci_poly.set_facecolor(colors[2])
        ci_poly.set_alpha(1)
        ci_poly.set_edgecolor((1, 1, 1, 1))
        ci_poly.set_zorder(1)
    sns.scatterplot(data=df_filtered, x="ICS_GPA", y="Q10", hue="is_redbrick",
                    edgecolor="black", ax=ax2, palette=colors[0:2])

    df_filtered = df[((df['Q10'].notnull()) & (df['tot_income'].notnull()))]
    sns.regplot(data=df_filtered, x="tot_income", y="Q10", scatter=False, ax=ax3,
                line_kws={'color': 'k',
                          'linewidth': 0.5,
                          'linestyle': '--'}, ci=99)
    if ax3.collections:
        ci_poly = ax3.collections[0]
        ci_poly.set_facecolor(colors[2])
        ci_poly.set_alpha(1)
        ci_poly.set_edgecolor((1, 1, 1, 1))
        ci_poly.set_zorder(1)
    sns.scatterplot(data=df_filtered, x="tot_income", y="Q10", hue="is_oxbridge",
                    edgecolor="black", ax=ax3, palette=colors[0:2])

    df_filtered = df[((df['Q10'].notnull()) & (df['fte'].notnull()))]
    sns.regplot(data=df_filtered, x="fte", y="Q10", scatter=False, ax=ax4,
                line_kws={'color': 'k',
                          'linewidth': 0.5,
                          'linestyle': '--'}, ci=99)
    if ax4.collections:
        ci_poly = ax4.collections[0]
        ci_poly.set_facecolor(colors[2])
        ci_poly.set_alpha(1)
        ci_poly.set_edgecolor((1, 1, 1, 1))
        ci_poly.set_zorder(1)
    sns.scatterplot(data=df_filtered, x="fte", y="Q10", hue="is_russell",
                    edgecolor="black", ax=ax4, palette=colors[0:2])

    ax0.set_title('a.', loc='left', fontsize=16)
    ax_top.set_title('b.', loc='left', fontsize=16)
    # ax_bottom.set_title('c.', loc='left', fontsize=16)
    ax1.set_title('c.', loc='left', fontsize=16)
    ax2.set_title('d.', loc='left', fontsize=16)
    ax3.set_title('e.', loc='left', fontsize=16)
    ax4.set_title('f.', loc='left', fontsize=16)

    for ax in [ax0, ax1, ax2, ax3, ax4]:
        ax.set_ylabel('Ideal Weight', labelpad=-5)
        ax.grid(which="both", linestyle='--', alpha=0.225)
    ax_top.set_ylabel('Density', labelpad=0)
    ax_bottom.set_ylabel('Density', labelpad=0)
    ax_top.set_ylim(0, 0.08)
    ax_bottom.set_ylim(0, 0.08)
    ax_top.set_xlim(-5, 100)
    ax_bottom.set_xlim(-5, 100)
    ax_top.set_xlabel('')
    ax_bottom.set_xlabel('')
    ax_top.grid(which="both", linestyle='--', alpha=0.225)
    ax_bottom.grid(which="both", linestyle='--', alpha=0.225)

    legend_elements1 = [Patch(facecolor=colors[0], edgecolor='k',
                              label=r'Female', alpha=1),
                        Patch(facecolor=colors[1], edgecolor='k',
                              label=r'Male', alpha=1)]
    ax0.legend(handles=legend_elements1, loc='upper right', frameon=True,
               fontsize=10, framealpha=1, facecolor='w',
               edgecolor=(0, 0, 0, 1)
               )
    ax1.legend(handles=legend_elements1, loc='upper right', frameon=True,
               fontsize=10, framealpha=1, facecolor='w',
               edgecolor=(0, 0, 0, 1)
               )
    legend_elements2 = [Patch(facecolor=colors[0], edgecolor='k',
                              label=r'Non-Redbrick', alpha=1),
                        Patch(facecolor=colors[1], edgecolor='k',
                              label=r'Redbrick', alpha=1)]
    ax2.legend(handles=legend_elements2, loc='upper left', frameon=True,
               fontsize=10, framealpha=1, facecolor='w',
               edgecolor=(0, 0, 0, 1)
               )
    legend_elements3 = [Patch(facecolor=colors[0], edgecolor='k',
                              label=r'Non-Oxbridge', alpha=1),
                        Patch(facecolor=colors[1], edgecolor='k',
                              label=r'Oxbridge', alpha=1)]
    ax3.legend(handles=legend_elements3, loc='upper right', frameon=True,
               fontsize=10, framealpha=1, facecolor='w',
               edgecolor=(0, 0, 0, 1)
               )

    legend_elements4 = [Patch(facecolor=colors[0], edgecolor='k',
                              label=r'Non-Russell', alpha=1),
                        Patch(facecolor=colors[1], edgecolor='k',
                              label=r'Russell', alpha=1)]
    ax4.legend(handles=legend_elements4, loc='upper right', frameon=True,
               fontsize=10, framealpha=1, facecolor='w',
               edgecolor=(0, 0, 0, 1)
               )

    legend_elements1 = [Patch(facecolor=colors[0], edgecolor='k',
                              label=r'Female', alpha=1)]
    ax_top.legend(handles=legend_elements1, loc='upper right', frameon=True,
                  fontsize=9, framealpha=1, facecolor='w',
                  edgecolor=(0, 0, 0, 1))

    legend_elements1 = [Patch(facecolor=colors[1], edgecolor='k',
                              label=r'Male', alpha=1)]
    ax_bottom.legend(handles=legend_elements1, loc='upper right', frameon=True,
                     fontsize=9, framealpha=1, facecolor='w',
                     edgecolor=(0, 0, 0, 1))

    ax2.set_xlabel('ICS GPA')
    ax3.set_xlabel('Total Income')
    ax4.set_xlabel('Full-time Employed')
    ax0.set_xlabel('')
    ax1.set_xlabel('')
    ax_top.yaxis.set_major_locator(ticker.MaxNLocator(nbins=2))
    ax_bottom.yaxis.set_major_locator(ticker.MaxNLocator(nbins=2))
    ax3.xaxis.set_major_formatter(ticker.FuncFormatter(billions_formatter))
    ax3.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
    plt.tight_layout()
    plt.savefig('../figures/who_wants_impact.pdf', bbox_inches='tight')