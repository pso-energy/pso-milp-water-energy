import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from .plot_style import (
    setup_plot_style,
    get_color_palette,
    save_figure,
    format_axes,
    TECH_COLORS,
)


def plot_stacked_installed_power():
    """
    Create a stacked bar chart showing installed capacity for FULL scenario
    with different water desalination multipliers (x1, x1.5, x2).
    Left chart: absolute values, Right chart: percentage change from x1.
    """
    # Setup the global plot style
    setup_plot_style()

    # Data from the original table (FULL scenarios only)
    # Keys are demand multipliers; values are capacities for each tech.
    data = {
        "x1": {"FOWT": 6.00, "BESS": 23.5, "DES": 0.75, "ELY": 3.08},  # ELY = ELY-FC
        "x1.5": {"FOWT": 6.00, "BESS": 23.0, "DES": 1.50, "ELY": 1.62},
        "x2": {"FOWT": 7.00, "BESS": 22.0, "DES": 1.50, "ELY": 3.00},
    }

    # Create DataFrame: index: x1, x1.5, x2; columns: FOWT, BESS, DES, ELY
    df = pd.DataFrame(data).T

    # Use consistent technology colors from our style guide
    colors = {
        "FOWT": TECH_COLORS["FOWT"],  # Blue for wind
        "BESS": TECH_COLORS["BESS"],  # Red for storage
        "DES": TECH_COLORS["DES"],  # Green for desalination
        "ELY": TECH_COLORS["ELY"],  # Purple for electrolyzer
    }

    # Create figure with two subplots
    fig, axes = plt.subplots(
        1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [1, 1]}
    )

    # === LEFT CHART: Absolute capacities ===
    x = np.arange(len(df.index))
    bottom = np.zeros(len(df.index))

    for tech in df.columns:
        axes[0].bar(
            x,
            df[tech],
            bottom=bottom,
            label=tech,
            color=colors[tech],
            width=0.7,
            edgecolor="black",
            linewidth=0.8,
        )
        bottom += df[tech]

    format_axes(axes[0], ylabel="Installed Capacity (MW/MWh)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(df.index)
    # Rimuovo la legenda individuale
    # axes[0].legend(title="Technology", loc="upper left", frameon=True)

    # === RIGHT CHART: Percentage change vs x1 (stacked positive/negative) ===
    baseline = df.loc["x1"]
    pct_change = (df / baseline - 1.0) * 100.0
    pct_change = pct_change.drop("x1")  # show only x1.5 and x2 vs x1

    x2 = np.arange(len(pct_change.index))
    bottom_pos = np.zeros(len(pct_change.index))
    bottom_neg = np.zeros(len(pct_change.index))

    for tech in pct_change.columns:
        vals = pct_change[tech].values
        # Positive values stack upward, negatives stack downward
        bottoms = np.where(vals >= 0, bottom_pos, bottom_neg)
        axes[1].bar(
            x2,
            vals,
            width=0.7,
            bottom=bottoms,
            color=colors[tech],
            label=tech,
            edgecolor="black",
            linewidth=0.8,
        )
        bottom_pos += np.where(vals >= 0, vals, 0)
        bottom_neg += np.where(vals < 0, vals, 0)

    # Add horizontal line at 0%
    axes[1].axhline(0, color="black", linewidth=1.0, alpha=0.8)

    format_axes(axes[1], ylabel="Change w.r.t. x1 (%)")
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels(pct_change.index)
    # Rimuovo la legenda individuale
    # axes[1].legend(title="Technology", loc="upper left", frameon=True)

    # Aggiungo una legenda unica condivisa
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        fontsize=14,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=4,  # 4 colonne per le 4 tecnologie
        frameon=True,
        fancybox=True,
        shadow=False,
        framealpha=0.9,
        edgecolor="black",
    )

    plt.tight_layout()
    plt.subplots_adjust(top=0.88)  # Faccio spazio per la legenda in alto

    # Save using our consistent save function
    save_figure(fig, "plots/stacked_installed_power_full.png")

    print("Stacked installed power plot saved to stacked_installed_power_full.png")


if __name__ == "__main__":
    plot_stacked_installed_power()
