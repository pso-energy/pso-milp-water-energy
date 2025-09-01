import matplotlib.pyplot as plt
import numpy as np
from .plot_style import (
    setup_plot_style,
    get_color_palette,
    save_figure,
    format_axes,
    setup_bar_plot,
)


# ---------------------------
# Dati
# ---------------------------
def plot_metrics_hardcoded():
    # Setup the global plot style
    setup_plot_style()

    scenarios = ["Reserve-DSL-Only", "DES-NoOptDispatch", "FULL"]
    wd_multipliers = ["x1", "x1.5", "x2"]

    f_obj_values = [
        [16.17, 17.00, 17.94],  # Reserve-DSL-Only
        [10.20, 11.49, 13.06],  # DES-NoOptDispatch
        [8.66, 9.56, 10.44],  # FULL
    ]

    res_share_values = [
        [41.30, 40.93, 38.65],  # Reserve-DSL-Only
        [85.08, 84.67, 82.86],  # DES-NoOptDispatch
        [86.64, 83.11, 81.49],  # FULL
    ]

    f_obj = np.array(f_obj_values).T
    res_share = np.array(res_share_values).T

    # Use consistent color palette
    colors = get_color_palette(n_colors=3, palette_type="scenario")

    x = np.arange(len(scenarios))
    bar_width = 0.25

    # ---------------------------
    # Creiamo 2 subplot semplici senza broken axis
    # ---------------------------
    fig, (ax_fobj, ax_res) = plt.subplots(1, 2, figsize=(14, 5))

    # ---------------------------
    # Grafico f_obj (sinistra) - range 7.5 a 20
    # ---------------------------
    for i in range(len(wd_multipliers)):
        ax_fobj.bar(
            x + i * bar_width - bar_width,
            f_obj[i],
            width=bar_width,
            color=colors[i],
            label=wd_multipliers[i],
            edgecolor="black",
            linewidth=0.8,
        )

    ax_fobj.set_ylim(7.5, 20)  # Parto direttamente da 7.5
    format_axes(ax_fobj, ylabel=r"$\mathbf{f_{obj}}$ (M€/year)", grid=True)
    ax_fobj.set_xticks(x)
    ax_fobj.set_xticklabels(scenarios)
    ax_fobj.grid(axis="x", visible=False)

    # ---------------------------
    # Grafico RES Share (destra) - range 35 a 100
    # ---------------------------
    for i in range(len(wd_multipliers)):
        ax_res.bar(
            x + i * bar_width - bar_width,
            res_share[i],
            width=bar_width,
            color=colors[i],
            edgecolor="black",
            linewidth=0.8,
        )

    ax_res.set_ylim(35, 90)  # Parto direttamente da 35
    format_axes(ax_res, ylabel="RES Share (%)", grid=True)
    ax_res.set_xticks(x)
    ax_res.set_xticklabels(scenarios)
    ax_res.grid(axis="x", visible=False)

    # ---------------------------
    # Legenda unica con stile migliorato
    # ---------------------------
    handles, labels = ax_fobj.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        title="Water demand multiplier",
        title_fontsize=16,
        fontsize=14,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.05),
        ncol=3,
        frameon=True,
        fancybox=True,
        shadow=False,
        framealpha=0.9,
        edgecolor="black",
    )

    plt.tight_layout()
    plt.subplots_adjust(top=0.85)  # Faccio più spazio per la legenda
    save_figure(fig, "plots/metrics_plot.png")
