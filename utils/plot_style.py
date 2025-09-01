"""
Style configuration for all plots in the project.
Follows best practices for scientific publications with emphasis on:
- High readability for half-column layouts
- Consistent color schemes
- Clean, professional appearance
- No titles on plots (handled by captions in paper)
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import numpy as np

# Font and style configuration
FONT_SIZE_LARGE = 16  # Main labels
FONT_SIZE_MEDIUM = 14  # Tick labels, legend
FONT_SIZE_SMALL = 12  # Minor text
LINE_WIDTH = 2.5  # Line thickness
MARKER_SIZE = 8  # Marker size
DPI = 300  # High resolution for publications

# Color palette - Scientific and colorblind-friendly
COLORS = {
    "primary": "#2E86AB",  # Blue
    "secondary": "#A23B72",  # Purple-pink
    "accent1": "#F18F01",  # Orange
    "accent2": "#C73E1D",  # Red
    "accent3": "#6A994E",  # Green
    "neutral": "#6C757D",  # Gray
}

# Technology-specific colors (consistent across all plots)
# Palette Okabe-Ito
# Palette ColorBrewer Set2
# Palette "The Grand Budapest Hotel"
# Palette Monocromatica (Verde)
TECH_COLORS = {
    "PV": "#3A24FF",  # Giallo tenue, simboleggia l'energia solare.
    "FOWT": "#FFB84C",  # Arancione chiaro, una transizione verso il calore.
    "DES": "#FF6B6B",  # Rosso mattone, per evidenziare l'importanza dello storage.
    "BESS": "#4ECDC4",  # Verde acqua, un colore che evoca la stabilità.
    "ELY": "#1A936F",  # Verde scuro, per un senso di solidità e affidabilità.
}


# Scenario-specific colors (for bar charts)
SCENARIO_COLORS = ["#364B7E", "#2B7C78", "#79B37A"]  # Blue, Teal, Green


def setup_plot_style():
    """
    Set up the global matplotlib style for all plots.
    Call this function at the beginning of any plotting script.
    """
    # Use seaborn style as base
    plt.style.use("seaborn-v0_8-whitegrid")

    # Override with custom settings
    mpl.rcParams.update(
        {
            # Font settings
            "font.size": FONT_SIZE_MEDIUM,
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Arial",
                "DejaVu Sans",
                "Liberation Sans",
                "sans-serif",
            ],
            # Axes settings
            "axes.titlesize": FONT_SIZE_LARGE,
            "axes.labelsize": FONT_SIZE_LARGE,
            "axes.titleweight": "bold",
            "axes.labelweight": "bold",
            "axes.linewidth": 1.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            # Tick settings
            "xtick.labelsize": FONT_SIZE_MEDIUM,
            "ytick.labelsize": FONT_SIZE_MEDIUM,
            "xtick.major.size": 6,
            "ytick.major.size": 6,
            "xtick.minor.size": 4,
            "ytick.minor.size": 4,
            # Legend settings
            "legend.fontsize": FONT_SIZE_MEDIUM,
            "legend.frameon": True,
            "legend.fancybox": True,
            "legend.shadow": False,
            "legend.framealpha": 0.9,
            "legend.edgecolor": "black",
            "legend.facecolor": "white",
            # Line and marker settings
            "lines.linewidth": LINE_WIDTH,
            "lines.markersize": MARKER_SIZE,
            "lines.markeredgewidth": 1.0,
            # Figure settings
            "figure.figsize": (10, 6),  # Good for half-column layouts
            "figure.dpi": DPI,
            "figure.facecolor": "white",
            "figure.autolayout": True,
            # Save settings
            "savefig.dpi": DPI,
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
            "savefig.edgecolor": "none",
            "savefig.transparent": False,
        }
    )


def get_color_palette(n_colors=None, palette_type="qualitative"):
    """
    Get a consistent color palette for plots.

    Args:
        n_colors: Number of colors needed (if None, returns all available)
        palette_type: 'qualitative', 'tech', 'scenario'

    Returns:
        List of color codes
    """
    if palette_type == "tech":
        colors = list(TECH_COLORS.values())
    elif palette_type == "scenario":
        colors = SCENARIO_COLORS
    else:  # qualitative
        colors = list(COLORS.values())

    if n_colors is None:
        return colors

    # If we need more colors than available, create a extended palette
    if n_colors > len(colors):
        return sns.color_palette("husl", n_colors).as_hex()

    return colors[:n_colors]


def save_figure(fig, filename, **kwargs):
    """
    Save figure with consistent settings.

    Args:
        fig: matplotlib figure object
        filename: output filename
        **kwargs: additional arguments for savefig
    """
    default_kwargs = {
        "dpi": DPI,
        "bbox_inches": "tight",
        "facecolor": "white",
        "edgecolor": "none",
        "transparent": False,
    }
    default_kwargs.update(kwargs)

    fig.savefig(filename, **default_kwargs)
    plt.close(fig)


def format_axes(ax, xlabel=None, ylabel=None, title=None, grid=True):
    """
    Apply consistent formatting to axes.

    Args:
        ax: matplotlib axes object
        xlabel: x-axis label
        ylabel: y-axis label
        title: plot title (generally avoided for publications)
        grid: whether to show grid
    """
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=FONT_SIZE_LARGE, fontweight="bold")
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=FONT_SIZE_LARGE, fontweight="bold")
    if title:  # Generally not used for publication plots
        ax.set_title(title, fontsize=FONT_SIZE_LARGE, fontweight="bold")

    ax.tick_params(axis="both", which="major", labelsize=FONT_SIZE_MEDIUM)

    if grid:
        ax.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)
        ax.set_axisbelow(True)

    # Remove top and right spines for cleaner look
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.5)
    ax.spines["bottom"].set_linewidth(1.5)


def create_legend(handles, labels, ax=None, fig=None, loc="best", **kwargs):
    """
    Create a consistent legend.

    Args:
        handles: legend handles
        labels: legend labels
        ax: axes object (for axes legend)
        fig: figure object (for figure legend)
        loc: legend location
        **kwargs: additional legend arguments
    """
    default_kwargs = {
        "fontsize": FONT_SIZE_MEDIUM,
        "frameon": True,
        "fancybox": True,
        "shadow": False,
        "framealpha": 0.9,
        "edgecolor": "black",
        "facecolor": "white",
    }
    default_kwargs.update(kwargs)

    if fig is not None:
        return fig.legend(handles, labels, loc=loc, **default_kwargs)
    else:
        return ax.legend(handles, labels, loc=loc, **default_kwargs)


# Utility functions for common plot types
def setup_bar_plot(ax, x_labels, bar_width=0.25, colors=None):
    """Setup consistent bar plot formatting."""
    if colors is None:
        colors = get_color_palette(palette_type="scenario")

    format_axes(ax, grid=True)
    ax.grid(axis="x", visible=False)  # Only horizontal grid for bar plots

    return colors


def setup_line_plot(ax, colors=None):
    """Setup consistent line plot formatting."""
    if colors is None:
        colors = get_color_palette(palette_type="tech")

    format_axes(ax, grid=True)

    return colors


def setup_heatmap(ax, cbar_label="Value"):
    """Setup consistent heatmap formatting."""
    format_axes(ax, grid=False)

    # Style the colorbar
    cbar = ax.collections[0].colorbar
    if cbar:
        cbar.ax.set_ylabel(cbar_label, fontsize=FONT_SIZE_LARGE, fontweight="bold")
        cbar.ax.tick_params(labelsize=FONT_SIZE_MEDIUM)

    return cbar


def create_single_or_combined_figure(plot_type="combined", n_plots=2, figsize=None):
    """
    Create figure layout for either single or combined plots.
    
    Args:
        plot_type: 'single' or 'combined'
        n_plots: number of subplots for combined layout
        figsize: custom figure size (if None, uses defaults)
    
    Returns:
        fig, axes: matplotlib figure and axes objects
    """
    if plot_type == "single":
        if figsize is None:
            figsize = (8, 6)  # Single plot size
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        return fig, ax
    else:  # combined
        if figsize is None:
            if n_plots == 2:
                figsize = (14, 6)  # Two plots side by side
            elif n_plots == 3:
                figsize = (18, 6)  # Three plots side by side
            else:
                figsize = (10, 6)  # Default
        
        if n_plots == 2:
            fig, axes = plt.subplots(1, 2, figsize=figsize)
        elif n_plots == 3:
            fig, axes = plt.subplots(1, 3, figsize=figsize)
        else:
            fig, axes = plt.subplots(1, n_plots, figsize=figsize)
        
        return fig, axes


def save_single_and_combined_figures(fig_combined, axes_combined, plot_functions, 
                                   base_filename, plot_titles=None, **save_kwargs):
    """
    Save both combined plot and individual single plots.
    
    Args:
        fig_combined: combined figure object
        axes_combined: combined axes objects
        plot_functions: list of functions that recreate individual plots
        base_filename: base filename without extension
        plot_titles: list of titles for individual plots
        **save_kwargs: additional arguments for save_figure
    """
    # Save combined plot
    save_figure(fig_combined, f"{base_filename}_combined.png", **save_kwargs)
    
    # Create and save individual plots
    if plot_titles is None:
        plot_titles = [f"plot_{i+1}" for i in range(len(plot_functions))]
    
    for i, (plot_func, title) in enumerate(zip(plot_functions, plot_titles)):
        # Create single figure
        fig_single, ax_single = create_single_or_combined_figure("single")
        
        # Execute the plot function on the single axes
        plot_func(ax_single)
        
        # Save individual plot
        save_figure(fig_single, f"{base_filename}_{title}.png", **save_kwargs)


def setup_subplot_layout(axes, n_plots, shared_legend=True, legend_kwargs=None):
    """
    Setup consistent layout for multiple subplots.
    
    Args:
        axes: axes objects (single ax or array of axes)
        n_plots: number of subplots
        shared_legend: whether to create a shared legend
        legend_kwargs: custom legend parameters
    """
    # Handle single axes case
    if n_plots == 1:
        return axes
    
    # Ensure axes is iterable for multiple plots
    if not hasattr(axes, '__iter__'):
        axes = [axes]
    
    # Remove individual legends if shared legend is requested
    if shared_legend:
        for ax in axes:
            legend = ax.get_legend()
            if legend:
                legend.remove()
    
    return axes


def create_shared_legend(fig, axes, location="upper center", bbox_to_anchor=(0.5, 1.02), 
                        ncol=None, **legend_kwargs):
    """
    Create a shared legend for multiple subplots.
    
    Args:
        fig: figure object
        axes: axes objects
        location: legend location
        bbox_to_anchor: legend position
        ncol: number of columns (auto-calculated if None)
        **legend_kwargs: additional legend parameters
    """
    # Get legend handles and labels from first axes
    if hasattr(axes, '__iter__'):
        handles, labels = axes[0].get_legend_handles_labels()
    else:
        handles, labels = axes.get_legend_handles_labels()
    
    # Auto-calculate number of columns if not specified
    if ncol is None:
        ncol = min(len(labels), 4)  # Max 4 columns
    
    # Default legend parameters
    default_kwargs = {
        "fontsize": FONT_SIZE_MEDIUM,
        "frameon": True,
        "fancybox": True,
        "shadow": False,
        "framealpha": 0.9,
        "edgecolor": "black",
        "facecolor": "white",
    }
    default_kwargs.update(legend_kwargs)
    
    # Create shared legend
    legend = fig.legend(
        handles, 
        labels,
        loc=location,
        bbox_to_anchor=bbox_to_anchor,
        ncol=ncol,
        **default_kwargs
    )
    
    return legend
