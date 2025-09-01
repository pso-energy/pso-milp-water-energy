import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from .plot_style import (
    setup_plot_style,
    save_figure,
    format_axes,
    TECH_COLORS,
    SCENARIO_COLORS,
)


def parse_particles_csv(csv_file_path: str) -> tuple:
    """
    Parse the particles_log.csv file to extract particle positions and fitness information.
    Returns a dictionary with particle histories and global best evolution for all iterations.
    """
    if not os.path.exists(csv_file_path):
        print(f"File {csv_file_path} not found")
        return {}, []

    df = pd.read_csv(csv_file_path)

    particles_data = {}
    global_best_history = []
    current_global_best = float("inf")
    current_best_position = None
    current_best_particle = None

    # Get all unique iterations
    all_iterations = sorted(df["iteration"].unique())

    # Process each iteration
    for iteration in all_iterations:
        iteration_data = df[df["iteration"] == iteration]

        # Check if any particle in this iteration improved the global best
        iteration_improved = False

        for _, row in iteration_data.iterrows():
            particle_id = int(row["particle"])
            fitness = float(row["fitness"])

            # Extract position vector (all columns except Iteration, ParticleID, Fitness)
            position_columns = [
                col
                for col in df.columns
                if col not in ["iteration", "particle", "fitness"]
            ]
            position = [row[col] for col in position_columns]

            # Store particle data
            if particle_id not in particles_data:
                particles_data[particle_id] = {
                    "positions": [],
                    "fitness": [],
                    "iterations": [],
                }

            particles_data[particle_id]["positions"].append(position)
            particles_data[particle_id]["fitness"].append(fitness)
            particles_data[particle_id]["iterations"].append(iteration)

            # Update global best if this is better and fitness is not inf
            if fitness != float("inf") and fitness < current_global_best:
                current_global_best = fitness
                current_best_position = position
                current_best_particle = particle_id
                iteration_improved = True

        # Always add an entry for this iteration (either new best or keep previous best)
        if current_best_position is not None:
            global_best_history.append(
                {
                    "iteration": iteration,
                    "fitness": current_global_best,
                    "position": current_best_position,
                    "particle": current_best_particle,
                    "improved": iteration_improved,
                }
            )

    return particles_data, global_best_history


def aggregate_position_by_technology(position: list) -> dict:
    """
    Aggregate the 10-dimensional position vector into 5 technology categories:
    Based on CSV column order: bess_0, bess_1, electrolyzer_0, electrolyzer_1,
    desalinator_0, desalinator_1, desalinator_2, desalinator_3, vres_0, vres_1

    - BESS: position[0] + position[1] (bess_0 + bess_1)
    - ELY: position[2] + position[3] (electrolyzer_0 + electrolyzer_1)
    - DES: position[4] + position[5] + position[6] + position[7] (desalinator_0-3)
    - PV: position[8] (vres_0 - assuming this is PV)
    - FOWT: position[9] (vres_1 - assuming this is Floating Offshore Wind)
    """
    return {
        "BESS": position[0] * 0.5 + position[1] * 0.5,
        "ELY": position[2] * 0.5 + position[3] * 0.5,
        "DES": position[4] * 0.25
        + position[5] * 0.5
        + position[6] * 0.75
        + position[7] * 1.00,
        "PV": position[8],
        "FOWT": position[9],
    }


def plot_best_particle_trajectory(simulation_path: str):
    """
    Plot 1: Trajectory of the best particle through all iterations.
    Shows how the best particle's position evolved over time - both absolute and normalized versions side by side.
    """
    # Setup the global plot style
    setup_plot_style()

    csv_file_path = os.path.join(simulation_path, "log", "particles_log.csv")
    plots_path = os.path.join(simulation_path, "plots")
    if not os.path.exists(plots_path):
        os.makedirs(plots_path)

    particles_data, global_best_history = parse_particles_csv(
        csv_file_path
    )  # Find the particle that achieved the overall best fitness
    best_fitness = float("inf")
    best_particle_num = None

    for particle_num, data in particles_data.items():
        if data["fitness"]:
            min_fitness = min(data["fitness"])
            if min_fitness < best_fitness:
                best_fitness = min_fitness
                best_particle_num = particle_num

    if best_particle_num is None:
        print("No best particle found")
        return

    # Get the positions of the best particle
    best_particle_positions = particles_data[best_particle_num]["positions"]

    # Aggregate positions by technology for each iteration
    tech_evolution = {"PV": [], "FOWT": [], "BESS": [], "DES": [], "ELY": []}

    for position in best_particle_positions:
        aggregated = aggregate_position_by_technology(position)
        for tech, value in aggregated.items():
            tech_evolution[tech].append(value)

    # Prepare normalized data
    tech_evolution_normalized = {}
    lowers = [0, 0, 0, 0, 0]
    uppers = [
        15 * 1.0,
        10 * 1.0,
        100 * 0.5 + 100 * 0.5,
        5 * 0.25 + 3 * 0.5 + 2 * 0.75 + 2 * 1.0,
        300 * 0.5 + 300 * 0.5,
    ]
    tech_names = ["PV", "FOWT", "BESS", "DES", "ELY"]
    bounds = dict(zip(tech_names, zip(lowers, uppers)))

    for tech, values in tech_evolution.items():
        if values:
            min_val, max_val = bounds[tech]
            if max_val > min_val:
                # Normalize to 0-100% range based on min/max
                normalized = [(v - min_val) / (max_val - min_val) * 100 for v in values]
            else:
                # If all values are the same, set to 50%
                normalized = [50] * len(values)
            tech_evolution_normalized[tech] = normalized

    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Use consistent technology colors
    colors = TECH_COLORS

    iterations = list(range(len(best_particle_positions)))

    # Left plot: Absolute values
    for tech, values in tech_evolution.items():
        if values:  # Only plot if we have data
            ax1.plot(
                iterations,
                values,
                marker="o",
                label=tech,
                color=colors[tech],
                linewidth=2.5,
                markersize=6,
                markeredgecolor="white",
                markeredgewidth=1.0,
            )

    format_axes(ax1, ylabel="Installed Capacity (MW)")
    ax1.legend(loc="best", frameon=True)

    # Right plot: Normalized values
    for tech, values in tech_evolution_normalized.items():
        if values:  # Only plot if we have data
            ax2.plot(
                iterations,
                values,
                marker="o",
                label=tech,
                color=colors[tech],
                linewidth=2.5,
                markersize=6,
                markeredgecolor="white",
                markeredgewidth=1.0,
            )

    format_axes(ax2, ylabel="Relative Change (0% = Min, 100% = Max)")
    ax2.set_ylim(0, 100)
    ax2.legend(loc="best", frameon=True)

    # Add a single legend for both plots, positioned at the top
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=5,
        fontsize=14,
        frameon=True,
        fancybox=True,
        shadow=False,
        framealpha=0.9,
        edgecolor="black",
    )

    # Remove individual legends
    ax1.legend().set_visible(False)
    ax2.legend().set_visible(False)

    plt.tight_layout()
    plt.subplots_adjust(top=0.88)  # Make room for the legend

    fig_path = os.path.join(plots_path, "best_particle_trajectory_combined.png")
    save_figure(fig, fig_path)

    print(f"Combined best particle trajectory plot saved to {fig_path}")


def plot_global_best_evolution(simulation_path: str):
    """
    Plot 2: Evolution of the global best position at each iteration.
    Shows how the overall best solution evolved over time - both absolute and normalized versions side by side.
    """
    # Setup the global plot style
    setup_plot_style()

    csv_file_path = os.path.join(simulation_path, "log", "particles_log.csv")
    plots_path = os.path.join(simulation_path, "plots")
    if not os.path.exists(plots_path):
        os.makedirs(plots_path)

    particles_data, global_best_history = parse_particles_csv(csv_file_path)

    if not global_best_history:
        print("No global best history found")
        return

    # Aggregate positions by technology for each global best update
    tech_evolution = {"PV": [], "FOWT": [], "BESS": [], "DES": [], "ELY": []}
    fitness_evolution = []

    for entry in global_best_history:
        aggregated = aggregate_position_by_technology(entry["position"])
        for tech, value in aggregated.items():
            tech_evolution[tech].append(value)
        fitness_evolution.append(entry["fitness"])

    # Prepare normalized data
    tech_evolution_normalized = {}
    lowers = [0, 0, 0, 0, 0]
    uppers = [
        15 * 1.0,
        10 * 1.0,
        100 * 0.5 + 100 * 0.5,
        5 * 0.25 + 3 * 0.5 + 2 * 0.75 + 2 * 1.0,
        300 * 0.5 + 300 * 0.5,
    ]
    tech_names = ["PV", "FOWT", "BESS", "DES", "ELY"]
    bounds = dict(zip(tech_names, zip(lowers, uppers)))

    for tech, values in tech_evolution.items():
        if values:
            min_val, max_val = bounds[tech]
            if max_val > min_val:
                # Normalize to 0-100% range based on min/max
                normalized = [(v - min_val) / (max_val - min_val) * 100 for v in values]
            else:
                # If all values are the same, set to 50%
                normalized = [50] * len(values)
            tech_evolution_normalized[tech] = normalized

    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Use consistent technology colors
    colors = TECH_COLORS

    iterations = [entry["iteration"] for entry in global_best_history]

    # Left plot: Absolute values
    for tech, values in tech_evolution.items():
        if values:  # Only plot if we have data
            ax1.plot(
                iterations,
                values,
                marker="s",
                label=tech,
                color=colors[tech],
                linewidth=2.5,
                markersize=6,
                markeredgecolor="white",
                markeredgewidth=1.0,
            )

    format_axes(ax1, ylabel="Installed Capacity (MW)")

    # Right plot: Normalized values
    for tech, values in tech_evolution_normalized.items():
        if values:  # Only plot if we have data
            ax2.plot(
                iterations,
                values,
                marker="s",
                label=tech,
                color=colors[tech],
                linewidth=2.5,
                markersize=6,
                markeredgecolor="white",
                markeredgewidth=1.0,
            )

    format_axes(ax2, ylabel="Installed Capacities \n (% of feasible bounds)")
    ax2.set_ylim(0, 100)

    # Add a single legend for both plots, positioned at the top
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=5,
        fontsize=14,
        frameon=True,
        fancybox=True,
        shadow=False,
        framealpha=0.9,
        edgecolor="black",
    )

    plt.tight_layout()
    plt.subplots_adjust(top=0.88)  # Make room for the legend

    fig_path = os.path.join(plots_path, "global_best_evolution_combined.png")
    save_figure(fig, fig_path)

    print(f"Combined global best evolution plot saved to {fig_path}")

    # Also create a secondary plot showing fitness evolution
    fig_fitness, ax_fitness = plt.subplots(figsize=(10, 6))
    iterations = [entry["iteration"] for entry in global_best_history]
    ax_fitness.plot(
        iterations,
        fitness_evolution,
        marker="o",
        color="#2E86AB",  # Primary color from style
        linewidth=2.5,
        markersize=6,
        markeredgecolor="white",
        markeredgewidth=1.0,
    )
    format_axes(ax_fitness, ylabel="Fitness Value")

    fitness_fig_path = os.path.join(plots_path, "global_best_fitness_evolution.png")
    save_figure(fig_fitness, fitness_fig_path)

    print(f"Global best fitness evolution plot saved to {fitness_fig_path}")


def plot_best_particle_trajectory_normalized(simulation_path: str):
    """
    Plot 1 (Normalized): Trajectory of the best particle through all iterations.
    Shows how each technology evolved relative to its own min/max range (0-100% scale).
    """
    # Setup the global plot style
    setup_plot_style()

    csv_file_path = os.path.join(simulation_path, "log", "particles_log.csv")
    plots_path = os.path.join(simulation_path, "plots")
    if not os.path.exists(plots_path):
        os.makedirs(plots_path)

    particles_data, global_best_history = parse_particles_csv(csv_file_path)

    # Find the particle that achieved the overall best fitness
    best_fitness = float("inf")
    best_particle_num = None

    for particle_num, data in particles_data.items():
        if data["fitness"]:
            min_fitness = min(data["fitness"])
            if min_fitness < best_fitness:
                best_fitness = min_fitness
                best_particle_num = particle_num

    if best_particle_num is None:
        print("No best particle found")
        return

    # Get the positions of the best particle
    best_particle_positions = particles_data[best_particle_num]["positions"]

    # Aggregate positions by technology for each iteration
    tech_evolution = {"PV": [], "FOWT": [], "BESS": [], "DES": [], "ELY": []}

    for position in best_particle_positions:
        aggregated = aggregate_position_by_technology(position)
        for tech, value in aggregated.items():
            tech_evolution[tech].append(value)

    # Normalize each technology to its own min/max range
    tech_evolution_normalized = {}
    lowers = [0, 0, 0, 0, 0]
    uppers = [
        15 * 1.0,
        10 * 1.0,
        100 * 0.5 + 100 * 0.5,
        5 * 0.25 + 3 * 0.5 + 2 * 0.75 + 2 * 1.0,
        300 * 0.5 + 300 * 0.5,
    ]
    tech_names = ["PV", "FOWTT", "BESS", "DES", "ELY"]
    bounds = dict(zip(tech_names, zip(lowers, uppers)))

    for tech, values in tech_evolution.items():
        if values:
            min_val, max_val = bounds[tech]
            if max_val > min_val:
                # Normalize to 0-100% range based on min/max
                normalized = [(v - min_val) / (max_val - min_val) * 100 for v in values]
            else:
                # If all values are the same, set to 50%
                normalized = [50] * len(values)
            tech_evolution_normalized[tech] = normalized

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Use consistent technology colors
    colors = TECH_COLORS

    for tech, values in tech_evolution_normalized.items():
        if values:  # Only plot if we have data
            iterations = range(len(values))
            ax.plot(
                iterations,
                values,
                marker="o",
                label=tech,
                color=colors[tech],
                linewidth=2.5,
                markersize=6,
                markeredgecolor="white",
                markeredgewidth=1.0,
            )

    format_axes(ax, ylabel="Relative Change (0% = Min, 100% = Max)")
    ax.set_ylim(0, 100)
    ax.legend(loc="best", frameon=True)

    fig_path = os.path.join(plots_path, "best_particle_trajectory_normalized.png")
    save_figure(fig, fig_path)

    print(f"Best particle trajectory normalized plot saved to {fig_path}")


def plot_global_best_evolution_normalized(simulation_path: str):
    """
    Plot 2 (Normalized): Evolution of the global best position at each iteration.
    Shows how each technology evolved relative to its own min/max range during global best updates.
    """
    # Setup the global plot style
    setup_plot_style()

    csv_file_path = os.path.join(simulation_path, "log", "particles_log.csv")
    plots_path = os.path.join(simulation_path, "plots")
    if not os.path.exists(plots_path):
        os.makedirs(plots_path)

    _, global_best_history = parse_particles_csv(csv_file_path)

    if not global_best_history:
        print("No global best history found")
        return

    # Aggregate positions by technology for each global best update
    tech_evolution = {"PV": [], "FOWT": [], "BESS": [], "DES": [], "ELY": []}

    for entry in global_best_history:
        aggregated = aggregate_position_by_technology(entry["position"])
        for tech, value in aggregated.items():
            tech_evolution[tech].append(value)

    # Normalize each technology to its own min/max range
    tech_evolution_normalized = {}
    lowers = [0, 0, 0, 0, 0]
    uppers = [
        15 * 1.0,
        10 * 1.0,
        100 * 0.5 + 100 * 0.5,
        5 * 0.25 + 3 * 0.5 + 2 * 0.75 + 2 * 1.0,
        300 * 0.5 + 300 * 0.5,
    ]
    tech_names = ["PV", "FOWT", "BESS", "DES", "ELY"]
    bounds = dict(zip(tech_names, zip(lowers, uppers)))

    for tech, values in tech_evolution.items():
        if values:
            min_val, max_val = bounds[tech]
            if max_val > min_val:
                # Normalize to 0-100% range based on min/max
                normalized = [(v - min_val) / (max_val - min_val) * 100 for v in values]
            else:
                # If all values are the same, set to 50%
                normalized = [50] * len(values)
            tech_evolution_normalized[tech] = normalized

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Use consistent technology colors
    colors = TECH_COLORS

    for tech, values in tech_evolution_normalized.items():
        if values:  # Only plot if we have data
            # Use iteration numbers instead of range index
            iterations = [entry["iteration"] for entry in global_best_history]
            ax.plot(
                iterations,
                values,
                marker="s",
                label=tech,
                color=colors[tech],
                linewidth=2.5,
                markersize=6,
                markeredgecolor="white",
                markeredgewidth=1.0,
            )

    format_axes(ax, ylabel="Installed Capacities (% of feasible bounds)")
    ax.set_ylim(0, 103)
    ax.legend(loc="best", frameon=True)

    fig_path = os.path.join(plots_path, "global_best_evolution_normalized.png")
    save_figure(fig, fig_path)

    print(f"Global best evolution normalized plot saved to {fig_path}")


def plot_desalinators_power_heatmap(simulation_path: str, power: pd.Series):
    """
    Plots the dispatched power of desalination units as a dual heatmap.
    Left plot: water_as_load data from the waterAsLoad.db database
    Right plot: actual dispatched power from the simulation

    The heatmap displays days of the year on the x-axis and hours of the day
    on the y-axis.
    """
    import sqlite3
    import json

    HEATMAP_FIGSIZE = (10, 6)

    # Setup the global plot style
    setup_plot_style()

    def sample_weekly(
        data_list: list, width: int = 24 * 7, period: int = 4, phase: int = 0
    ) -> list:
        """
        Sample function that takes 1 week every 4 weeks, same as in data_import.py
        """
        return [
            data_list[i]
            for i in range(len(data_list))
            if (i // width) % period == phase
        ]

    # Extract WD multiplier from simulation path to find correct database
    wd_multiplier = "x1"  # default
    if "_x1_5" in simulation_path:
        wd_multiplier = "x1_5"
    elif "_x2" in simulation_path:
        wd_multiplier = "x2"
    elif "_x1" in simulation_path:
        wd_multiplier = "x1"

    # Convert underscores back to dots for database path
    db_folder = wd_multiplier.replace("_", "")
    if db_folder == "x15":
        db_folder = "x1_5"

    # Load water_as_load data from database
    water_db_path = os.path.join(
        simulation_path, "data", f"water_{db_folder}", "WaterAsLoad.db"
    )
    water_as_load_data = None

    try:
        conn = sqlite3.connect(water_db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT `values` FROM timeseries WHERE name='water_as_load'")
        result = cursor.fetchone()
        conn.close()

        if result and result[0]:
            # Parse the JSON array from the database
            water_as_load_values = json.loads(result[0])

            # Apply the same weekly sampling as in data_import.py
            sampled_water_values = sample_weekly(water_as_load_values, 24 * 7, 4)

            # Create datetime index matching the power series
            # Since we sampled, we need to ensure the length matches
            if len(sampled_water_values) <= len(power):
                water_as_load_series = pd.Series(
                    sampled_water_values, index=power.index[: len(sampled_water_values)]
                )
            else:
                # If sampled data is still longer, truncate
                water_as_load_series = pd.Series(
                    sampled_water_values[: len(power)], index=power.index
                )

            water_as_load_data = water_as_load_series
        else:
            print(f"Warning: No water_as_load data found in {water_db_path}")
            water_as_load_data = pd.Series(0, index=power.index)

    except Exception as e:
        print(f"Warning: Could not load water_as_load data from {water_db_path}: {e}")
        # Create a dummy series with zeros if we can't load the data
        water_as_load_data = pd.Series(0, index=power.index)

    # Ensure water_as_load_data is never None
    if water_as_load_data is None:
        print(f"Warning: water_as_load_data is None, creating dummy data")
        water_as_load_data = pd.Series(0, index=power.index)

    # Check and compare the sums of both time series
    power_sum = power.sum()
    water_sum = water_as_load_data.sum()
    print(f"Time series comparison for {simulation_path}:")
    print(f"  Dispatched power sum: {power_sum:.2f} MW")
    print(f"  Water as load sum: {water_sum:.2f} MW")
    print(f"  Difference: {abs(power_sum - water_sum):.2f} MW")
    print(f"  Sums are equal: {abs(power_sum - water_sum) < 1e-6}")

    # Convert both series to DataFrames for easier manipulation
    power_df = power.to_frame(name="power")
    power_df["dayofyear"] = power_df.index.dayofyear
    power_df["hour"] = power_df.index.hour

    water_df = water_as_load_data.to_frame(name="power")
    water_df["dayofyear"] = water_df.index.dayofyear
    water_df["hour"] = water_df.index.hour

    # Pivot both DataFrames
    power_pivot = power_df.pivot(index="hour", columns="dayofyear", values="power")
    water_pivot = water_df.pivot(index="hour", columns="dayofyear", values="power")

    # Create the dual heatmap plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # Determine common color scale for both plots
    vmin = min(power_pivot.min().min(), water_pivot.min().min())
    vmax = max(power_pivot.max().max(), water_pivot.max().max())

    # Left plot: water_as_load data
    _ = sns.heatmap(
        water_pivot,
        cmap="viridis",
        ax=ax1,
        linewidths=0,
        rasterized=True,
        vmin=vmin,
        vmax=vmax,
        cbar=False,  # We'll add a shared colorbar later
    )

    # Right plot: actual dispatched power
    heatmap2 = sns.heatmap(
        power_pivot,
        cmap="viridis",
        cbar_kws={"label": "Power (MW)"},
        ax=ax2,
        linewidths=0,
        rasterized=True,
        vmin=vmin,
        vmax=vmax,
    )

    # Apply consistent formatting to both plots
    format_axes(ax1, xlabel="Day of the Year", ylabel="Hour of the Day")
    format_axes(
        ax2,
        xlabel="Day of the Year",
        ylabel="",
    )

    # Set titles
    ax1.set_title("DES-NoOptDispatchx1", fontsize=16, fontweight="bold")
    ax2.set_title("FULLx1", fontsize=16, fontweight="bold")

    # Explicitly remove the automatic "hour" label from the right plot
    ax2.set_ylabel("")

    # Style the colorbar (only on the right plot)
    cbar = heatmap2.collections[0].colorbar
    cbar.ax.set_ylabel("Power (MW)", fontsize=16, fontweight="bold")
    cbar.ax.tick_params(labelsize=14)

    # Improve tick spacing for better readability on both plots
    for ax, pivot in [(ax1, water_pivot), (ax2, power_pivot)]:
        ax.set_xticks(range(0, len(pivot.columns), 30))  # Every 30 days
        ax.set_xticklabels(range(0, len(pivot.columns), 30))
        ax.set_yticks(range(0, 24, 4))  # Every 4 hours
        ax.set_yticklabels(range(0, 24, 4))

    plt.tight_layout()

    # --- Save combined figure ---
    save_figure(
        fig, f"{simulation_path}/plots/desalinator_power_heatmap_comparison.png"
    )

    # --- Save left figure (water_as_load only) ---
    fig_left, ax_left = plt.subplots(figsize=HEATMAP_FIGSIZE)
    sns.heatmap(
        water_pivot,
        cmap="viridis",
        linewidths=0,
        rasterized=True,
        vmin=vmin,
        vmax=vmax,
        cbar_kws={"label": "Power (MW)"},
        ax=ax_left,
    )
    format_axes(ax_left, xlabel="Day of the Year", ylabel="Hour of the Day")
    ax_left.set_title("DES-NoOptDispatchx1", fontsize=16, fontweight="bold")
    ax_left.set_yticks(range(0, 24, 4))
    ax_left.set_yticklabels(range(0, 24, 4))
    ax_left.set_xticks(range(0, len(water_pivot.columns), 30))
    ax_left.set_xticklabels(range(0, len(water_pivot.columns), 30))
    plt.tight_layout()
    save_figure(fig_left, f"{simulation_path}/plots/desalinator_power_heatmap_left.png")

    # --- Save right figure (dispatched power only) ---
    fig_right, ax_right = plt.subplots(figsize=HEATMAP_FIGSIZE)
    sns.heatmap(
        power_pivot,
        cmap="viridis",
        linewidths=0,
        rasterized=True,
        vmin=vmin,
        vmax=vmax,
        cbar_kws={"label": "Power (MW)"},
        ax=ax_right,
    )
    format_axes(ax_right, xlabel="Day of the Year", ylabel="Hour of the Day")
    ax_right.set_title("FULLx1", fontsize=16, fontweight="bold")
    ax_right.set_yticks(range(0, 24, 4))
    ax_right.set_yticklabels(range(0, 24, 4))
    ax_right.set_xticks(range(0, len(power_pivot.columns), 30))
    ax_right.set_xticklabels(range(0, len(power_pivot.columns), 30))
    plt.tight_layout()
    save_figure(
        fig_right, f"{simulation_path}/plots/desalinator_power_heatmap_right.png"
    )

    # Save the figure
    save_figure(
        fig, f"{simulation_path}/plots/desalinator_power_heatmap_comparison.png"
    )


def plot_scenarios_fitness_evolution(base_path: str = "simulations_unzipped"):
    """
    Plot fitness evolution for all scenarios across PSO iterations.
    Creates a plot with 9 lines where:
    - Color represents WD multiplier (x1, x1.5, x2) from SCENARIO_COLORS
    - Marker style represents scenario type (FULL, NoWDfeatures, ffgReserveOnly)
    """
    setup_plot_style()

    # Define scenario types and their corresponding markers
    scenario_types = {
        "FULL": "o",  # circle
        "NoWDfeatures": "s",  # square
        "ffgReserveOnly": "^",  # triangle
    }

    # Define WD multipliers and their corresponding colors
    wd_multipliers = {
        "x1": 0,  # Index in SCENARIO_COLORS
        "x1_5": 1,  # Index in SCENARIO_COLORS
        "x2": 2,  # Index in SCENARIO_COLORS
    }

    # Get colors from the style
    colors = SCENARIO_COLORS

    fig, ax = plt.subplots(figsize=(12, 8))

    # Track scenarios for legend
    color_legend_handles = []
    marker_legend_handles = []
    color_labels = ["x1", "x1.5", "x2"]
    marker_labels = ["FULL", "NoWDfeatures", "ffgReserveOnly"]

    # Process each scenario
    for scenario_type, marker in scenario_types.items():
        for wd_mult, color_idx in wd_multipliers.items():
            # Find simulation directory matching this scenario and WD multiplier
            simulation_dirs = [
                d
                for d in os.listdir(base_path)
                if os.path.isdir(os.path.join(base_path, d))
                and scenario_type in d
                and wd_mult in d
                and "csv" not in d  # Exclude CSV files
            ]

            if not simulation_dirs:
                print(f"No simulation found for {scenario_type}_{wd_mult}")
                continue

            # Use the first (should be only) matching directory
            sim_dir = simulation_dirs[0]
            fitness_file = os.path.join(
                base_path, sim_dir, "log", "best_fitness_history.csv"
            )

            if not os.path.exists(fitness_file):
                print(f"Fitness file not found: {fitness_file}")
                continue

            # Read fitness data
            df = pd.read_csv(fitness_file)
            iterations = df["Iteration"].values
            fitness_values = df["Fitness"].values

            # Plot the line
            _ = ax.plot(
                iterations,
                fitness_values,
                marker=marker,
                color=colors[color_idx],
                linewidth=2.5,
                markersize=8,
                markeredgecolor="white",
                markeredgewidth=1.0,
                label=f"{scenario_type}_{wd_mult.replace('_', '.')}",
            )[0]

    # Create custom legends
    # Color legend for WD multipliers
    for color_label, color in zip(color_labels, colors):
        color_legend_handles.append(
            plt.Line2D([0], [0], color=color, linewidth=3, label=f"WD {color_label}")
        )

    # Marker legend for scenario types
    for scenario_type, marker in scenario_types.items():
        marker_legend_handles.append(
            plt.Line2D(
                [0],
                [0],
                marker=marker,
                color="black",
                linestyle="None",
                markersize=8,
                markeredgecolor="white",
                markeredgewidth=1.0,
                label=scenario_type,
            )
        )

    # Format axes
    format_axes(ax, ylabel="Fitness Value")

    # Add legends
    color_legend = ax.legend(
        handles=color_legend_handles,
        loc="upper right",
        title="WD Multiplier",
        frameon=True,
    )
    _ = ax.legend(
        handles=marker_legend_handles,
        loc="upper left",
        title="Scenario Type",
        frameon=True,
    )

    # Add the color legend back since the second legend call overwrites the first
    ax.add_artist(color_legend)

    # Set title and save
    plots_dir = os.path.join(base_path, "plots")
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    # Save fitness evolution for all scenarios
    fig_path = os.path.join(plots_dir, "scenarios_fitness_evolution.png")
    save_figure(fig, fig_path)

    print(f"Scenarios fitness evolution plot saved to {fig_path}")
