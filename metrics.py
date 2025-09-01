from contextlib import contextmanager
import itertools
import os
import sys
import zipfile
import math
import pypsa
import pandas as pd
import numpy as np
import re
from src.data_import import load_environment
from itertools import combinations_with_replacement

from utils.csv_from_log import csv_from_log
from utils.plots import (
    plot_global_best_evolution,
    plot_desalinators_power_heatmap,
    plot_global_best_evolution_normalized,
    plot_scenarios_fitness_evolution,
)
from utils.metrics_plot import plot_metrics_hardcoded
from utils.stacked_installed_power import plot_stacked_installed_power
from utils.plot_style import setup_plot_style


TOL = 1e-3
ZIP_FOLDER_PATH = "simulations"
EXTRACTED_FOLDER_PATH = "simulations_unzipped"

@contextmanager
def suppress_output():
    """
    A context manager to temporarily suppress both stdout and stderr.

    Usage:
    with suppress_output():
        # Code with print statements to be silenced
        print("This will not be printed to stdout.")
        print("This error will not be printed to stderr.", file=sys.stderr)

    print("This will be printed.")
    """
    # Save the original stdout and stderr so we can restore them later
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    # Redirect stdout and stderr to the null device
    sys.stdout = open(os.devnull, "w")
    sys.stderr = open(os.devnull, "w")

    try:
        # Yield control back to the 'with' block
        yield
    finally:
        # Always restore the original stdout and stderr, even if errors occur
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = original_stdout
        sys.stderr = original_stderr


def get_index(indices: pd.Index, pattern: str) -> pd.Index:
    return pd.Index(filter(lambda ind: pattern in ind, indices))


def scenario_path_info(scenario_path: str) -> dict[str, str]:
    dir = os.path.basename(scenario_path)

    name = dir.split("_", maxsplit=1)[-1]
    (scenario, tag) = name.split("_", maxsplit=1)

    return {
        "name": name,
        "scenario": scenario,
        "tag": tag,
        "dir": dir,
    }


def to_store(link_indexer: pd.Index) -> dict[str, str]:
    translation = {}
    PATTERN: re.Pattern = re.compile(r"link_(charge|discharge)-(\w+-\d+)-\d+")

    for link in link_indexer:
        store = PATTERN.match(link).group(2)
        translation |= {link: store}

    return translation


def df_roll(df: pd.DataFrame) -> pd.DataFrame:
    PERIODS: int = 1

    last = df.iloc[-1]
    rolled = df.shift(PERIODS)
    rolled.iloc[0] = last

    return rolled


def mean_when_on(
    values: pd.DataFrame | pd.Series, status: pd.DataFrame | pd.Series
) -> float:
    true_values: pd.Series

    assert type(values) == type(
        status
    ), f"Incompatible types for values ({type(values)}) and status ({type(status)})"

    if isinstance(values, pd.DataFrame):
        true_values = (values * status).sum(axis=1)
    elif isinstance(values, pd.Series):
        true_values = values * status
    else:
        raise TypeError(f"{type(values)} is not a valid type for 'values'")

    res = true_values[true_values > TOL].mean()
    return 0.0 if math.isnan(res) else res


# Funzione per trovare la minima combinazione lineare
# a coefficienti interi di elementi del vettore che sia maggiore del numero
def find_min_linear_combination(vector, target):
    vector = sorted(vector)  # Assicurarsi che il vettore sia ordinato
    min_result = float("inf")
    best_combination = None

    # Limitiamo la lunghezza delle combinazioni per evitare ricerche inutili
    for length in range(1, len(vector) + 1):
        for combination in combinations_with_replacement(vector, length):
            result = sum(combination)
            if result > target and result < min_result:
                min_result = result
                best_combination = combination

    return min_result, best_combination


# Extract all zips in zip folder into therir folder in dest
def extract_all_zips(zip_folder: str, dest: str) -> list[str]:
    extracted_folders = []

    for item in os.listdir(zip_folder):
        if item.endswith(".zip"):

            # Build path names
            zip_path = os.path.join(zip_folder, item)
            folder_name = os.path.splitext(item)[0]
            extract_path = os.path.join(dest, folder_name)
            os.makedirs(extract_path, exist_ok=True)

            # Extract
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(extract_path)
                extracted_folders.append(extract_path)

    return extracted_folders


def extract_network(folder: str) -> pypsa.Network:
    network_path = os.path.join(folder, "log", "solution", "network")

    assert os.path.exists(network_path), RuntimeError("Network not found")
    return pypsa.Network(network_path)


def find_FULL_x1(folder: str) -> str:
    for item in os.listdir(folder):
        if "FULL_x1" in item and not item.endswith(".zip"):
            return os.path.join(folder, item)
    raise RuntimeError("FULL_x1 not found in: " + folder)


def fuel_consumption(network: pypsa.Network, ffg_i: pd.Index) -> float:
    m = 0.215
    q = 0.00457

    consumi = (
        m * network.generators_t.p[ffg_i] * 1000
        + q * network.generators.p_nom[ffg_i] * 1000
    ) * network.generators_t.status.loc[:, ffg_i]

    return float(consumi.sum().sum())


def estimate_desalinator_capital_cost(
    network: pypsa.Network, scenario_path: str
) -> float:

    scenario_info = scenario_path_info(scenario_path)
    tag = scenario_info["tag"]
    scenario = scenario_info["scenario"]

    # Extract environment for relevant scenario
    scenario_db_path = f"{scenario_path}/data/water_{tag}/{scenario}.db"
    environment = load_environment(
        scenario_db_path, moduleNames=["demand", "electrolyzer", "desalinator"]
    )

    H2_BETA = abs(environment["electrolyzer"][0]["beta"])
    DES_BETA = abs(environment["desalinator"][0]["beta"])

    # Extract water demand from demand modules
    des_electric_demand = sum(
        map(
            lambda x: x["loads"] * x["scale"],
            filter(
                lambda x: x["name"] == "water_as_electric_load", environment["demand"]
            ),
        )
    )

    h2_electric_demand = network.links_t["p0"][
        get_index(network.links_t["p0"].columns.values, "link_charge-electrolyzer")
    ].sum(axis=1)

    # Convert to electric load
    electric_load = des_electric_demand + h2_electric_demand * H2_BETA / DES_BETA
    load_estimate = electric_load.max()

    des_powers = list(map(lambda x: x["p_nom"], environment["desalinator"]))
    min_result, _ = find_min_linear_combination(des_powers, load_estimate)

    return min_result * environment["desalinator"][0]["cc"]


def calc_res_requirement(network: pypsa.Network) -> pd.Series:
    vres_i = network.generators.index[network.generators.index.str.contains("vres")]

    loads = network.loads_t.p_set
    demandReserve = 0.1 * loads.sum(1)
    vResReserve = (
        network.generators.p_nom_opt.loc[vres_i]
        * network.generators_t.p_max_pu.loc[:, vres_i]
    ).sum(1) * 0.1
    req_reserve = demandReserve + vResReserve + 1
    return req_reserve


def calc_reserve(network: pypsa.Network) -> dict[str, pd.Series]:
    des_i = get_index(network.generators.index, "desalinator")
    status_des: pd.DataFrame = network.generators_t.status.loc[:, des_i]

    # Sum over resources
    return {
        "ffg": compute_ffg_reserve(network).sum(1),
        "bess": calc_storage_reserve(network, "bess").sum(1),
        "electrolyzer": calc_storage_reserve(network, "electrolyzer").sum(1),
        "desalinator": calc_desalinator_reserve(network, status_des).sum(1),
    }


# Returns a pd.Dataframe with [snapshot] x [ffg]
def compute_ffg_reserve(network: pypsa.Network) -> pd.DataFrame:
    ffg_i = get_index(network.generators.index, "ffg")

    ffg_p = network.generators_t.p.loc[:, ffg_i]
    status = network.generators_t.status.loc[:, ffg_i]

    ffg_p_min = network.generators.loc[ffg_i, "p_min_pu"]
    ffg_p_nom = network.generators.loc[ffg_i, "p_nom"]

    return ffg_p - ffg_p_min * ffg_p_nom * status


# Returns a pd.Dataframe with [snapshot] x [storage]
def calc_storage_reserve(network: pypsa.Network, storage: str) -> pd.DataFrame:
    charge_i = get_index(network.links.index, f"link_charge-{storage}")
    disch_i = get_index(network.links.index, f"link_discharge-{storage}")

    tank_reserve = calc_storage_tank_reserve(network, storage)
    link_charge_reserve = calc_storage_link_charge_reserve(network, storage).rename(
        columns=to_store(charge_i)
    )
    link_discharge_reserve = calc_storage_link_discharge_reserve(
        network, storage
    ).rename(columns=to_store(disch_i))

    return np.minimum(
        tank_reserve + link_discharge_reserve,
        link_charge_reserve + link_discharge_reserve,
    )


# Returns a pd.Dataframe with [snapshot] x [storage]
def calc_storage_tank_reserve(network: pypsa.Network, storage: str) -> pd.DataFrame:
    store_i = get_index(network.stores.index, storage)
    disch_i = get_index(network.links.index, f"link_discharge-{storage}")

    eta_disch = network.links.loc[disch_i, "efficiency"].rename(to_store(disch_i))
    e_max = network.stores.loc[store_i, "e_nom_max"]
    e_nom = network.stores.loc[store_i, "e_nom"]

    e = network.stores_t.e.loc[:, store_i]

    return np.minimum(
        (eta_disch * e_max * e_nom - eta_disch * df_roll(e)),
        (eta_disch * e_max * e_nom - eta_disch * e),
    )


# Returns a pd.Dataframe with [snapshot] x [storage]
def calc_storage_link_discharge_reserve(
    network: pypsa.Network, storage: str
) -> pd.DataFrame:
    disch_i = get_index(network.links.index, f"link_discharge-{storage}")

    d_out = network.links_t.p0.loc[:, disch_i]
    eta = network.links.loc[disch_i, "efficiency"]

    return (eta * d_out).rename(columns=to_store(disch_i))


# Returns a pd.Dataframe with [snapshot] x [storage]
def calc_storage_link_charge_reserve(
    network: pypsa.Network, storage: str
) -> pd.DataFrame:
    charge_i = get_index(network.links.index, f"link_charge-{storage}")

    P_nom = network.links.loc[charge_i, "p_nom"]
    d_in = network.links_t.p0.loc[:, charge_i]

    return (P_nom - d_in).rename(columns=to_store(charge_i))


# Returns a pd.Dataframe with [snapshot] x [desalinator]
def calc_desalinator_reserve(
    network: pypsa.Network, status: pd.DataFrame
) -> pd.DataFrame:
    des_i = get_index(network.generators.index, "desalinator")

    p_nom = network.generators.loc[des_i, "p_nom"]
    d_max = -network.generators.loc[des_i, "p_min_pu"]
    d = network.generators_t.p.loc[:, des_i]

    assert (d < TOL).all().all()
    return d_max * p_nom * status + d


def setup():
    os.makedirs(EXTRACTED_FOLDER_PATH, exist_ok=True)
    _ = extract_all_zips(ZIP_FOLDER_PATH, EXTRACTED_FOLDER_PATH)


def filter_available_scenarios(scenarios: list[str] = None) -> list[str]:
    extracted_paths = [
        os.path.join(EXTRACTED_FOLDER_PATH, dir)
        for dir in os.listdir(EXTRACTED_FOLDER_PATH)
        if os.path.isdir(os.path.join(EXTRACTED_FOLDER_PATH, dir))
    ]

    # exclude the plots directory
    extracted_paths = list(
        filter(lambda x: x.split("/")[-1] != "plots", extracted_paths)
    )

    # if scenarios is not None, filter by scenario whitelist
    if scenarios is not None:
        extracted_paths = list(
            filter(
                lambda x: re.sub(
                    r"^\d{4}-\d{2}-\d{2}T\d{2}-\d{2}_", "", x.split("/")[-1]
                )
                in scenarios,
                extracted_paths,
            )
        )

    extracted_paths = sorted(extracted_paths)
    return extracted_paths


def compute_work_hours(
    network: pypsa.Network, req_reserve: pd.Series, reserve: dict[str, pd.Series]
) -> dict[str, int]:
    reserve_types = [
        ("ffg",),
        ("ffg", "bess"),
        ("ffg", "bess", "electrolyzer"),
        ("ffg", "bess", "desalinator"),
        ("ffg", "bess", "electrolyzer", "desalinator"),
    ]

    counters = {
        "ffg_h_counter": 0,
        "ffg_bess_h_counter": 0,
        "ffg_bess_ely_h_counter": 0,
        "ffg_bess_wd_h_counter": 0,
        "ffg_bess_ely_wd_h_counter": 0,
    }

    for snap in network.snapshots:
        for reserve_type, counter in zip(reserve_types, counters.keys()):
            if sum(reserve[rt][snap] for rt in reserve_type) >= req_reserve[snap] - TOL:
                counters[counter] += 1
    return counters


def calc_status_des(
    network: pypsa.Network,
    des_i: pd.Index,
    reserve: dict[str, pd.Series],
    req_reserve: pd.Series,
) -> tuple[pd.DataFrame, pd.Series]:
    if not (
        network.generators.p_max_pu.loc[des_i].any() == 0
        and network.generators_t.stand_by_cost.empty
    ):
        status_des = network.generators_t.status.loc[:, des_i]
    else:
        status_des = abs(network.generators_t.p.loc[:, des_i]) > TOL

        r_des = calc_desalinator_reserve(network, status_des).sum(1)
        reserve_to_satisfy = (
            r_des
            + reserve["ffg"]
            + reserve["bess"]
            + reserve["electrolyzer"]
            - req_reserve
        )
        reserve_to_satisfy = reserve_to_satisfy.loc[reserve_to_satisfy < -TOL]

        index_reserve_to_satisfy = reserve_to_satisfy.index

        for snap in index_reserve_to_satisfy:
            des_i_spenti = des_i[network.generators_t.p.loc[snap, des_i] < TOL]

            combs = list(
                itertools.chain.from_iterable(
                    itertools.combinations(des_i_spenti, r)
                    for r in range(1, len(des_i_spenti) + 1)
                )
            )

            valid_combs = []

            for comb in combs:
                comb_sum = network.generators.loc[list(comb), "p_nom"].sum()
                lower_bound = comb_sum * 0.1
                upper_bound = comb_sum

                if (
                    upper_bound
                    + reserve["ffg"].loc[snap]
                    + reserve["bess"].loc[snap]
                    + reserve["electrolyzer"].loc[snap]
                    - req_reserve[snap]
                    >= -TOL
                ):
                    valid_combs.append((comb, lower_bound))

            if valid_combs:
                best_comb = min(valid_combs, key=lambda x: x[1])[0]
                status_des.loc[snap, best_comb] = 1

    # Status of the equivalent SBC desalinator
    # ON when at least one is ON, otherwise OFF
    status_des_eq = status_des.loc[:, des_i].any(axis=1)

    assert (status_des <= network.generators_t.status.loc[:, des_i]).all().all()
    return status_des, status_des_eq


def f_obj(
    network: pypsa.Network, verbose: bool = False, logfile=None
) -> tuple[float, float, float]:
    """
    Calculates the objective function value directly from the PyPSA network post-optimization.
    This reflects the costs as seen by the solver.
    If verbose is True, it prints a detailed breakdown of the costs to the specified logfile.
    """
    # --- Capital Cost Calculations ---
    bess_capital = network.stores["e_nom_opt"].mul(network.stores["capital_cost"]).sum()
    link_capital = network.links["p_nom_opt"].mul(network.links["capital_cost"]).sum()
    generators_capital = (
        network.generators["p_nom_opt"].mul(network.generators["capital_cost"]).sum()
    )
    capital = float(bess_capital + link_capital + generators_capital)

    # --- Marginal & Operational Cost Calculations ---
    weight = network.snapshot_weightings.iloc[0].iloc[0]
    link_marginal = (
        network.links_t["p0"].mul(network.links["marginal_cost"]).sum().sum() * weight
    )
    generators_marginal = (
        network.generators_t["p"].mul(network.generators["marginal_cost"]).sum().sum()
        * weight
    )
    constant_sbcs = (
        network.generators_t["status"]
        .mul(network.generators["stand_by_cost"])
        .sum()
        .sum()
        * weight
    )
    time_varying_sbcs = (
        network.generators_t["status"]
        .mul(network.generators_t["stand_by_cost"])
        .sum()
        .sum()
        * weight
    )
    marginal = float(
        link_marginal + generators_marginal + constant_sbcs + time_varying_sbcs
    )

    total_cost = capital + marginal

    # --- Verbose Logging ---
    if verbose:
        print(
            "\n" + "=" * 20 + " Calculating f_obj (Model Costs) " + "=" * 20,
            file=logfile,
        )
        print("\n--- Capital Costs ---", file=logfile)
        print(f"BESS Capital Cost: {bess_capital:,.2f}", file=logfile)
        print(f"Links Capital Cost: {link_capital:,.2f}", file=logfile)
        print(f"Generators Capital Cost: {generators_capital:,.2f}", file=logfile)
        print(f"==> TOTAL CAPITAL: {capital:,.2f}", file=logfile)

        print("\n--- Marginal & Operational Costs ---", file=logfile)
        print(f"Snapshot Weighting Applied: {weight}", file=logfile)
        print(f"Links Marginal Cost: {link_marginal:,.2f}", file=logfile)
        print(f"Generators Marginal Cost: {generators_marginal:,.2f}", file=logfile)
        print(
            f"Constant Stand-by Costs (from generators.stand_by_cost): {constant_sbcs:,.2f}",
            file=logfile,
        )
        print(
            f"Time-Varying Stand-by Costs (from generators_t.stand_by_cost): {time_varying_sbcs:,.2f}",
            file=logfile,
        )
        print(f"==> TOTAL MARGINAL & OPERATIONAL: {marginal:,.2f}", file=logfile)

        print("\n" + "-" * 60, file=logfile)
        print(f"### Total f_obj Cost: {total_cost:,.2f} ###", file=logfile)
        print("=" * 63 + "\n", file=logfile)

    return (total_cost, capital, marginal)


def f_obj_real(
    scenario_path: str,
    network: pypsa.Network,
    ffg_i: pd.Index,
    des_i: pd.Index,
    verbose: bool = False,
    logfile=None,
) -> tuple[float, float, float]:
    """
    Calculates the "real" objective function post-simulation, applying specific rules
    for stand-by costs (SBCs) and handling special scenarios like 'WaterAsLoad'.
    If verbose is True, it prints a detailed breakdown of the costs to the specified logfile.
    """
    scenario_info = scenario_path_info(scenario_path)
    scenario = scenario_info["scenario"]

    # --- Capital Cost Calculations ---
    bess_capital = network.stores["e_nom_opt"].mul(network.stores["capital_cost"]).sum()
    links_capital = network.links["p_nom_opt"].mul(network.links["capital_cost"]).sum()
    base_generators_capital = (
        network.generators["p_nom_opt"].mul(network.generators["capital_cost"]).sum()
    )

    estimated_des_capital = 0.0
    if scenario == "WaterAsLoad":
        estimated_des_capital = estimate_desalinator_capital_cost(
            network, scenario_path
        )

    capital = float(
        bess_capital + links_capital + base_generators_capital + estimated_des_capital
    )

    # --- Marginal & Operational Cost Calculations ---
    weight = network.snapshot_weightings.iloc[0].iloc[0]
    links_marginal = (
        network.links_t["p0"].mul(network.links["marginal_cost"]).sum().sum() * weight
    )
    generators_marginal = (
        network.generators_t["p"].mul(network.generators["marginal_cost"]).sum().sum()
        * weight
    )

    network_path_full = find_FULL_x1(EXTRACTED_FOLDER_PATH)

    with suppress_output():
        n_full = extract_network(network_path_full)

    ffg_sbc = (
        network.generators_t["status"]
        .loc[:, ffg_i]
        .mul(n_full.generators["stand_by_cost"].loc[ffg_i])
        .sum()
        .sum()
        * weight
    )

    water_sbc = 0.0
    if scenario == "WaterAsLoad":
        water_sbc = (
            n_full.generators_t["stand_by_cost"].loc[:, "sbc_DSL"].sum() * weight
        )
    else:
        reserve = calc_reserve(network)
        req_reserve = calc_res_requirement(network)
        _, status_des_eq = calc_status_des(network, des_i, reserve, req_reserve)
        water_sbc = (
            status_des_eq.mul(
                n_full.generators_t["stand_by_cost"].loc[:, "sbc_DSL"]
            ).sum()
            * weight
        )

    marginal = float(links_marginal + generators_marginal + ffg_sbc + water_sbc)

    total_cost = capital + marginal

    # --- Verbose Logging ---
    if verbose:
        print(
            "\n"
            + "=" * 20
            + f" Calculating f_obj_real (Scenario: {scenario}) "
            + "=" * 20,
            file=logfile,
        )
        print("\n--- Capital Costs ---", file=logfile)
        print(f"BESS Capital Cost: {bess_capital:,.2f}", file=logfile)
        print(f"Links Capital Cost: {links_capital:,.2f}", file=logfile)
        print(
            f"Base Generators Capital Cost: {base_generators_capital:,.2f}",
            file=logfile,
        )
        if scenario == "WaterAsLoad":
            print(
                f"Estimated Desalinator Capital Cost (for WaterAsLoad): {estimated_des_capital:,.2f}",
                file=logfile,
            )
        print(f"==> TOTAL CAPITAL: {capital:,.2f}", file=logfile)

        print("\n--- Marginal & Operational Costs ---", file=logfile)
        print(f"Snapshot Weighting Applied: {weight}", file=logfile)
        print(f"Links Marginal Cost: {links_marginal:,.2f}", file=logfile)
        print(f"Generators Marginal Cost: {generators_marginal:,.2f}", file=logfile)

        print("\nRecalculating SBCs using 'FULL_x1' network costs:", file=logfile)
        print(f"FFG Stand-by Cost (from FULL_x1): {ffg_sbc:,.2f}", file=logfile)
        if scenario == "WaterAsLoad":
            print(
                f"Water SBC (Approximation for WaterAsLoad, from FULL_x1): {water_sbc:,.2f}",
                file=logfile,
            )
        else:
            print(
                f"Water SBC (Calculated from DES status & FULL_x1): {water_sbc:,.2f}",
                file=logfile,
            )

        print(f"==> TOTAL MARGINAL & OPERATIONAL: {marginal:,.2f}", file=logfile)

        print("\n" + "-" * 60, file=logfile)
        print(f"### Total f_obj_real Cost: {total_cost:,.2f} ###", file=logfile)
        print("=" * 63 + "\n", file=logfile)

    return (total_cost, capital, marginal)


def generate_metrics(scenarios: list[str] = None):
    digits = {
        "real f_obj (M€)": 4,
        "real capital (M€)": 4,
        "real marginal (M€)": 4,
        "f_obj (M€)": 4,
        "capital (M€)": 4,
        "marginal (M€)": 4,
    }

    extracted_paths = filter_available_scenarios(scenarios)

    all_data = []

    for scenario_path in extracted_paths:
        if not os.path.isdir(scenario_path):
            continue

        scenario_info = scenario_path_info(scenario_path)

        with suppress_output():
            network = extract_network(scenario_path)
        assert network is not None, f"Network not found in folder {scenario_path}"

        # Prepare indexers
        vres_i: pd.Index = network.generators.index[
            network.generators.index.str.contains("vres")
        ]
        des_i: pd.Index = network.generators.index[
            network.generators.index.str.contains("desalinator")
        ]
        ffg_i: pd.Index = network.generators.index[
            network.generators.index.str.contains("ffg")
        ]

        # Compute reserve
        req_reserve = calc_res_requirement(network)
        reserve = calc_reserve(network)

        # Compute work hours
        counters = compute_work_hours(network, req_reserve, reserve)

        (real_cost, real_capital, real_marginal) = f_obj_real(
            scenario_path, network, ffg_i, des_i
        )
        (cost, capital, marginal) = f_obj(network)

        # Extract data from the network
        data = {
            "Scenario": scenario_info["name"],
            # -- Global Information --
            "Typ. h./yr.": len(network.snapshots),
            "real f_obj (M€)": real_cost / 1e6,
            "real capital (M€)": real_capital / 1e6,
            "real marginal (M€)": real_marginal / 1e6,
            "f_obj (M€)": cost / 1e6,
            "capital (M€)": capital / 1e6,
            "marginal (M€)": marginal / 1e6,
            # -- Planning --
            "PV (MW)": network.generators.p_nom.loc["vres-0"],
            "FOWT (MW)": network.generators.p_nom.loc["vres-1"],
            "BESS TOT (MWh)": network.stores.e_nom_opt["bess-0"]
            + network.stores.e_nom_opt["bess-1"],
            "BESS 1h (MWh)": network.stores.e_nom_opt["bess-0"],
            "BESS 1h link charge (MW)": network.links.p_nom_opt["link_charge-bess-0-0"],
            "BESS 1h link discharge (MW)": network.links.p_nom_opt[
                "link_discharge-bess-0-0"
            ],
            "BESS 2h (MWh)": network.stores.e_nom_opt["bess-1"],
            "BESS 2h link charge (MW)": network.links.p_nom_opt["link_charge-bess-1-0"],
            "BESS 2h link discharge (MW)": network.links.p_nom_opt[
                "link_discharge-bess-1-0"
            ],
            "H2 TOT (MWh)": network.stores.e_nom_opt["electrolyzer-0"]
            + network.stores.e_nom_opt["electrolyzer-1"],
            "H2 50h (MWh)": network.stores.e_nom_opt["electrolyzer-0"],
            "ELY 50h (MW)": network.links.p_nom_opt["link_charge-electrolyzer-0-0"],
            "FC 50h (MW)": network.links.p_nom_opt["link_discharge-electrolyzer-0-0"],
            "H2 100h (MWh)": network.stores.e_nom_opt["electrolyzer-1"],
            "ELY 100h (MW)": network.links.p_nom_opt["link_charge-electrolyzer-1-0"],
            "FC 100h (MW)": network.links.p_nom_opt["link_discharge-electrolyzer-1-0"],
            # -- Reserve --
            "Share RES (%)": (
                network.loads_t.p.sum().sum()
                - network.generators_t.p.loc[:, ffg_i].sum().sum()
            )
            / network.loads_t.p.sum().sum()
            * 100,
            "DGs (h)": counters["ffg_h_counter"],
            "DGs+BESS (h)": counters["ffg_bess_h_counter"],
            "DGs+BESS+H2 (h)": counters["ffg_bess_ely_h_counter"],
            "DGs+BESS+WD (h)": counters["ffg_bess_wd_h_counter"],
            "DGs+BESS+H2+WD (h)": counters["ffg_bess_ely_wd_h_counter"],
            "Work. hours (h)": network.generators_t.status.loc[:, ffg_i].sum().sum(),
            "Avg. DGs on (#/h)": network.generators_t.status.loc[:, ffg_i]
            .sum(1)
            .mean(),
            "Max DGs ON (#/h)": network.generators_t.status.loc[:, ffg_i].sum(1).max(),
            "Fuel (kl)": int(fuel_consumption(network, ffg_i)),
            "Avg Reserve Demand (MW)": sum(calc_res_requirement(network))
            / len(network.snapshots),
            "Max Reserve demand (MW)": max(calc_res_requirement(network)),
            "DGs Reserve (MW)": reserve["ffg"].mean(),
            "BESS Reserve (MW)": reserve["bess"].mean(),
            "DES Reserve (MW)": reserve["desalinator"].mean(),
            "ELY Reserve (MW)": reserve["electrolyzer"].mean(),
            "Water produced (l)": -network.generators_t.p.loc[:, des_i].sum().sum()
            * 222.2222222,
        }

        # Compute working hours
        status_des, status_des_eq = calc_status_des(
            network, des_i, reserve, req_reserve
        )

        for i in range(4):
            des_class_i = get_index(network.generators.index, f"desalinator-{i}")
            sol = network.generators.p_nom.loc[des_class_i].sum()
            data[f"DES {(i+1)*0.25}MW units"] = sol / ((i + 1) * 0.25)

        for des in des_i:
            p_max_pu = network.generators.loc[des, "p_max_pu"]
            p_nom = network.generators.loc[des, "p_nom"]
            # assert type(p_nom) == float
            assert isinstance(p_nom, (float, np.float64))
            under_threshold_i = (
                network.generators_t.p.loc[:, des] > p_max_pu * p_nom
            ) & (network.generators_t.p.loc[:, des] < -TOL)

            submodule_id = des.split("-")[-1]
            sid = f"des-{p_nom}MW-{submodule_id}"

            data[f"{sid} ON (h)"] = status_des[des].sum(axis=0)
            data[f"{sid} Load (%)"] = (
                abs(
                    mean_when_on(
                        values=-network.generators_t.p.loc[:, des],
                        status=status_des[des],
                    )
                )
                / p_nom
                * 100
            )

            data[f"{sid} Idling (h)"] = under_threshold_i.sum()
            data[f"{sid} ON at Power 0 (h)"] = (
                abs(network.generators_t.p.loc[:, des]) < TOL
            ).sum()
            data[f"{sid} Idling Mean Power (%)"] = (
                network.generators_t.p.loc[under_threshold_i, des].sum()
                / (len(under_threshold_i) * p_nom)
                * 100
            )

        for ffg in ffg_i:
            p_nom = network.generators.loc[ffg, "p_nom"]

            submodule_id = des.split("-")[-1]
            sid = f"ffg-{p_nom}MW-{submodule_id}"

            data[f"{sid} ON (h)"] = network.generators_t.status.loc[:, ffg].sum()
            data[f"{sid} Load (%)"] = (
                mean_when_on(
                    values=network.generators_t.p.loc[:, ffg],
                    status=network.generators_t.status.loc[:, ffg],
                )
                / p_nom
                * 100
            )

        for key in data:
            if isinstance(data[key], (float, np.float64)):
                data[key] = np.round(data[key], digits.get(key, 2))

        # Add new calculated columns for the final table
        data["DES Total (MW)"] = (
            data.get("DES 0.25MW units", 0) * 0.25
            + data.get("DES 0.5MW units", 0) * 0.5
            + data.get("DES 0.75MW units", 0) * 0.75
            + data.get("DES 1.0MW units", 0) * 1.0
        )

        data["ELY-FC (MW)"] = data.get("ELY 50h (MW)", 0) + data.get("ELY 100h (MW)", 0)
        data["BESS Planning (MWh)"] = data["BESS TOT (MWh)"]
        data["H2 Planning (MWh)"] = data["H2 TOT (MWh)"]

        all_data.append(data)

    # Create a DataFrame from the accumulated data
    df = pd.DataFrame(all_data)

    # Rinomina gli scenari per il file finale
    df["Scenario"] = df["Scenario"].str.replace("ffgReserveOnly", "Reserve-DSL-Only")
    df["Scenario"] = df["Scenario"].str.replace("WaterAsLoad", "DES-NoOptDispatch")

    # Riordina gli scenari
    order = ["FULL", "Reserve-DSL-Only", "DES-NoOptDispatch"]

    def sort_key(scenario_name):
        parts = scenario_name.split("_")
        prefix = parts[0]
        version = parts[1]
        try:
            return (order.index(prefix), float(version.replace("x", "")))
        except (ValueError, IndexError):
            return (len(order), 0)  # Mette gli scenari non riconosciuti alla fine

    df["SortKey"] = df["Scenario"].apply(sort_key)
    df = df.sort_values(by="SortKey").drop(columns="SortKey").reset_index(drop=True)

    # Separate sections
    sections = {
        "global_info": [
            "Typ. h./yr.",
            "real f_obj (M€)",
            "real capital (M€)",
            "real marginal (M€)",
            "f_obj (M€)",
            "capital (M€)",
            "marginal (M€)",
        ],
        "planning": [
            "PV (MW)",
            "FOWT (MW)",
            "DES 0.25MW units",
            "DES 0.5MW units",
            "DES 0.75MW units",
            "DES 1.0MW units",
            "BESS TOT (MWh)",
            "BESS 1h (MWh)",
            "BESS 1h link charge (MW)",
            "BESS 1h link discharge (MW)",
            "BESS 2h (MWh)",
            "BESS 2h link charge (MW)",
            "BESS 2h link discharge (MW)",
            "H2 TOT (MWh)",
            "H2 50h (MWh)",
            "ELY 50h (MW)",
            "FC 50h (MW)",
            "H2 100h (MWh)",
            "ELY 100h (MW)",
            "FC 100h (MW)",
        ],
        "reserve": [
            "Share RES (%)",
            "DGs (h)",
            "DGs+BESS (h)",
            "DGs+BESS+H2 (h)",
            "DGs+BESS+WD (h)",
            "DGs+BESS+H2+WD (h)",
            "Work. hours (h)",
            "Avg. DGs on (#/h)",
            "Max DGs ON (#/h)",
            "Fuel (kl)",
            "Avg Reserve Demand (MW)",
            "Max Reserve demand (MW)",
            "DGs Reserve (MW)",
            "BESS Reserve (MW)",
            "DES Reserve (MW)",
            "ELY Reserve (MW)",
            "Water produced (l)",
        ],
        "unit_commitment": [
            col
            for col in df.columns
            if col
            not in [
                "Scenario",
                "Typ. h./yr.",
                "f_obj_real (M€)",
                "f_obj (M€)",
                "PV (MW)",
                "FOWT (MW)",
                "DES 0.25MW units",
                "DES 0.5MW units",
                "DES 0.75MW units",
                "DES 1.0MW units",
                "BESS TOT (MWh)",
                "BESS 1h (MWh)",
                "BESS 1h link charge (MW)",
                "BESS 1h link discharge (MW)",
                "BESS 2h (MWh)",
                "BESS 2h link charge (MW)",
                "BESS 2h link discharge (MW)",
                "H2 TOT (MWh)",
                "H2 50h (MWh)",
                "ELY 50h (MW)",
                "FC 50h (MW)",
                "H2 100h (MWh)",
                "ELY 100h (MW)",
                "FC 100h (MW)",
                "Share RES (%)",
                "DGs (h)",
                "DGs+BESS (h)",
                "DGs+BESS+H2 (h)",
                "DGs+BESS+WD (h)",
                "DGs+BESS+H2+WD (h)",
                "Work. hours (h)",
                "Avg. DGs on (#/h)",
                "Max DGs ON (#/h)",
                "Fuel (kl)",
                "Avg Reserve Demand (MW)",
                "Max Reserve demand (MW)",
                "DGs Reserve (MW)",
                "BESS Reserve (MW)",
                "DES Reserve (MW)",
                "ELY Reserve (MW)",
                "Water produced (l)",
                "DES Total (MW)",
                "ELY-FC (MW)",
                "BESS Planning (MWh)",
                "H2 Planning (MWh)",
            ]
        ],
    }

    for section, cols in sections.items():
        section_df = df[["Scenario"] + cols]
        csv_output_path = os.path.join(EXTRACTED_FOLDER_PATH, f"{section}_output.csv")
        if section == "unit_commitment":
            section_df = section_df.sort_index(axis=1)
        section_df.to_csv(csv_output_path, index=False)
        print(f"{section.capitalize()} CSV written to {csv_output_path}")

    # --- INIZIO NUOVA SEZIONE: Creazione della tabella finale ---
    final_columns = [
        "Scenario",
        "real f_obj (M€)",
        "Share RES (%)",
        "Fuel (kl)",
        "PV (MW)",
        "FOWT (MW)",
        "BESS Planning (MWh)",
        "DES Total (MW)",
        "H2 Planning (MWh)",
        "ELY-FC (MW)",
        "DGs Reserve (MW)",
        "BESS Reserve (MW)",
        "DES Reserve (MW)",
        "ELY Reserve (MW)",
    ]

    final_df = df[final_columns].copy()

    # Rinomina le colonne per la tabella finale
    final_df.rename(
        columns={
            "f_obj (M€)": "$f_{obj}$ (M€)",
            "Share RES (%)": "RES Share (%)",
            "PV (MW)": "PV",
            "FOWT (MW)": "FOWT",
            "BESS Planning (MWh)": "BESS",
            "DES Total (MW)": "DES",
            "H2 Planning (MWh)": "H2 Planning (MWh)",  # Ho corretto il nome, mancava MWh
            "ELY-FC (MW)": "ELY-FC",
            "DGs Reserve (MW)": "DGs",
            "BESS Reserve (MW)": "BESS_reserve",
            "DES Reserve (MW)": "DES_reserve",
            "ELY Reserve (MW)": "ELY_reserve",
        },
        inplace=True,
    )

    final_df.to_csv(
        os.path.join(EXTRACTED_FOLDER_PATH, "final_table_data.csv"), index=False
    )
    print("CSV with final table data successfully generated.")
    # --- FINE NUOVA SEZIONE ---


# Extract from the network a sum of the dispatched powers of the desalination units
def extract_desalinators_power(scenario_path: str) -> pd.Series:

    # Extract the network from the scenario
    with suppress_output():
        network = extract_network(scenario_path)

    # Extract the dispatched powers from the network
    des_i: pd.Index = network.generators.index[
        network.generators.index.str.contains("desalinator")
    ]
    powers: pd.DataFrame = network.generators_t.p[des_i]

    return -powers.sum(1)


def generate_all_metrics_and_plots(
    do_setup: bool = False,
    do_generate_metrics: bool = False,
    scenarios: list[str] = None,
):
    # Setup global plot style at the beginning
    setup_plot_style()

    # Extract all zip files in simulations folder
    if do_setup:
        setup()

    # Generate global report (metrics tables) for all scenarios
    if do_generate_metrics:
        generate_metrics(scenarios)

    extracted_paths = filter_available_scenarios(scenarios)

    # Generate all plots
    plot_metrics_hardcoded()
    plot_stacked_installed_power()  # Add the new stacked power plot
    plot_scenarios_fitness_evolution()
    scenario = "simulations_unzipped/2024-12-07T21-09_FULL_x1"
    des_pow = extract_desalinators_power(scenario)
    plot_desalinators_power_heatmap(scenario, des_pow)

    # Comment out the exit for now to generate all plots
    for scenario in extracted_paths:
        print(f"Processing scenario: {scenario}")
        csv_from_log(f"{scenario}/log", "main.log", "particles_log.csv")

        plot_global_best_evolution(scenario)
        plot_global_best_evolution_normalized(scenario)


if __name__ == "__main__":
    """
    Prepare scenarios whitelist by combining
    ["FULL", "ffgReserveOnly", "WaterAsLoad"]
    with ["_x1", "_x1_5", "_x2"]
    """
    scenarios_base = ["FULL", "ffgReserveOnly", "WaterAsLoad"]
    multipliers = ["_x1", "_x1_5", "_x2"]
    scenarios_whitelist = []
    for base in scenarios_base:
        for mult in multipliers:
            scenarios_whitelist.append(f"{base}{mult}")

    generate_all_metrics_and_plots(scenarios=scenarios_whitelist)
