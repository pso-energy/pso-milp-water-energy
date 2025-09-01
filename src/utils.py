import os
import pandas as pd
import pypsa
from src.typedefs import IndexerType, ModuleType, Solution, StageType, SubmoduleType

from typing import Callable
from xarray import DataArray, concat
from linopy import Variable, LinearExpression
from pypsa.optimization.compat import linexpr


def join_modules(module1: ModuleType, module2: ModuleType) -> ModuleType:
    assert len(module1) == len(module2), "Modules length does not match"

    jointModule = []
    for submodule1, submodule2 in zip(module1, module2):
        jointModule.append(submodule1 | submodule2)

    return jointModule


def module_indexer(module: ModuleType) -> IndexerType:
    indexer: IndexerType = IndexerType([], name=module[0]["class"])

    for submodule in module:
        indexer = indexer.append(submodule["indexer"])

    return indexer


def foreach_submodule(
    module: ModuleType,
    network: pypsa.Network,
    function: Callable[[SubmoduleType], float],
) -> DataArray:
    submoduleVariables: list[DataArray] = []

    for submodule in module:
        submoduleVariables.append(
            DataArray(
                [[function(submodule)] * submodule["units"]] * len(network.snapshots),
                coords={
                    "snapshot": network.snapshots,
                    submodule["class"]: submodule["indexer"],
                },
            )
        )

    return concat(submoduleVariables, dim=submodule["class"])


def sum_over_resource(reserve: Variable) -> LinearExpression:
    expr = linexpr((1, reserve))
    dims = list(reserve.sizes.keys())
    dims.remove("snapshot")
    return expr.sum(dims)


def filter_module_by_class(module: ModuleType, class_name: str) -> ModuleType:
    return list(filter(lambda x: x["class"] == class_name, module))


def define_snapshots(environment: StageType) -> IndexerType:
    START_DATE = "2021-01-01 00:00"

    # [WARNING] assert timeseries length equal to periods
    periods = len(environment["demand"][0]["loads"])

    snapshots = pd.date_range(
        start=START_DATE,
        periods=periods,
        freq="h",
        name="snapshot",
    )

    return snapshots


# SolutionType
def dump_solution(solution: Solution, directory: str):

    for pypsa_class, solution_dict in solution.solution.items():
        class_dir = directory + f"/{pypsa_class}"
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)

        for variable_name, solution_df in solution_dict.items():
            solution_df.to_csv(class_dir + f"/{variable_name}.csv")

    network_folder = directory + "/network"
    if not os.path.exists(network_folder):
        os.makedirs(network_folder)

    solution.network.export_to_csv_folder(network_folder)


def calc_objective(network: pypsa.Network) -> tuple[float, float]:
    tot_cc = 0
    tot_mc = 0

    tot_cc += network.stores["e_nom"].mul(network.stores["capital_cost"]).sum()
    tot_cc += network.links["p_nom"].mul(network.links["capital_cost"]).sum()
    tot_cc += +network.generators["p_nom"].mul(network.generators["capital_cost"]).sum()

    tot_mc += (
        network.links_t["p0"].mul(network.links["marginal_cost"]).sum().sum()
        * network.snapshot_weightings.iloc[0].iloc[0]
    )
    tot_mc += (
        network.generators_t["p"].mul(network.generators["marginal_cost"]).sum().sum()
        * network.snapshot_weightings.iloc[0].iloc[0]
    )
    tot_mc += (
        network.generators_t["status"]
        .mul(network.generators["stand_by_cost"])
        .sum()
        .sum()
        * network.snapshot_weightings.iloc[0].iloc[0]
    )
    tot_mc += (
        network.generators_t["status"]
        .mul(network.generators_t["stand_by_cost"])
        .sum()
        .sum()
        * network.snapshot_weightings.iloc[0].iloc[0]
    )
    return (tot_cc, tot_mc)
