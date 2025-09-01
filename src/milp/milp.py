import os
import gurobipy as gp
import psutil
from src.milp import water, reserve, generator
from src.milp.constants import WATER_TANK_SIZE
from src.milp.modules import ffg, bess, vres, desalinator, electrolyzer, load
from src.pso.particle import Particle
from src.typedefs import (
    OptNonPromisingError,
    Solution,
    StageType,
    IndexerType,
    OptFailError,
    RedirectOutput,
)
import pypsa
from linopy import Variable
from src.utils import calc_objective, filter_module_by_class


def launch(
    environment: StageType,
    planning: StageType,
    snapshots: IndexerType,
    particle: Particle,
) -> Solution:
    network: pypsa.Network = init_network(environment, planning, snapshots)

    def extra_functionality(network: pypsa.Network, snapshots: IndexerType):

        do_reserve: dict[str, bool] = {
            module["name"]: module["do_reserve"] for module in environment["modules"]
        }

        modules = [
            ("ffg", ffg),
            ("bess", bess),
            ("desalinator", desalinator),
            ("electrolyzer", electrolyzer),
        ]

        # Loop through the modules
        reserves: list[Variable] = []
        for module_name, module in modules:
            if do_reserve[module_name]:
                r = module.define_reserve(network, environment, planning)
                reserves.append(r)

        reserve.define_reserve_requirement_constraint(
            network, environment, planning, reserves
        )

        # Water definition
        water_module = filter_module_by_class(environment["demand"], "WaterLoad")
        assert len(water_module) == 1, "No support for more than one water module"
        water_submodule = water_module[0]

        w_d = desalinator.define_water_production(network, environment, planning)
        w_e = electrolyzer.define_water_production(network, environment, planning)
        water.define_global_water_tank(
            network,
            WATER_TANK_SIZE,
            water_submodule["loads"] * water_submodule["scale"],
            w_d,
            w_e,
        )

        generator.define_desalinator_sbc_constraint(
            network, environment, planning, "desalinator"
        )

    (capital, _) = calc_objective(network)

    def gurobi_callback(model, where):
        if where == gp.GRB.Callback.MIP:
            current_bound = model.cbGet(gp.GRB.Callback.MIP_OBJBND)

            score = (
                (particle.best_fitness - (current_bound + capital))
                if particle.best_fitness is not None
                else 1
            )
            if score < 0:
                model.terminate()

    problem_fn = f"./output/part{particle.id:03}.lp"
    log_fn = f"./log/particles/part{particle.id:03}.log"

    # Memory limit set to 90/100 of total available memory
    # Expressed in GiB (bytes / (1024*1024*1024))
    memory_limit = (
        (90 / 100)
        * (psutil.virtual_memory().total + psutil.swap_memory().total)
        / (1024 * 1024 * 1024)
        / os.cpu_count()
    )

    solver_options = (
        {
            "MIPGap": 0.02,
            "TimeLimit": 3600 * 4,
            # "PreSparsify": 2,
            "callback": gurobi_callback,
            "NoRelHeurTime": 60,
        }
        | ({"SoftMemLimit": memory_limit})
        | ({"NodeFileStart": memory_limit * 3 / 4})
    )

    with RedirectOutput():
        try:
            _, condition = network.optimize(
                solver_name="gurobi",
                problem_fn=problem_fn,
                keep_files=True,
                log_fn=log_fn,
                solver_options=solver_options,
                extra_functionality=extra_functionality,
            )
        # This is thrown if no solution is found before the time limit expires
        except AttributeError:
            raise OptFailError(f"No solution found before time or mem limit expiration")

    if condition == "optimal":
        pass
    elif condition == "user_interrupt":
        raise OptNonPromisingError(f"Non-promising particle")
    elif condition == "time_limit":
        logger.warning(
            f"Particle {particle.id} reached time limit without finding a solution"
        )
    elif condition == "internal_solver_error":
        logger.error(
            f"Internal solver error for particle {particle.id}. Check the log file."
        )
    else:
        raise OptFailError("Infeasible problem")

    solution = Solution(network)

    if solution.marginal < 0:
        raise OptFailError(f"Negative marginal cost: {solution.marginal}")
    elif solution.capital < 0:
        raise OptFailError(f"Negative capital cost: {solution.capital}")

    return solution

# REDACTED
# If you need access to the definition of the PyPSA network,
# contact the authors
def init_network(
    environment: StageType, planning: StageType, snapshots: IndexerType
) -> pypsa.Network:
    network = pypsa.Network()

    raise RuntimeError("Network initialization redacted, contact the authors to get access")

    return network
