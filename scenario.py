import logging
from src import milp, pso
from src.data_import import load_environment
from src.pso.planner import Planner
from src.pso.pso import dump_solution
from src.utils import define_snapshots
from dotenv import load_dotenv

logger = logging.getLogger(__name__)


def run_milp(scenario: str, position: list[int]):
    assert len(position) == 10, "Invalid position length"
    environment = load_environment(
        scenario,
        moduleNames=[
            "ffg",
            "bess",
            "electrolyzer",
            "desalinator",
            "vres",
            "demand",
            "modules",
        ],
    )

    planner = Planner(environment)
    snapshots = define_snapshots(environment)

    planning = planner.translate(position)

    print("Launching MILP...")
    solution = milp.launch(environment, planning, snapshots)

    print(f"capital: {solution.capital:.2e}")
    print(f"marginal: {solution.marginal:.2e}")
    dump_solution(solution, "log/solution/")


def run_pso(path: str, scenario: str, initial_positions: list[list[int]]) -> list[int]:
    environment = load_environment(
        path,
        moduleNames=[
            "ffg",
            "bess",
            "electrolyzer",
            "desalinator",
            "vres",
            "demand",
            "modules",
        ],
    )

    (best_position, _) = pso.launch(
        scenario,
        environment,
        concurrency=4,
        num_particles=28,
        max_iterations=50,
        iteration_tolerance=5,
        inertia_weight=0.7,
        cognitive_weight=1.2,
        social_weight=2.0,
        initial_positions=initial_positions,
    )

    return best_position


def run_scenario(
    path, scenario: str, initial_positions: list[list[int]] = []
) -> list[int]:
    load_dotenv()

    logger.info(f"Starting {scenario}")
    best_position = run_pso(path, scenario, initial_positions)

    return best_position
