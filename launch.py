import itertools
from src.logger import setup_logger
import json
import logging
import sys
import zipfile
import datetime
import os
from scenario import run_scenario

logger = logging.getLogger(__name__)

TAGS = ["x1", "x1_5", "x2"]
SCENARIOS = [
    "FULL",
    "ffgReserveOnly",
    "WaterAsLoad",
]

COMPLETED_SCENARIOS_PATH = "completed_scenarios.json"


def zipdir(path: str, ziph: zipfile.ZipFile):
    for root, _, files in os.walk(path):
        for file in files:
            ziph.write(
                os.path.join(root, file),
                os.path.relpath(os.path.join(root, file), os.path.join(path, "..")),
            )


def delete_files(directory):
    for path, _, files in os.walk(directory):
        for file in files:
            os.remove(os.path.join(path, file))
            print(f"Deleted file: {os.path.join(path, file)}")


def store_completed_scenarios(
    filename: str, scenarios: dict[str, dict[str, list[int]]]
):
    with open(filename, "w") as outfile:
        json.dump(scenarios, outfile)


def load_completed_scenarios(filename: str) -> dict[str, dict[str, list[int]]]:
    with open(filename, "r") as infile:
        scenarios = json.load(infile)

    assert scenarios is not None, f"[LAUNCH] Expected a valid dictionary in {filename}"
    return scenarios


def dummy_run_scenario(x, y, z) -> list[int]:
    print("[DUMMY RUN SCENARIO]")
    return [1, 2, 3, 4]


def launch():

    scenarios_completed = load_completed_scenarios(COMPLETED_SCENARIOS_PATH)

    for tag in TAGS:

        scenarios_to_run = filter(
            lambda x: x not in scenarios_completed.get(tag, {}).keys(), SCENARIOS
        )
        initial_positions = list(scenarios_completed.get(tag, {}).values())
        for scenario in scenarios_to_run:

            print("[LAUNCH] Delete logs...")
            delete_files("log")

            path = f"data/water_{tag}/{scenario}.db"
            print(f"[LAUNCH] Scenario {path}")

            setup_logger()
            best_position = run_scenario(path, scenario, initial_positions)
            # best_position = dummy_run_scenario("a", "b", initial_positions)

            print("[LAUNCH] Compressing logs...")
            timestamp = (
                datetime.datetime.now().isoformat(timespec="minutes").replace(":", "-")
            )
            path = os.path.join("simulations", f"{timestamp}_{scenario}_{tag}.zip")
            with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
                zipdir("data", archive)
                zipdir("log", archive)
                zipdir("src", archive)

            print("[LAUNCH] Delete logs...")
            delete_files("log")

            initial_positions.append(best_position)

            # python on fire
            temp = scenarios_completed.setdefault(tag, {})
            temp[scenario] = best_position
            store_completed_scenarios(COMPLETED_SCENARIOS_PATH, scenarios_completed)


if __name__ == "__main__":

    launch()
