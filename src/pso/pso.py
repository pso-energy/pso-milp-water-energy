from concurrent.futures import ProcessPoolExecutor, as_completed
import logging
from multiprocessing import get_context
import os
from random import random
import numpy as np
import psutil
import signal
import sys

from src.logger import setup_logger
from src.telegram import TelegramNotifier
from src.pso.particle import Particle
from src.pso.planner import Planner
from src.typedefs import INF_SOLUTION, Solution, StageType
from src.utils import dump_solution

from multiprocessing import Manager

logger = logging.getLogger(__name__)

# Create a global Manager dict before starting the pool
_manager = None
_particle_cache = None


def init_cache():
    """Initialize the cache manager"""
    global _manager, _particle_cache
    if _manager is None:
        _manager = Manager()
        _particle_cache = _manager.dict()


def cleanup_cache():
    """Clean up the cache manager"""
    global _manager, _particle_cache
    if _particle_cache is not None:
        _particle_cache.clear()
    if _manager is not None:
        _manager.shutdown()
        _manager = None
        _particle_cache = None


def evaluate(particle: Particle, environment: StageType, planner: Planner) -> Particle:
    setup_logger()

    # Initialize cache if not already done
    global _particle_cache
    if _particle_cache is None:
        init_cache()

    solution = particle.evaluate_fitness(environment, planner, _particle_cache)
    particle.update_solution(solution)

    return particle


def pso(
    environment: StageType,
    planner: Planner,
    num_particles: int,
    max_iterations: int,
    inertia_weight: float,
    cognitive_weight: float,
    social_weight: float,
    iteration_tolerance: int,
    tolerance: float,
    concurrency: int,
    notifier: TelegramNotifier,
    initial_positions: list[list[float]] = None,
) -> tuple[list[int], Solution]:

    # Initialize cache for this run
    init_cache()

    # Reset the cache for each run
    global _particle_cache
    if _particle_cache is not None:
        _particle_cache.clear()

    assert initial_positions is None or len(initial_positions) <= num_particles

    num_dimensions = planner.num_dimensions()
    lower_bounds: list[int] = planner.antitranslate(environment, "lower")
    upper_bounds: list[int] = planner.antitranslate(environment, "upper")

    particles: list[Particle] = []

    # Initialize particles to initial positions
    for i in range(len(initial_positions)):
        particle = Particle(
            i,
            num_dimensions,
            lower_bounds,
            upper_bounds,
            inertia_weight,
            cognitive_weight,
            social_weight,
            initial_position=initial_positions[i],
        )
        particles.append(particle)

    # Initialize the remaining particles at random position
    for i in range(num_particles - len(initial_positions)):
        particle = Particle(
            i + len(initial_positions),
            num_dimensions,
            lower_bounds,
            upper_bounds,
            inertia_weight,
            cognitive_weight,
            social_weight,
        )
        particles.append(particle)

    # Initialize pso global best state variables
    best_position = particles[0].position.copy()
    best_solution = INF_SOLUTION

    best_fitness_history = []

    executor = None
    try:
        for i in range(max_iterations):

            logger.info(f"Iteration {i + 1}/{max_iterations}")
            if i % 3 == 0:
                notifier.send(f"Reached iteration {i}")

            # This avoids forking the entire process
            context = get_context("spawn")

            # Evaluate particles in parallel
            executor = ProcessPoolExecutor(concurrency, mp_context=context)

            try:
                futures = [
                    executor.submit(evaluate, p, environment, planner)
                    for p in particles
                ]

                for future in as_completed(futures):
                    particle = future.result()
                    particles[particle.id] = particle

                    # Log particle position and fitness after evaluation
                    logger.info(
                        f"Particle {particle.id} Position: {[round(x, 5) for x in particle.position]} Fitness: {particle.current_solution.fitness}"
                    )

                    # Relaxation of global best
                    if particle.best_solution.fitness < best_solution.fitness:
                        best_position = particle.best_position.copy()
                        best_solution = particle.best_solution.copy()

                    # Reduce memory occupation for each particle
                    particle.best_solution.shrink()
                    particle.current_solution.shrink()

            finally:
                # Always shutdown the executor properly
                executor.shutdown(wait=False)
                executor = None

            # Update particles' best positions
            best_fitness_history.append(best_solution.fitness)

            # Log particle data to CSV
            log_particle_data(i + 1, particles, "log/")

            # Early exit if the best solution is already good enough
            if i >= iteration_tolerance:
                improvement = (
                    abs(
                        best_fitness_history[i]
                        - best_fitness_history[i - iteration_tolerance]
                    )
                    / best_fitness_history[i - iteration_tolerance]
                )
                if improvement < tolerance:
                    logger.info(
                        f"Converged on iteration {i} because improvement was less than tolerance ({tolerance}) in {iteration_tolerance} iterations"
                    )
                    notifier.send(
                        f"Converged on iteration {i} because improvement was less than tolerance ({tolerance}) in {iteration_tolerance} iterations"
                    )
                    break

            # Move the particles
            for particle in particles:
                particle.step(best_position)

    except KeyboardInterrupt:
        logger.info("PSO interrupted by user")
        notifier.send("PSO execution interrupted by user")

        # Clean shutdown of executor if it exists
        if executor is not None:
            logger.info("Shutting down executor...")
            executor.shutdown(wait=False)

        # Clean up cache
        cleanup_cache()

        # Re-raise the interrupt to allow proper cleanup at higher levels
        raise

    except Exception as e:
        logger.error(f"PSO failed with error: {e}")

        # Clean shutdown of executor if it exists
        if executor is not None:
            logger.info("Shutting down executor due to error...")
            executor.shutdown(wait=False)

        # Clean up cache
        cleanup_cache()

        # Re-raise the exception
        raise

    finally:
        # Always clean up cache when function exits
        cleanup_cache()

    # Round the best position to integers
    best_position = [round(x) for x in best_position]

    dump_best_fitness_history(best_fitness_history, "log/")

    return best_position, best_solution


class NoKeyFakeNotifier:
    def __init__(self):
        pass

    def send(self, msg):
        pass


def launch(
    scenario: str,
    environment: StageType,
    num_particles: int = 10,
    max_iterations: int = 30,
    inertia_weight: float = 0.9,
    cognitive_weight: float = 2.0,
    social_weight: float = 2.0,
    iteration_tolerance: int = 5,
    tolerance: float = 0.001,
    concurrency: int = None,
    initial_positions: list[list[float]] = None,
) -> tuple[list[int], Solution]:

    # Set up signal handler for graceful shutdown
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        cleanup_cache()
        sys.exit(1)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    token = os.environ.get("TOKEN")
    chat_id = os.environ.get("CHAT_ID")

    notifier = TelegramNotifier(token, parse_mode="HTML", chat_id=chat_id)
    notifier.send(f"<b> Starting simulation: {scenario} </b>")

    planner = Planner(environment)

    # Prepare PSO constraints
    # lhs = [(1, "des1"), (1, "des2"), (1, "des3"), (1, "des4"), (1, "des5")]
    # des_constraint = planner.create_linear_constraint(environment, lhs, "<=", 5)

    # Set concurrency automatically
    if concurrency is None:
        resources = cpu_count()
        if resources is None:
            raise RuntimeError("Cannot get number of CPUs, fix your PC")
        concurrency = min(resources, num_particles)

    logger.info(f"num_particles={num_particles}, max_iterations={max_iterations}")
    logger.info(
        f"inertia_weight={inertia_weight}, cognitive_weight={cognitive_weight}, social_weight={social_weight}"
    )
    logger.info(f"iteration_tolerance={iteration_tolerance}, tolerance={tolerance}")

    try:
        (best_position, best_solution) = pso(
            environment,
            planner,
            num_particles,
            max_iterations,
            inertia_weight,
            cognitive_weight,
            social_weight,
            iteration_tolerance,
            tolerance,
            concurrency,
            notifier=notifier,
            initial_positions=initial_positions,
        )

        # Log unit commitment
        dump_solution(best_solution, "log/solution")

        notifier.send(
            f"fitness: {best_solution.fitness} = marginal: {best_solution.capital} + capital: {best_solution.marginal}"
        )

        return best_position, best_solution

    except KeyboardInterrupt:
        logger.info("Launch interrupted by user")
        notifier.send("Simulation interrupted by user")
        cleanup_cache()
        raise

    except Exception as e:
        logger.error(f"Launch failed with error: {e}")
        notifier.send(f"Simulation failed with error: {e}")
        cleanup_cache()
        raise


def dump_best_fitness_history(best_fitness_history: list[float], directory: str):
    import csv

    indexed_float_list = [(i, value) for i, value in enumerate(best_fitness_history)]

    with open(directory + "/best_fitness_history.csv", "w", newline="") as file:
        writer = csv.writer(file, delimiter=",")
        writer.writerow(["Iteration", "Fitness"])
        writer.writerows(indexed_float_list)


def log_particle_data(iteration: int, particles: list[Particle], directory: str):
    """Log particle data to CSV file for each iteration"""
    import csv
    import os

    csv_file = os.path.join(directory, "particles_log.csv")

    # Check if file exists to determine if we need to write headers
    file_exists = os.path.exists(csv_file)

    with open(csv_file, "a", newline="") as file:
        writer = csv.writer(file, delimiter=",")

        # Write headers only if file doesn't exist
        if not file_exists:
            # Create headers: Iteration, ParticleID, Fitness, Position_0, Position_1, ..., Position_n
            headers = ["Iteration", "ParticleID", "Fitness"]
            if particles:
                num_dimensions = len(particles[0].position)
                headers.extend([f"Position_{i}" for i in range(num_dimensions)])
            writer.writerow(headers)

        # Write data for each particle
        for particle in particles:
            row = [iteration, particle.id, particle.current_solution.fitness]
            row.extend(particle.position)
            writer.writerow(row)
