import logging
import random
from typing import Callable
import numpy as np

from src import milp
from src.pso.planner import Planner
from src.typedefs import INF_SOLUTION, OptError, Solution, StageType
from src.utils import define_snapshots

logger = logging.getLogger(__name__)


class Particle:
    ConstraintFunType = Callable[[], bool]

    def __init__(
        self,
        id: int,
        num_dimensions: int,
        lower_bounds: list[float],
        upper_bounds: list[float],
        inertia_weight: float,
        cognitive_weight: float,
        social_weight: float,
        starting_velocity_coefficient: float = 0.1,
        initial_position: list[float] = None,
    ):
        """
        Parameters:
            num_dimensions: number of variables to optimize, i.e. space in which
                particles move
            lower_bound: lower bounds for each dimension
            upper_bound: upper bounds for each dimension
        """

        assert len(lower_bounds) == num_dimensions, "Wrong number of lower bounds"
        assert len(upper_bounds) == num_dimensions, "Wrong number of upper bounds"

        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds

        self.cognitive_weight = cognitive_weight
        self.social_weight = social_weight
        self.inertia_weight = inertia_weight

        self.id = id
        if not initial_position:
            self.position = [
                random.uniform(lower, upper)
                for lower, upper in zip(lower_bounds, upper_bounds)
            ]
        else:
            assert (
                len(initial_position) == num_dimensions
            ), "Wrong number of dimensions for initial position"
            self.position = initial_position

        max_starting_velocity = [
            starting_velocity_coefficient * (upper - lower)
            for lower, upper in zip(lower_bounds, upper_bounds)
        ]
        self.velocity = [
            random.uniform(0, v) * random.choice([-1, 1]) for v in max_starting_velocity
        ]

        self.current_solution = INF_SOLUTION

        self.best_position = self.position.copy()
        self.best_solution = INF_SOLUTION

    def update_velocity(self, global_best_position: list[float]):
        """
        Parameters:
            global_best_position: position of the particle inside the space
            inertia_weight: weigth of the particle (i.e. the particle continues to move in the same direction)
            cognitive_weight: wieght of the particle's own best position (i.e. the particle moves towards its own best position)
            social_weight: wieght of the global best position (i.e. the particle moves towards the global best position)
        """

        for i in range(len(self.velocity)):
            r1 = random.random()
            r2 = random.random()
            cognitive_component = (
                self.cognitive_weight * r1 * (self.best_position[i] - self.position[i])
            )
            social_component = (
                self.social_weight * r2 * (global_best_position[i] - self.position[i])
            )
            self.velocity[i] = (
                self.inertia_weight * self.velocity[i]
                + cognitive_component
                + social_component
            )

            # Clamp velocity
            max_velocity = abs(self.upper_bounds[i] - self.lower_bounds[i]) * 0.5
            self.velocity[i] = Particle.confine_in_bounds(
                self.velocity[i], -max_velocity, max_velocity
            )

    @staticmethod
    def confine_in_bounds(x: float, lower: float, upper: float) -> float:
        return max(min(upper, x), lower)

    def update_position(self):
        for i in range(len(self.position)):
            self.position[i] += self.velocity[i]
            self.position[i] = Particle.confine_in_bounds(
                self.position[i], self.lower_bounds[i], self.upper_bounds[i]
            )

    def update_solution(self, solution: Solution):
        """
        Updates the particle's best solution if the new solution is better.
        If the new solution is better than the current solution, it updates both
        the current and the best solution.
        """
        self.current_solution = solution

        # Relaxation of local best
        if solution.fitness < self.best_solution.fitness:
            self.best_position = self.position.copy()
            self.best_solution = solution

    def evaluate_fitness(
        self,
        environment: StageType,
        planner: Planner,
        _particle_cache: dict[tuple, Solution] = {},
    ) -> Solution:
        """
        Parameters:
            fitness_function: function that maps each position in the search space to its cost
            environment: internal parameters of the fitness function
            structure: form of the stage in which the data taken from the search space is organized
        """

        # Round the position to the nearest integer, with a small random perturbation
        # to avoid local minima traps. This is a common practice in PSO for discrete problems
        # where the position represents discrete variables.
        rounded_position = [
            int(np.ceil(x)) if random.random() < (x - int(x)) else int(np.floor(x))
            for x in self.position
        ]

        # Setup the planning with the current position and initialize the snapshots
        planning = planner.translate(rounded_position)
        snapshots = define_snapshots(environment)

        position_key = tuple(rounded_position)

        if position_key in _particle_cache:
            logger.info(f"Using cached fitness for particle {self.id}")
            solution = _particle_cache[position_key]
        else:
            try:
                # solution = milp.launch(environment, planning, snapshots, particle=self)
                solution = Solution.norm(rounded_position)

                _particle_cache[position_key] = solution
            except OptError as err:
                solution = INF_SOLUTION
                logger.error(f"Particle {self.id} failed to find a solution: {err}")

        return solution

    def step(
        self,
        global_best_position: list[float],
    ):
        """
        Performs a single step of the particle's movement in the search space.
        Updates the particle's velocity and position, evaluates its fitness,
        """

        self.update_velocity(global_best_position)
        self.update_position()

    def log_velocity(self):
        logger.info(
            f"Particle {self.id} velocity: {[round(x, 5) for x in self.velocity]}"
        )

    def log_position(self):
        logger.info(
            f"Particle {self.id} Position: {[round(x, 5) for x in self.position]} Fitness: {self.current_solution.fitness}"
        )
