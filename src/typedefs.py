from __future__ import annotations

import logging
import os
import pickle
import random
from typing import Callable, Any
import numpy as np
import pandas as pd
import pypsa

IndexerType = pd.Index
SubmoduleType = dict[str, Any]
ModuleType = list[SubmoduleType]
StageType = dict[str, ModuleType]
SolutionType = dict[str, dict[str, pd.DataFrame]]

logger = logging.getLogger(__name__)


class Solution:

    def __init__(
        self: Solution,
        network: pypsa.Network,
    ):
        self._capital = float("inf")
        self._marginal = float("inf")

        self.network = network
        if self.network is not None:
            self.network.model = None

    @staticmethod
    def random_solution() -> Solution:
        """
        Creates a Solution with random capital and marginal costs.

        This factory method generates a "shrunken" solution instance,
        meaning it does not contain a full PyPSA network object. Instead,
        it directly sets the cost components to random values, making it
        a lightweight way to create initial solutions for optimization.

        Returns:
            Solution: An instance with random fitness.
        """
        # Instantiate the class without a network
        sol = Solution(None)

        # Manually set the internal cost attributes to random floats.
        # Since self.network is None, the property getters will return
        # these values directly without recalculating.
        sol._capital = random.random()
        sol._marginal = 0

        return sol

    def norm(position: list[int]) -> Solution:
        # Instantiate the class without a network
        sol = Solution(None)

        # Manually set the internal cost attributes to random floats.
        # Since self.network is None, the property getters will return
        # these values directly without recalculating.
        sol._capital = np.linalg.norm(position)
        sol._marginal = 0

        # Assert that either position is non-zero or capital is zero
        if sol._capital == 0 and any(position):
            raise ValueError(
                "Capital is zero but position is non-zero. This should not happen."
            )

        return sol

    @property
    def capital(self: Solution) -> float:
        from src.utils import calc_objective

        if self.network is not None:
            (self._capital, _) = calc_objective(self.network)
        return self._capital

    @property
    def marginal(self: Solution) -> float:
        from src.utils import calc_objective

        if self.network is not None:
            (_, self._marginal) = calc_objective(self.network)
        return self._marginal

    @property
    def fitness(self: Solution) -> float:
        return self.capital + self.marginal

    @property
    def solution(self: Solution) -> SolutionType:
        if self.network is None:
            raise OptimizedOutException(
                "Cannot extract network pnl: network has been optimized out."
            )

        return {
            "Generator": self.network.pnl("Generator"),
            "Store": self.network.pnl("Store"),
            "Link": self.network.pnl("Link"),
        }

    def shrink(self: Solution):
        self.marginal
        self.capital
        self.network = None

    def copy(self: Solution) -> Solution:
        return pickle.loads(pickle.dumps(self))


INF_SOLUTION = Solution(None)
FitFunType = Callable[[StageType, StageType, IndexerType], Solution]


class OptError(RuntimeError):
    pass


class OptFailError(OptError):
    pass


class OptNonPromisingError(OptError):
    pass


class OptimizedOutException(RuntimeError):
    pass


class RedirectOutput:
    def __init__(self, filename: str = os.devnull, ignore: bool = False):
        self.filename = filename
        self.ignore = ignore

    def __enter__(self):
        if self.ignore:
            return
        if os.name == "posix":
            self.null_fds = [
                os.open(self.filename, os.O_RDWR | os.O_CREAT) for _ in range(2)
            ]
            self.save_fds = [os.dup(1), os.dup(2)]
            os.dup2(self.null_fds[0], 1)
            os.dup2(self.null_fds[1], 2)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.ignore:
            return
        if os.name == "posix":
            os.dup2(self.save_fds[0], 1)
            os.dup2(self.save_fds[1], 2)
            os.close(self.null_fds[0])
            os.close(self.null_fds[1])
