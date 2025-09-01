import copy
import pandas as pd
from src.typedefs import IndexerType, StageType


class Planner:

    def __init__(self, environment: StageType):
        self.planningTemplate = None
        self.indexToSubmodule: list[tuple[str, int]] = []

        self.__set_planningTemplate(environment)

        for moduleName, module in self.planningTemplate.items():
            for submoduleIndex, submodule in enumerate(module):
                if submodule["units"] is None:
                    self.indexToSubmodule.append((moduleName, submoduleIndex))

    def __set_planningTemplate(self, environment: StageType) -> StageType:
        planningTemplate = {}

        for moduleName, module in environment.items():
            skipModule: bool = len(module) <= 0 or "units" not in module[0]

            if skipModule:
                continue

            newModule = []
            for submodule in module:
                planningSubmodule = {
                    "modular": submodule["modular"],
                    "committable": submodule["committable"],
                    "class": submodule["class"],
                    "units": submodule["units"],
                    "indexer": None,
                }

                # Make p_nom visible for non-modular submodules
                if not submodule["modular"]:
                    if submodule["class"] == "Generator":
                        planningSubmodule |= {"p_nom": submodule["p_nom"]}
                    elif submodule["class"] == "Store":
                        planningSubmodule |= {
                            "e_nom": submodule["e_nom"],
                            "p_nom_ch": submodule["p_nom_ch"],
                            "p_nom_disch": submodule["p_nom_disch"],
                        }
                    else:
                        raise ValueError(
                            f"Non-modularity not supported for class {submodule['class']}"
                        )

                newModule.append(planningSubmodule)
            planningTemplate[moduleName] = newModule

        self.planningTemplate = planningTemplate

    @staticmethod
    def generate_indexer(
        planning: StageType, moduleName: str, submoduleIndex: int
    ) -> IndexerType:
        submodule = planning[moduleName][submoduleIndex]

        if submodule["modular"]:
            indexer = pd.Index(
                [
                    f"{moduleName}-{submoduleIndex}-{i}"
                    for i in range(submodule["units"])
                ],
                name=submodule["class"],
            )
        else:
            indexer = pd.Index(
                [f"{moduleName}-{submoduleIndex}"],
                name=submodule["class"],
            )

        return indexer

    @staticmethod
    def index_info(indexer: IndexerType) -> tuple[str, int]:
        (moduleName, submoduleIndex, _) = indexer[0].split("-")
        return moduleName, int(submoduleIndex)

    # Translate a position vector into a planning stage
    def translate(self, position: list[int]) -> StageType:
        assert len(position) == len(
            self.indexToSubmodule
        ), "Length of position does not match its expected size"

        planning = copy.deepcopy(self.planningTemplate)

        # Fill planning with modular and non-modular units
        for index, element in enumerate(position):
            (moduleName, submoduleIndex) = self.indexToSubmodule[index]
            submodule = planning[moduleName][submoduleIndex]
            submodule["units"] = element

        # Instantiate indexers for each submodule
        for moduleName, module in planning.items():
            for submoduleIndex, submodule in enumerate(module):

                # Adjust for non modularity
                if not submodule["modular"]:
                    if submodule["class"] == "Generator":
                        submodule["p_nom"] = submodule["units"] * submodule["p_nom"]
                    elif submodule["class"] == "Store":
                        submodule["e_nom"] = submodule["units"] * submodule["e_nom"]
                        submodule["p_nom_ch"] = (
                            submodule["units"] * submodule["p_nom_ch"]
                        )
                        submodule["p_nom_disch"] = (
                            submodule["units"] * submodule["p_nom_disch"]
                        )
                    submodule["units"] = 1

                submodule["indexer"] = Planner.generate_indexer(
                    planning, moduleName, submoduleIndex
                )

        return planning

    # Translate a planning stage into a position vector
    def antitranslate(self, stage: StageType, key: str) -> list:
        values = []

        for moduleName, submoduleIndex in self.indexToSubmodule:
            assert (
                key in stage[moduleName][submoduleIndex]
            ), f"missing key {key} in plannable submodule {moduleName}-{submoduleIndex}"

            values.append(stage[moduleName][submoduleIndex][key])

        return values

    def create_linear_constraint(
        self,
        environment: StageType,
        lhs_template: list[tuple[int, str]],
        sign: str,
        rhs: int,
    ):
        from src.pso.particle import Particle

        assert sign in [">", "<", "==", ">=", "<=", "!="], "Invalid sign"
        assert type(rhs) in [
            int,
            float,
        ], "Expressions right-hand side (rhs) must be a number"

        lhs: list[tuple[int, int]] = []
        names = self.antitranslate(environment, "name")

        for coeff, name in lhs_template:
            coord = names.index(name)
            lhs.append((coeff, coord))

        def linear_constraint(particle: Particle):
            sum = 0
            for coeff, coord in lhs:
                sum += coeff * round(particle.position[coord])
            return eval(f"sum {sign} {rhs}")

        return linear_constraint

    def get_technology_names(self) -> list[str]:
        """
        Returns a list of technology names in the same order as current_sample.
        Each name is formatted as 'moduleName_submoduleIndex'.
        """
        return [
            f"{moduleName}_{submoduleIndex}"
            for moduleName, submoduleIndex in self.indexToSubmodule
        ]

    def num_dimensions(self) -> int:
        return len(self.indexToSubmodule)

    def __str__(self):
        return str(self.indexToSubmodule)
