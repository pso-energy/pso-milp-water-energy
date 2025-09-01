import os
import pandas as pd
from linopy.common import as_dataarray, DataArray
from src.typedefs import ModuleType, StageType
import sqlite3
import json
import numpy as np


def load_module(cursor: sqlite3.Cursor, moduleName: str) -> ModuleType:
    MODULE_SERIES_MAP = {
        "vres": "cf",
        "demand": "loads",
        "desalinator": "stand_by_cost",
    }
    TABLE_NAMES_MAP = {"l_c": "ch", "l_d": "disch"}

    def suffixate_link_properties(table_name: str):
        assert table_name in TABLE_NAMES_MAP

        short_name = TABLE_NAMES_MAP[table_name]
        table_properties = ["name", "cc", "oc", "p_nom", "eta", "p_min_pu", "p_max_pu"]

        remappings = [
            f"{table_name}.{table_property} AS {table_property}_{short_name}"
            for table_property in table_properties
        ]

        return ",".join(remappings)

    query = f"SELECT m.* FROM {moduleName} m"
    needs_link_join = (
        dict(
            cursor.execute(
                f"SELECT COUNT(*) AS CNTREC FROM pragma_table_info('{moduleName}') WHERE name='link_charge' OR name='link_discharge'"
            ).fetchone()
        )["CNTREC"]
        > 0
    )
    needs_resource_join = (
        dict(
            cursor.execute(
                f"""
                    SELECT COUNT(*) AS CNTREC
                    FROM pragma_table_info('{moduleName}')
                    WHERE name='timeseries'
                """
            ).fetchone()
        )["CNTREC"]
        > 0
    )
    assert not (
        needs_resource_join and needs_link_join
    ), "We cannot handle both resource and link join"
    if needs_resource_join:
        query = f"""
            SELECT m.*, t.`values` AS series_values
            FROM {moduleName} m, timeseries t
            WHERE m.timeseries = t.id
        """
    if needs_link_join:
        query = f"""
            SELECT m.*, {suffixate_link_properties('l_d')}, {suffixate_link_properties('l_c')}
            FROM {moduleName} m, link l_c, link l_d
            WHERE m.link_charge = l_c.id
            AND m.link_discharge = l_d.id
        """
    query += " ORDER BY m.id"
    res = cursor.execute(query)
    module = []
    for row in res.fetchall():
        submodule = dict(row)
        if "series_values" in submodule:
            submodule[MODULE_SERIES_MAP.get(moduleName, "values")] = np.array(
                # we take 1 week (24 hours * 7 days) every 4 weeks
                sample(json.loads(submodule["series_values"]), 24 * 7, 4)
            )
            del submodule["series_values"]
            del submodule["timeseries"]
        module.append(submodule)
    return module


def sample(list: list, width: int, period: int, phase: int = 0) -> list:
    """
    This function takes `width` elements of a list every `width*period` elements. Optionally, we can specify a phase.
    """
    return [list[i] for i in range(len(list)) if (i // width) % period == phase]


def load_environment(filename: str, moduleNames: list[str]) -> StageType:
    environment = dict()

    assert os.path.exists(filename), f"The database file '{filename}' does not exist."

    connection = sqlite3.connect(filename)
    connection.row_factory = sqlite3.Row
    cursor = connection.cursor()

    for moduleName in moduleNames:
        module = load_module(cursor, moduleName)
        environment[moduleName] = module

    return environment


def set_snapshots(x, snapshots) -> DataArray:
    return as_dataarray(x, coords=[snapshots])


def init_dimension(name: str, attr: str, number: int) -> pd.Index:
    return pd.Index([f"{attr}-{i}" for i in range(number)], name=name)
