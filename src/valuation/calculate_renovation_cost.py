"""Renovation cost calculation API."""

try:
    from .config import (
        CONDITIONS_TABLE,
        NON_RENOVATABLE_CONDITION,
        PACKAGES_TABLE,
        PACKAGE_WORKS_TABLE,
        RENOVATION_MULTIPLIER,
        WORK_COST_COLUMN,
        WORK_IDS_COLUMN,
        WORK_NAME_COLUMN,
        WORKS_TABLE,
    )
    from .core import read_table
except ImportError:
    from config import (
        CONDITIONS_TABLE,
        NON_RENOVATABLE_CONDITION,
        PACKAGES_TABLE,
        PACKAGE_WORKS_TABLE,
        RENOVATION_MULTIPLIER,
        WORK_COST_COLUMN,
        WORK_IDS_COLUMN,
        WORK_NAME_COLUMN,
        WORKS_TABLE,
    )
    from core import read_table


def _load_renovation_tables():
    try:
        return {
            "conditions": read_table(CONDITIONS_TABLE),
            "packages": read_table(PACKAGES_TABLE),
            "package_works": read_table(PACKAGE_WORKS_TABLE),
            "works": read_table(WORKS_TABLE),
        }
    except Exception:
        return None


def _empty_result(renovation_possible):
    return {
        "total_cost": 0,
        "package_ids": [],
        "works": [],
        "renovation_possible": renovation_possible,
    }


def _find_package(packages_df, from_condition, to_condition):
    package_match = packages_df[
        (packages_df["from"] == from_condition)
        & (packages_df["to"] == to_condition)
    ]

    if package_match.empty:
        return None

    return package_match.iloc[0]["Kulcs"]


def _work_ids_for_package(package_works_df, package_id):
    return package_works_df[
        package_works_df["Csomag"] == package_id
    ][WORK_IDS_COLUMN].tolist()


def _works_for_ids(works_df, work_ids):
    selected_works = works_df[works_df["Index"].isin(work_ids)].copy()
    selected_works[WORK_COST_COLUMN] = (
        selected_works[WORK_COST_COLUMN]
        .astype(str)
        .str.replace(",", ".")
        .astype(float)
    )
    return selected_works


def _calculate_work_costs(selected_works, building_area_m2):
    selected_works["total_work_cost"] = (
        selected_works[WORK_COST_COLUMN] * building_area_m2
    )
    return selected_works


def calculate_renovation_cost(
    current_condition,
    target_condition,
    building_area_m2,
):
    if current_condition == NON_RENOVATABLE_CONDITION:
        return _empty_result(renovation_possible=False)

    if current_condition >= target_condition:
        return _empty_result(renovation_possible=True)

    tables = _load_renovation_tables()
    if tables is None:
        return _empty_result(renovation_possible=True)

    packages_df = tables["packages"]
    package_works_df = tables["package_works"]
    works_df = tables["works"]

    total_cost = 0
    collected_works = []
    package_ids = []

    for condition_step in range(current_condition, target_condition):
        from_condition = condition_step
        to_condition = condition_step + 1
        package_id = _find_package(packages_df, from_condition, to_condition)

        if package_id is None:
            continue

        package_ids.append(int(package_id))
        work_ids = _work_ids_for_package(package_works_df, package_id)
        selected_works = _works_for_ids(works_df, work_ids)
        selected_works = _calculate_work_costs(selected_works, building_area_m2)

        total_cost += selected_works["total_work_cost"].sum()
        collected_works.extend(selected_works[WORK_NAME_COLUMN].tolist())

    collected_works = list(set(collected_works))
    total_cost *= RENOVATION_MULTIPLIER

    return {
        "total_cost": round(total_cost, 0),
        "package_ids": package_ids,
        "works": collected_works,
        "renovation_possible": True,
    }


if __name__ == "__main__":
    result = calculate_renovation_cost(
        current_condition=2,
        target_condition=5,
        building_area_m2=100,
    )

    print(result)
