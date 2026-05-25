import pandas as pd

from sqlalchemy import create_engine

# =====================================================
# DATABASE CONNECTION
# =====================================================

engine = create_engine(
    "postgresql+psycopg2://postgres:postgres@localhost:5432/real_estate"
)

# =====================================================
# LOAD TABLES
# =====================================================

conditions_df = pd.read_sql(
    "SELECT * FROM conditions",
    engine
)

packages_df = pd.read_sql(
    "SELECT * FROM packages",
    engine
)

package_works_df = pd.read_sql(
    "SELECT * FROM package_works",
    engine
)

works_df = pd.read_sql(
    "SELECT * FROM works",
    engine
)

print("Tables loaded successfully!")

# =====================================================
# CALIBRATION
# =====================================================

RENOVATION_MULTIPLIER = 4

# =====================================================
# RENOVATION COST FUNCTION
# =====================================================

def calculate_renovation_cost(
    current_condition,
    target_condition,
    building_area_m2
):

    # =================================================
    # SKIP NON-RENOVATABLE PROPERTIES
    # =================================================

    if current_condition == 1:

        return {
            "total_cost": 0,
            "package_ids": [],
            "works": [],
            "renovation_possible": False
        }

    # =================================================
    # NO RENOVATION NEEDED
    # =================================================

    if current_condition >= target_condition:

        return {
            "total_cost": 0,
            "package_ids": [],
            "works": [],
            "renovation_possible": True
        }

    # =================================================
    # INITIALIZE
    # =================================================

    total_cost = 0

    collected_works = []

    package_ids = []

    # =================================================
    # LOOP THROUGH CONDITION STEPS
    # =================================================

    for condition_step in range(
        current_condition,
        target_condition
    ):

        from_condition = condition_step

        to_condition = condition_step + 1

        # =============================================
        # FIND PACKAGE
        # =============================================

        package_match = packages_df[
            (
                packages_df["from"]
                == from_condition
            )
            &
            (
                packages_df["to"]
                == to_condition
            )
        ]

        if package_match.empty:

            continue

        package_id = (
            package_match.iloc[0]["Kulcs"]
        )

        package_ids.append(
            int(package_id)
        )

        # =============================================
        # FIND WORK IDS
        # =============================================

        work_ids = package_works_df[
            package_works_df["Csomag"]
            == package_id
        ]["Munkák"].tolist()

        # =============================================
        # FILTER WORKS
        # =============================================

        selected_works = works_df[
            works_df["Index"].isin(work_ids)
        ].copy()

        # =============================================
        # CLEAN COST COLUMN
        # =============================================

        selected_works[
            "Med Ft/nm (2024)"
        ] = (
            selected_works[
                "Med Ft/nm (2024)"
            ]
            .astype(str)
            .str.replace(",", ".")
            .astype(float)
        )

        # =============================================
        # CALCULATE WORK COSTS
        # =============================================

        selected_works["total_work_cost"] = (
            selected_works[
                "Med Ft/nm (2024)"
            ]
            * building_area_m2
        )

        # =============================================
        # AGGREGATE COST
        # =============================================

        step_cost = (
            selected_works[
                "total_work_cost"
            ].sum()
        )

        total_cost += step_cost

        # =============================================
        # COLLECT WORK NAMES
        # =============================================

        collected_works.extend(
            selected_works[
                "Munkamenet"
            ].tolist()
        )

    # =================================================
    # REMOVE DUPLICATE WORKS
    # =================================================

    collected_works = list(
        set(collected_works)
    )

    # =================================================
    # APPLY CALIBRATION MULTIPLIER
    # =================================================

    total_cost *= RENOVATION_MULTIPLIER

    # =================================================
    # RETURN RESULTS
    # =================================================

    return {
        "total_cost": round(total_cost, 0),
        "package_ids": package_ids,
        "works": collected_works,
        "renovation_possible": True
    }


# =====================================================
# MANUAL TEST
# =====================================================

if __name__ == "__main__":

    result = calculate_renovation_cost(
        current_condition=2,
        target_condition=5,
        building_area_m2=100
    )

    print(result)