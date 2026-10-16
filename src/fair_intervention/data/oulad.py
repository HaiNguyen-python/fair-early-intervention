"""Load the real OULAD CSV files, or the synthetic copies written by `generate`.

Get the data from https://analyse.kmi.open.ac.uk/open_dataset (CC BY 4.0) and unzip it into one
folder. Only five of the seven files are needed. Two quirks of the real files are handled here:

* missing values are written as ``?`` (for example ``score`` and ``date_unregistration``);
* one ``imd_band`` value is spelled ``10-20`` without the percent sign.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS: dict[str, tuple[str, ...]] = {
    "studentInfo": (
        "code_module",
        "code_presentation",
        "id_student",
        "gender",
        "region",
        "highest_education",
        "imd_band",
        "age_band",
        "num_of_prev_attempts",
        "studied_credits",
        "disability",
        "final_result",
    ),
    "studentRegistration": (
        "code_module",
        "code_presentation",
        "id_student",
        "date_registration",
        "date_unregistration",
    ),
    "studentVle": (
        "code_module",
        "code_presentation",
        "id_student",
        "id_site",
        "date",
        "sum_click",
    ),
    "studentAssessment": ("id_assessment", "id_student", "date_submitted", "is_banked", "score"),
    "assessments": (
        "code_module",
        "code_presentation",
        "id_assessment",
        "assessment_type",
        "date",
        "weight",
    ),
}


def validate(tables: dict[str, pd.DataFrame]) -> None:
    """Fail early with a clear message instead of a KeyError deep inside feature code."""
    for name, columns in REQUIRED_COLUMNS.items():
        if name not in tables:
            raise ValueError(f"missing table {name!r}")
        missing = set(columns) - set(tables[name].columns)
        if missing:
            raise ValueError(f"{name} is missing columns: {sorted(missing)}")


def load(data_dir: Path) -> dict[str, pd.DataFrame]:
    """Read ``<data_dir>/<table>.csv`` for every required table and normalise known quirks."""
    tables = {}
    for name in REQUIRED_COLUMNS:
        path = data_dir / f"{name}.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Download OULAD or run `fair-intervention generate`."
            )
        tables[name] = pd.read_csv(path, na_values=["?"])
    validate(tables)
    info = tables["studentInfo"]
    info["imd_band"] = info["imd_band"].replace({"10-20": "10-20%"})
    return tables


def save(tables: dict[str, pd.DataFrame], data_dir: Path) -> list[Path]:
    """Write tables as OULAD-named CSVs so the real-data loader can read them back."""
    validate(tables)
    data_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name in REQUIRED_COLUMNS:
        path = data_dir / f"{name}.csv"
        tables[name].to_csv(path, index=False)
        paths.append(path)
    return paths
