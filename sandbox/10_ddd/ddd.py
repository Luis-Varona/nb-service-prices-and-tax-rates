# Copyright 2025 Craig Brett and Luis M. B. Varona
#
# Licensed under the MIT license <LICENSE or
# http://opensource.org/licenses/MIT>. This file may not be copied, modified, or
# distributed except according to those terms.


# %%
import pickle
from pathlib import Path

import polars as pl
from linearmodels.panel.model import PanelOLS, PooledOLS

# %%
WD = Path(__file__).parent
SRC = WD.parent.parent / "data" / "data_final"
TXT_DIR = WD / "txt"
TEX_DIR = WD / "tex"
PKL_DIR = WD / "pkl"


# %%
DEP_VAR = "AvgTaxRate"
ENTITY_COL = "Municipality"
TIME_COL = "Year"
POST_YEAR = 2012


# %%
DDD_CORE = [
    "PolExpCapita",
    "PolExpCapita:Provider_PPSA",
    "PolExpCapita:Post2012",
    "Provider_PPSA:Post2012",
    "PolExpCapita:Provider_PPSA:Post2012",
]

POOLED_EXTRA = ["Provider_PPSA", "Post2012"]
FE_EXTRA = ["Post2012"]


# %%
CONTROL_SETS = [
    ("no_ctrl", []),
    ("exp", ["OtherExpCapita"]),
    ("exp_rev", ["OtherExpCapita", "OtherRevCapita"]),
    ("full", ["OtherExpCapita", "OtherRevCapita", "UnconditionalGrantCapita"]),
]


# %%
def main() -> None:
    TXT_DIR.mkdir(parents=True, exist_ok=True)
    TEX_DIR.mkdir(parents=True, exist_ok=True)
    PKL_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data()
    results = {}

    for ctrl_name, ctrl_vars in CONTROL_SETS:
        key = f"pooled_{ctrl_name}"
        terms = POOLED_EXTRA + DDD_CORE + ctrl_vars
        results[key] = run_pooled(df, terms)
        save_summary(key, results[key])

        key = f"fe_{ctrl_name}"
        terms = FE_EXTRA + DDD_CORE + ctrl_vars
        results[key] = run_panel(df, terms)
        save_summary(key, results[key])

    with open(PKL_DIR / "results.pkl", "wb") as f:
        pickle.dump(results, f)


# %%
def save_summary(key: str, result) -> None:
    (TXT_DIR / f"{key}.txt").write_text(result.summary.as_text())
    (TEX_DIR / f"{key}.tex").write_text(
        result.summary.as_latex().replace("\\begin{table}", "\\begin{table}[H]", 1)
    )


# %%
def load_data() -> pl.DataFrame:
    df_master = pl.read_excel(SRC / "data_master.xlsx").select(
        [
            TIME_COL,
            ENTITY_COL,
            "AvgTaxRate",
            "PolExpCapita",
            "OtherExpCapita",
            "OtherRevCapita",
            "Provider_PPSA",
            "LatestCensusPop",
        ]
    )
    df_revs = pl.read_excel(
        SRC / "data_bgt_revs.xlsx",
        columns=[TIME_COL, ENTITY_COL, "Unconditional Grant"],
    )

    return (
        df_master.join(df_revs, on=[TIME_COL, ENTITY_COL], how="left")
        .rename({"Unconditional Grant": "UnconditionalGrant"})
        .with_columns(
            (pl.col("UnconditionalGrant") / pl.col("LatestCensusPop")).alias(
                "UnconditionalGrantCapita"
            ),
            (pl.col(TIME_COL) >= POST_YEAR).cast(pl.Int8).alias("Post2012"),
            pl.col("Provider_PPSA").cast(pl.Int8),
        )
        .drop(["UnconditionalGrant", "LatestCensusPop"])
    )


# %%
def run_pooled(df: pl.DataFrame, terms: list[str]):
    formula = f"{DEP_VAR} ~ 1 + {' + '.join(terms)}"
    df_pd = df.to_pandas().set_index([ENTITY_COL, TIME_COL])
    model = PooledOLS.from_formula(formula, df_pd)
    return model.fit(cov_type="clustered", cluster_entity=True)


def run_panel(df: pl.DataFrame, terms: list[str]):
    formula = f"{DEP_VAR} ~ 1 + {' + '.join(terms)} + EntityEffects"
    df_pd = df.to_pandas().set_index([ENTITY_COL, TIME_COL])
    model = PanelOLS.from_formula(formula, df_pd)
    return model.fit(cov_type="clustered", cluster_entity=True)


# %%
if __name__ == "__main__":
    main()
