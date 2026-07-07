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
SRC = WD.parent.parent / "data" / "data_final" / "data_master.xlsx"
TXT_DIR = WD / "txt"
TEX_DIR = WD / "tex"
PKL_DIR = WD / "pkl"


# %%
DEP_VAR = "TaxBaseCapita"
ENTITY_COL = "Municipality"
TIME_COL = "Year"
POST_YEAR = 2012


# %%
def main() -> None:
    TXT_DIR.mkdir(parents=True, exist_ok=True)
    TEX_DIR.mkdir(parents=True, exist_ok=True)
    PKL_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()
    results = {}

    results["pooled"] = run_pooled(
        df, ["Provider_PPSA", "Post2012", "Provider_PPSA:Post2012"]
    )
    save_summary("pooled", results["pooled"])

    results["fe"] = run_panel(df, ["Post2012", "Provider_PPSA:Post2012"])
    save_summary("fe", results["fe"])

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
    return (
        pl.read_excel(SRC)
        .select(
            [
                TIME_COL,
                ENTITY_COL,
                "TaxBaseCapita",
                "Provider_PPSA",
            ]
        )
        .with_columns(
            (pl.col(TIME_COL) >= POST_YEAR).cast(pl.Int8).alias("Post2012"),
            pl.col("Provider_PPSA").cast(pl.Int8),
        )
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
