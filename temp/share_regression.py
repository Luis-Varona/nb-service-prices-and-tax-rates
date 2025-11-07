# Copyright 2025 Craig Brett and Luis M. B. Varona
#
# Licensed under the MIT license <LICENSE or
# http://opensource.org/licenses/MIT>. This file may not be copied, modified, or
# distributed except according to those terms.


# %%
from pathlib import Path

import polars as pl

from linearmodels.panel.model import PooledOLS


# %%
WD = Path(__file__).parent
DATA_DIR = WD.parent / "data" / "data_final"
SRC_STEM = "data"


# %%
COLUMNS = {
    "master": [
        "Year",
        "Municipality",
        "AvgTaxRate",
        "PolExpCapita",
        "OtherExpCapita",
        "Provider_PPSA",
    ],
    "bgt_revs": ["Year", "Municipality", "Unconditional Grant"],
}
JOIN_COLS = ["Year", "Municipality"]
TIME_VAR = "Year"


# %%
def main() -> None:
    dfs = [
        pl.read_excel(DATA_DIR / f"{SRC_STEM}_{key}.xlsx", columns=cols)
        for key, cols in COLUMNS.items()
    ]
    df = dfs[0]

    for other_df in dfs[1:]:
        df = df.join(other_df, on=JOIN_COLS, how="left")

    df = (
        df.with_columns(
            (
                pl.col("PolExpCapita")
                / (pl.col("PolExpCapita") + pl.col("OtherExpCapita"))
            ).alias("PolExpShare"),
        )
        .rename({"Unconditional Grant": "UnconditionalGrant"})
        .drop(["PolExpCapita", "OtherExpCapita"])
        .to_pandas()
    )

    formula = (
        "AvgTaxRate ~ 1 + PolExpShare*Provider_PPSA + UnconditionalGrant*Provider_PPSA"
    )

    df["entity"] = 1
    df = df.set_index(["entity", TIME_VAR])
    print(df)

    model = PooledOLS.from_formula(formula, df)
    result = model.fit()
    print(result.summary)


# %%
if __name__ == "__main__":
    main()
