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
        "Provider_PPSA",
        "LatestCensusPop",
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
        df.rename({"Unconditional Grant": "UnconditionalGrant"})
        .with_columns(
            (pl.col("UnconditionalGrant") / pl.col("LatestCensusPop")).alias(
                "UnconditionalGrantCapita"
            )
        )
        .drop(["UnconditionalGrant", "LatestCensusPop"])
        .to_pandas()
    )

    df["entity"] = 1
    df.set_index(["entity", TIME_VAR], inplace=True)

    formula1 = "AvgTaxRate ~ 1 + PolExpCapita*Provider_PPSA + UnconditionalGrantCapita*Provider_PPSA"
    model1 = PooledOLS.from_formula(formula1, df)
    result1 = model1.fit()
    print(result1.summary)

    formula2 = "AvgTaxRate ~ 1 + PolExpCapita + UnconditionalGrantCapita + PolExpCapita:Provider_PPSA + UnconditionalGrantCapita:Provider_PPSA"
    model2 = PooledOLS.from_formula(formula2, df)
    result2 = model2.fit()
    print(result2.summary)


# %%
if __name__ == "__main__":
    main()
