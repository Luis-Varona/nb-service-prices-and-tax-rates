# Copyright 2025 Craig Brett and Luis M. B. Varona
#
# Licensed under the MIT license <LICENSE or
# http://opensource.org/licenses/MIT>. This file may not be copied, modified, or
# distributed except according to those terms.


# %%
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns

from linearmodels.panel.model import PooledOLS


# %%
WD = Path(__file__).parent
DATA_DIR = WD.parent / "data" / "data_final"
SRC_STEM = "data"
TEX_DIR = WD / "tex"
PLOTS_DIR = WD / "plots"


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
    df.set_index(["entity", TIME_VAR], inplace=True)

    model = PooledOLS.from_formula(formula, df)
    result = model.fit()
    print("====================SHARE REGRESSION====================")
    print(result.summary)
    print()

    TEX_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(TEX_DIR / "share_regression.tex", "w") as f:
        f.write(result.summary.as_latex())

    df["AvgTaxRate_adj"] = 100 * (
        df["AvgTaxRate"]
        - result.params["UnconditionalGrant"] * df["UnconditionalGrant"]
        - result.params["UnconditionalGrant:Provider_PPSA"]
        * df["UnconditionalGrant"]
        * df["Provider_PPSA"]
    )

    df["Fitted"] = 100 * (
        result.params["Intercept"]
        + result.params["PolExpShare"] * df["PolExpShare"]
        + result.params["Provider_PPSA"] * df["Provider_PPSA"]
        + result.params["PolExpShare:Provider_PPSA"]
        * df["PolExpShare"]
        * df["Provider_PPSA"]
        + result.params["UnconditionalGrant"] * df["UnconditionalGrant"].mean()
        + result.params["UnconditionalGrant:Provider_PPSA"]
        * df["UnconditionalGrant"].mean()
        * df["Provider_PPSA"]
    )

    sns.scatterplot(
        df,
        x="PolExpShare",
        y="AvgTaxRate_adj",
        hue="Provider_PPSA",
        s=20,
        alpha=0.75,
    )

    sns.lineplot(
        df,
        x="PolExpShare",
        y="Fitted",
        hue="Provider_PPSA",
        lw=2,
        palette=["green", "red"],
    )

    plt.xlabel("Police Expenditure/Total Expenditure")
    plt.ylabel("Avg. Tax Rate (%, adj. for grant effects)")
    plt.title("Average Tax Rate vs. Police Expenditure Share")
    plt.savefig(PLOTS_DIR / "share_regression.png", dpi=300, bbox_inches="tight")
    plt.close()


# %%
if __name__ == "__main__":
    main()
