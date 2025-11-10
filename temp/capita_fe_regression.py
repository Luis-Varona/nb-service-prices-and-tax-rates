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

from linearmodels.panel.model import PanelOLS


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
        "Provider_PPSA",
        "LatestCensusPop",
    ],
    "bgt_revs": ["Year", "Municipality", "Unconditional Grant"],
}
JOIN_COLS = ["Year", "Municipality"]
ENTITY_VAR = "Municipality"
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

    df.set_index([ENTITY_VAR, TIME_VAR], inplace=True)
    formula = "AvgTaxRate ~ 1 + PolExpCapita + UnconditionalGrantCapita + PolExpCapita:Provider_PPSA + UnconditionalGrantCapita:Provider_PPSA + EntityEffects"

    model1 = PanelOLS.from_formula(formula, df)
    result1 = model1.fit(cov_type="clustered", cluster_entity=True)
    print("====================FULL CAPITA FE REGRESSION====================")
    print(result1.summary)
    print()

    df = df.loc[df.index.get_level_values("Year") >= 2012].copy()
    model2 = PanelOLS.from_formula(formula, df)
    result2 = model2.fit(cov_type="clustered", cluster_entity=True)
    print("====================2012+ CAPITA FE REGRESSION====================")
    print(result2.summary)
    print()

    TEX_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(TEX_DIR / "capita_fe_regression_full.tex", "w") as f:
        f.write(result1.summary.as_latex())

    with open(TEX_DIR / "capita_fe_regression_2012plus.tex", "w") as f:
        f.write(result2.summary.as_latex())

    df["AvgTaxRate_adj_full"] = 100 * (
        df["AvgTaxRate"]
        - result1.params["UnconditionalGrantCapita"] * df["UnconditionalGrantCapita"]
        - result1.params["UnconditionalGrantCapita:Provider_PPSA"]
        * df["UnconditionalGrantCapita"]
        * df["Provider_PPSA"]
    )

    df["Fitted_full"] = 100 * (
        result1.params["Intercept"]
        + result1.params["PolExpCapita"] * df["PolExpCapita"]
        + result1.params["PolExpCapita:Provider_PPSA"]
        * df["PolExpCapita"]
        * df["Provider_PPSA"]
    )

    sns.scatterplot(
        df,
        x="PolExpCapita",
        y="AvgTaxRate_adj_full",
        hue="Provider_PPSA",
        s=20,
        alpha=0.75,
    )

    sns.lineplot(
        df,
        x="PolExpCapita",
        y="Fitted_full",
        hue="Provider_PPSA",
        legend=False,
        palette=["green", "red"],
    )

    plt.xlabel("Police Expenditure/Capita")
    plt.ylabel("Avg. Tax Rate (%, adj. for grant effects)")
    plt.title("Avg. Tax Rate vs. Police Exp./Capita (FE, all years)")
    plt.savefig(
        PLOTS_DIR / "capita_fe_regression_full.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    df["AvgTaxRate_adj_2012plus"] = 100 * (
        df["AvgTaxRate"]
        - result2.params["UnconditionalGrantCapita"] * df["UnconditionalGrantCapita"]
        - result2.params["UnconditionalGrantCapita:Provider_PPSA"]
        * df["UnconditionalGrantCapita"]
        * df["Provider_PPSA"]
    )

    df["Fitted_2012plus"] = 100 * (
        result2.params["Intercept"]
        + result2.params["PolExpCapita"] * df["PolExpCapita"]
        + result2.params["PolExpCapita:Provider_PPSA"]
        * df["PolExpCapita"]
        * df["Provider_PPSA"]
    )

    sns.scatterplot(
        df,
        x="PolExpCapita",
        y="AvgTaxRate_adj_2012plus",
        hue="Provider_PPSA",
        s=20,
        alpha=0.75,
    )

    sns.lineplot(
        df,
        x="PolExpCapita",
        y="Fitted_2012plus",
        hue="Provider_PPSA",
        legend=False,
        palette=["green", "red"],
    )

    plt.xlabel("Police Expenditure/Capita")
    plt.ylabel("Avg. Tax Rate (%, adj. for grant effects)")
    plt.title("Avg. Tax Rate vs. Police Exp./Capita (FE, 2012 onwards)")
    plt.savefig(
        PLOTS_DIR / "capita_fe_regression_2012plus.png", dpi=300, bbox_inches="tight"
    )
    plt.close()


# %%
if __name__ == "__main__":
    main()
