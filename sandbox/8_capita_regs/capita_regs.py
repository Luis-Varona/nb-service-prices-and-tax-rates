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

from linearmodels.panel.model import PanelOLS, PooledOLS


# %%
WD = Path(__file__).parent
DATA_DIR = WD.parent.parent / "data" / "data_final"
SRC_STEM = "data"
TXT_DIR = WD / "txt"
TEX_DIR = WD / "tex"
PLOTS_DIR = WD / "plots"


# %%
COLUMNS_SHARE = {
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

COLUMNS_CAPITA = {
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
    TXT_DIR.mkdir(parents=True, exist_ok=True)
    TEX_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    run_share_regression()
    run_capita_regression()
    run_capita_fe_regression()


# %%
def run_share_regression() -> None:
    dfs = [
        pl.read_excel(DATA_DIR / f"{SRC_STEM}_{key}.xlsx", columns=cols)
        for key, cols in COLUMNS_SHARE.items()
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

    (TXT_DIR / "share_regression.txt").write_text(str(result.summary))
    (TEX_DIR / "share_regression.tex").write_text(result.summary.as_latex())

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
def run_capita_regression() -> None:
    dfs = [
        pl.read_excel(DATA_DIR / f"{SRC_STEM}_{key}.xlsx", columns=cols)
        for key, cols in COLUMNS_CAPITA.items()
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

    formula2 = (
        "AvgTaxRate ~ 1 + PolExpCapita + UnconditionalGrantCapita + "
        "PolExpCapita:Provider_PPSA + UnconditionalGrantCapita:Provider_PPSA"
    )
    model2 = PooledOLS.from_formula(formula2, df)
    result2 = model2.fit()

    (TXT_DIR / "capita_regression_int.txt").write_text(str(result1.summary))
    (TXT_DIR / "capita_regression_full.txt").write_text(str(result2.summary))
    (TEX_DIR / "capita_regression_int.tex").write_text(result1.summary.as_latex())
    (TEX_DIR / "capita_regression_full.tex").write_text(result2.summary.as_latex())

    df["AvgTaxRate_adj_int"] = 100 * (
        df["AvgTaxRate"]
        - result1.params["UnconditionalGrantCapita"] * df["UnconditionalGrantCapita"]
        - result1.params["UnconditionalGrantCapita:Provider_PPSA"]
        * df["UnconditionalGrantCapita"]
        * df["Provider_PPSA"]
    )

    df["Fitted_int"] = 100 * (
        result1.params["Intercept"]
        + result1.params["PolExpCapita"] * df["PolExpCapita"]
        + result1.params["Provider_PPSA"] * df["Provider_PPSA"]
        + result1.params["PolExpCapita:Provider_PPSA"]
        * df["PolExpCapita"]
        * df["Provider_PPSA"]
        + result1.params["UnconditionalGrantCapita"]
        * df["UnconditionalGrantCapita"].mean()
        + result1.params["UnconditionalGrantCapita:Provider_PPSA"]
        * df["UnconditionalGrantCapita"].mean()
        * df["Provider_PPSA"]
    )

    sns.scatterplot(
        df,
        x="PolExpCapita",
        y="AvgTaxRate_adj_int",
        hue="Provider_PPSA",
        s=20,
        alpha=0.75,
    )

    sns.lineplot(
        df,
        x="PolExpCapita",
        y="Fitted_int",
        hue="Provider_PPSA",
        lw=2,
        palette=["green", "red"],
    )

    plt.xlabel("Police Expenditure/Capita")
    plt.ylabel("Avg. Tax Rate (%, adj. for grant effects)")
    plt.title("Avg. Tax Rate vs. Police Exp./Capita (interaction only)")
    plt.savefig(PLOTS_DIR / "capita_regression_int.png", dpi=300, bbox_inches="tight")
    plt.close()

    df["AvgTaxRate_adj_full"] = (
        100 * df["AvgTaxRate"]
        - result2.params["UnconditionalGrantCapita"] * df["UnconditionalGrantCapita"]
        - result2.params["UnconditionalGrantCapita:Provider_PPSA"]
        * df["UnconditionalGrantCapita"]
        * df["Provider_PPSA"]
    )

    df["Fitted_full"] = 100 * (
        result2.params["Intercept"]
        + result2.params["PolExpCapita"] * df["PolExpCapita"]
        + result2.params["PolExpCapita:Provider_PPSA"]
        * df["PolExpCapita"]
        * df["Provider_PPSA"]
        + result2.params["UnconditionalGrantCapita"]
        * df["UnconditionalGrantCapita"].mean()
        + result2.params["UnconditionalGrantCapita:Provider_PPSA"]
        * df["UnconditionalGrantCapita"].mean()
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
        lw=2,
        palette=["green", "red"],
    )

    plt.xlabel("Police Expenditure/Capita")
    plt.ylabel("Avg. Tax Rate (%, adj. for grant effects)")
    plt.title("Avg. Tax Rate vs. Police Exp./Capita (full form)")
    plt.savefig(PLOTS_DIR / "capita_regression_full.png", dpi=300, bbox_inches="tight")
    plt.close()


# %%
def run_capita_fe_regression() -> None:
    dfs = [
        pl.read_excel(DATA_DIR / f"{SRC_STEM}_{key}.xlsx", columns=cols)
        for key, cols in COLUMNS_CAPITA.items()
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

    df_2012 = df.loc[df.index.get_level_values("Year") >= 2012].copy()
    model2 = PanelOLS.from_formula(formula, df_2012)
    result2 = model2.fit(cov_type="clustered", cluster_entity=True)

    (TXT_DIR / "capita_fe_regression_full.txt").write_text(str(result1.summary))
    (TXT_DIR / "capita_fe_regression_2012plus.txt").write_text(str(result2.summary))
    (TEX_DIR / "capita_fe_regression_full.tex").write_text(result1.summary.as_latex())
    (TEX_DIR / "capita_fe_regression_2012plus.tex").write_text(
        result2.summary.as_latex()
    )

    df_2012["AvgTaxRate_adj_full"] = 100 * (
        df_2012["AvgTaxRate"]
        - result1.params["UnconditionalGrantCapita"]
        * df_2012["UnconditionalGrantCapita"]
        - result1.params["UnconditionalGrantCapita:Provider_PPSA"]
        * df_2012["UnconditionalGrantCapita"]
        * df_2012["Provider_PPSA"]
    )

    df_2012["Fitted_full"] = 100 * (
        result1.params["Intercept"]
        + result1.params["PolExpCapita"] * df_2012["PolExpCapita"]
        + result1.params["PolExpCapita:Provider_PPSA"]
        * df_2012["PolExpCapita"]
        * df_2012["Provider_PPSA"]
    )

    sns.scatterplot(
        df_2012,
        x="PolExpCapita",
        y="AvgTaxRate_adj_full",
        hue="Provider_PPSA",
        s=20,
        alpha=0.75,
    )

    sns.lineplot(
        df_2012,
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

    df_2012["AvgTaxRate_adj_2012plus"] = 100 * (
        df_2012["AvgTaxRate"]
        - result2.params["UnconditionalGrantCapita"]
        * df_2012["UnconditionalGrantCapita"]
        - result2.params["UnconditionalGrantCapita:Provider_PPSA"]
        * df_2012["UnconditionalGrantCapita"]
        * df_2012["Provider_PPSA"]
    )

    df_2012["Fitted_2012plus"] = 100 * (
        result2.params["Intercept"]
        + result2.params["PolExpCapita"] * df_2012["PolExpCapita"]
        + result2.params["PolExpCapita:Provider_PPSA"]
        * df_2012["PolExpCapita"]
        * df_2012["Provider_PPSA"]
    )

    sns.scatterplot(
        df_2012,
        x="PolExpCapita",
        y="AvgTaxRate_adj_2012plus",
        hue="Provider_PPSA",
        s=20,
        alpha=0.75,
    )

    sns.lineplot(
        df_2012,
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
