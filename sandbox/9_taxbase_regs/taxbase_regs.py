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
COLUMNS = {
    "master": [
        "Year",
        "Municipality",
        "AvgTaxRate",
        "Provider_PPSA",
        "LatestCensusPop",
    ],
    "bgt_exps": ["Year", "Municipality", "Police"],
    "bgt_revs": ["Year", "Municipality", "Unconditional Grant"],
    "tax_base": ["Municipality", "Year", "Total Tax Base for Rate"],
}
JOIN_COLS = ["Year", "Municipality"]
ENTITY_VAR = "Municipality"
TIME_VAR = "Year"


# %%
def main() -> None:
    TXT_DIR.mkdir(parents=True, exist_ok=True)
    TEX_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    run_tax_base_regression()
    run_tax_base_fe_regression()


# %%
def run_tax_base_regression() -> None:
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
            (pl.col("Police") / pl.col("Total Tax Base for Rate")).alias(
                "PolExpTaxBase"
            ),
            (pl.col("UnconditionalGrant") / pl.col("Total Tax Base for Rate")).alias(
                "UnconditionalGrantTaxBase"
            ),
        )
        .drop(
            [
                "Police",
                "UnconditionalGrant",
                "LatestCensusPop",
                "Total Tax Base for Rate",
            ]
        )
        .to_pandas()
    )

    df["entity"] = 1
    df.set_index(["entity", TIME_VAR], inplace=True)

    formula1 = "AvgTaxRate ~ 1 + PolExpTaxBase*Provider_PPSA + UnconditionalGrantTaxBase*Provider_PPSA"
    model1 = PooledOLS.from_formula(formula1, df)
    result1 = model1.fit()

    formula2 = (
        "AvgTaxRate ~ 1 + PolExpTaxBase + UnconditionalGrantTaxBase + "
        "PolExpTaxBase:Provider_PPSA + UnconditionalGrantTaxBase:Provider_PPSA"
    )
    model2 = PooledOLS.from_formula(formula2, df)
    result2 = model2.fit()

    (TXT_DIR / "tax_base_regression_int.txt").write_text(str(result1.summary))
    (TXT_DIR / "tax_base_regression_full.txt").write_text(str(result2.summary))
    (TEX_DIR / "tax_base_regression_int.tex").write_text(result1.summary.as_latex())
    (TEX_DIR / "tax_base_regression_full.tex").write_text(result2.summary.as_latex())

    df["AvgTaxRate_adj_int"] = 100 * (
        df["AvgTaxRate"]
        - result1.params["UnconditionalGrantTaxBase"] * df["UnconditionalGrantTaxBase"]
        - result1.params["UnconditionalGrantTaxBase:Provider_PPSA"]
        * df["UnconditionalGrantTaxBase"]
        * df["Provider_PPSA"]
    )

    df["Fitted_int"] = 100 * (
        result1.params["Intercept"]
        + result1.params["PolExpTaxBase"] * df["PolExpTaxBase"]
        + result1.params["Provider_PPSA"] * df["Provider_PPSA"]
        + result1.params["PolExpTaxBase:Provider_PPSA"]
        * df["PolExpTaxBase"]
        * df["Provider_PPSA"]
        + result1.params["UnconditionalGrantTaxBase"]
        * df["UnconditionalGrantTaxBase"].mean()
        + result1.params["UnconditionalGrantTaxBase:Provider_PPSA"]
        * df["UnconditionalGrantTaxBase"].mean()
        * df["Provider_PPSA"]
    )

    sns.scatterplot(
        df,
        x="PolExpTaxBase",
        y="AvgTaxRate_adj_int",
        hue="Provider_PPSA",
        s=20,
        alpha=0.75,
    )

    sns.lineplot(
        df,
        x="PolExpTaxBase",
        y="Fitted_int",
        hue="Provider_PPSA",
        lw=2,
        palette=["green", "red"],
    )

    plt.xlabel("Police Expenditure/Unit Tax Base")
    plt.ylabel("Avg. Tax Rate (%, adj. for grant effects)")
    plt.title("Avg. Tax Rate vs. Police Exp./Tax Base (interaction only)")
    plt.savefig(PLOTS_DIR / "tax_base_regression_int.png", dpi=300, bbox_inches="tight")
    plt.close()

    df["AvgTaxRate_adj_full"] = (
        100 * df["AvgTaxRate"]
        - result2.params["UnconditionalGrantTaxBase"] * df["UnconditionalGrantTaxBase"]
        - result2.params["UnconditionalGrantTaxBase:Provider_PPSA"]
        * df["UnconditionalGrantTaxBase"]
        * df["Provider_PPSA"]
    )

    df["Fitted_full"] = 100 * (
        result2.params["Intercept"]
        + result2.params["PolExpTaxBase"] * df["PolExpTaxBase"]
        + result2.params["PolExpTaxBase:Provider_PPSA"]
        * df["PolExpTaxBase"]
        * df["Provider_PPSA"]
        + result2.params["UnconditionalGrantTaxBase"]
        * df["UnconditionalGrantTaxBase"].mean()
        + result2.params["UnconditionalGrantTaxBase:Provider_PPSA"]
        * df["UnconditionalGrantTaxBase"].mean()
        * df["Provider_PPSA"]
    )

    sns.scatterplot(
        df,
        x="PolExpTaxBase",
        y="AvgTaxRate_adj_full",
        hue="Provider_PPSA",
        s=20,
        alpha=0.75,
    )

    sns.lineplot(
        df,
        x="PolExpTaxBase",
        y="Fitted_full",
        hue="Provider_PPSA",
        lw=2,
        palette=["green", "red"],
    )

    plt.xlabel("Police Expenditure/Unit Tax Base")
    plt.ylabel("Avg. Tax Rate (%, adj. for grant effects)")
    plt.title("Avg. Tax Rate vs. Police Exp./Tax Base (full form)")
    plt.savefig(
        PLOTS_DIR / "tax_base_regression_full.png", dpi=300, bbox_inches="tight"
    )
    plt.close()


# %%
def run_tax_base_fe_regression() -> None:
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
            (pl.col("Police") / pl.col("Total Tax Base for Rate")).alias(
                "PolExpTaxBase"
            ),
            (pl.col("UnconditionalGrant") / pl.col("Total Tax Base for Rate")).alias(
                "UnconditionalGrantTaxBase"
            ),
        )
        .drop(
            [
                "Police",
                "UnconditionalGrant",
                "LatestCensusPop",
                "Total Tax Base for Rate",
            ]
        )
        .to_pandas()
    )

    df.set_index([ENTITY_VAR, TIME_VAR], inplace=True)
    formula = "AvgTaxRate ~ 1 + PolExpTaxBase + UnconditionalGrantTaxBase + PolExpTaxBase:Provider_PPSA + UnconditionalGrantTaxBase:Provider_PPSA + EntityEffects"

    model1 = PanelOLS.from_formula(formula, df)
    result1 = model1.fit(cov_type="clustered", cluster_entity=True)

    df_2012 = df.loc[df.index.get_level_values("Year") >= 2012].copy()
    model2 = PanelOLS.from_formula(formula, df_2012)
    result2 = model2.fit(cov_type="clustered", cluster_entity=True)

    (TXT_DIR / "tax_base_fe_regression_full.txt").write_text(str(result1.summary))
    (TXT_DIR / "tax_base_fe_regression_2012plus.txt").write_text(str(result2.summary))
    (TEX_DIR / "tax_base_fe_regression_full.tex").write_text(result1.summary.as_latex())
    (TEX_DIR / "tax_base_fe_regression_2012plus.tex").write_text(
        result2.summary.as_latex()
    )

    df_2012["AvgTaxRate_adj_full"] = 100 * (
        df_2012["AvgTaxRate"]
        - result1.params["UnconditionalGrantTaxBase"]
        * df_2012["UnconditionalGrantTaxBase"]
        - result1.params["UnconditionalGrantTaxBase:Provider_PPSA"]
        * df_2012["UnconditionalGrantTaxBase"]
        * df_2012["Provider_PPSA"]
    )

    df_2012["Fitted_full"] = 100 * (
        result1.params["Intercept"]
        + result1.params["PolExpTaxBase"] * df_2012["PolExpTaxBase"]
        + result1.params["PolExpTaxBase:Provider_PPSA"]
        * df_2012["PolExpTaxBase"]
        * df_2012["Provider_PPSA"]
    )

    sns.scatterplot(
        df_2012,
        x="PolExpTaxBase",
        y="AvgTaxRate_adj_full",
        hue="Provider_PPSA",
        s=20,
        alpha=0.75,
    )

    sns.lineplot(
        df_2012,
        x="PolExpTaxBase",
        y="Fitted_full",
        hue="Provider_PPSA",
        legend=False,
        palette=["green", "red"],
    )

    plt.xlabel("Police Expenditure/Unit Tax Base")
    plt.ylabel("Avg. Tax Rate (%, adj. for grant effects)")
    plt.title("Avg. Tax Rate vs. Police Exp./Tax Base (FE, all years)")
    plt.savefig(
        PLOTS_DIR / "tax_base_fe_regression_full.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    df_2012["AvgTaxRate_adj_2012plus"] = 100 * (
        df_2012["AvgTaxRate"]
        - result2.params["UnconditionalGrantTaxBase"]
        * df_2012["UnconditionalGrantTaxBase"]
        - result2.params["UnconditionalGrantTaxBase:Provider_PPSA"]
        * df_2012["UnconditionalGrantTaxBase"]
        * df_2012["Provider_PPSA"]
    )

    df_2012["Fitted_2012plus"] = 100 * (
        result2.params["Intercept"]
        + result2.params["PolExpTaxBase"] * df_2012["PolExpTaxBase"]
        + result2.params["PolExpTaxBase:Provider_PPSA"]
        * df_2012["PolExpTaxBase"]
        * df_2012["Provider_PPSA"]
    )

    sns.scatterplot(
        df_2012,
        x="PolExpTaxBase",
        y="AvgTaxRate_adj_2012plus",
        hue="Provider_PPSA",
        s=20,
        alpha=0.75,
    )

    sns.lineplot(
        df_2012,
        x="PolExpTaxBase",
        y="Fitted_2012plus",
        hue="Provider_PPSA",
        legend=False,
        palette=["green", "red"],
    )

    plt.xlabel("Police Expenditure/Unit Tax Base")
    plt.ylabel("Avg. Tax Rate (%, adj. for grant effects)")
    plt.title("Avg. Tax Rate vs. Police Exp./Tax Base (FE, 2012 onwards)")
    plt.savefig(
        PLOTS_DIR / "tax_base_fe_regression_2012plus.png", dpi=300, bbox_inches="tight"
    )
    plt.close()


# %%
if __name__ == "__main__":
    main()
