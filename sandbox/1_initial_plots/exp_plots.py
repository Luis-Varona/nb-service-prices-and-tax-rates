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


# %%
WD = Path(__file__).parent
SRC = WD.parent.parent / "data" / "data_final" / "data_master.xlsx"
TEX_DIR = WD / "tex"
PLOTS_DIR = WD / "plots"
DIVIDE_YEAR = 2012


# %%
def main() -> None:
    df = pl.read_excel(SRC).select(
        [
            "Year",
            "PolExpCapita",
            "Provider_PPSA",
            "LatestCensusPop",
        ]
    )

    df_weighted = (
        df.group_by("Year")
        .agg(
            (
                (pl.col("PolExpCapita") * pl.col("LatestCensusPop"))
                .filter(pl.col("Provider_PPSA"))
                .sum()
                / pl.col("LatestCensusPop").filter(pl.col("Provider_PPSA")).sum()
            ).alias("PolExpCapita_PPSA"),
            (
                (pl.col("PolExpCapita") * pl.col("LatestCensusPop"))
                .filter(~pl.col("Provider_PPSA"))
                .sum()
                / pl.col("LatestCensusPop").filter(~pl.col("Provider_PPSA")).sum()
            ).alias("PolExpCapita_nonPPSA"),
        )
        .sort("Year")
        .with_columns(
            (
                (pl.col("PolExpCapita_PPSA") - pl.col("PolExpCapita_PPSA").shift(1))
                / pl.col("PolExpCapita_PPSA").shift(1)
                * 100
            ).alias("PolExpCapita_PPSA_yoy"),
            (
                (
                    pl.col("PolExpCapita_nonPPSA")
                    - pl.col("PolExpCapita_nonPPSA").shift(1)
                )
                / pl.col("PolExpCapita_nonPPSA").shift(1)
                * 100
            ).alias("PolExpCapita_nonPPSA_yoy"),
            (pl.col("PolExpCapita_nonPPSA") - pl.col("PolExpCapita_PPSA")).alias(
                "PolExpCapita_diff"
            ),
        )
    )

    years_w = get_vals(df_weighted, "Year").astype(int)

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure()
    sns.lineplot(x=years_w, y=get_vals(df_weighted, "PolExpCapita_PPSA"), label="PPSA")
    sns.lineplot(
        x=years_w, y=get_vals(df_weighted, "PolExpCapita_nonPPSA"), label="non-PPSA"
    )
    plt.title("PolExpCapita")
    add_divider(plt.gca())
    plt.xticks(years_w, rotation=90)
    plt.savefig(PLOTS_DIR / "pec_weighted.png")
    plt.close()

    plt.figure()
    sns.lineplot(
        x=years_w, y=get_vals(df_weighted, "PolExpCapita_PPSA_yoy"), label="PPSA"
    )
    sns.lineplot(
        x=years_w, y=get_vals(df_weighted, "PolExpCapita_nonPPSA_yoy"), label="non-PPSA"
    )
    plt.title("YoY PolExpCapita (%)")
    add_divider(plt.gca())
    plt.xticks(years_w, rotation=90)
    plt.savefig(PLOTS_DIR / "pec_weighted_yoy.png")
    plt.close()

    plt.figure()
    sns.lineplot(x=years_w, y=get_vals(df_weighted, "PolExpCapita_diff"))
    plt.title("non-PPSA - PPSA Difference")
    add_divider(plt.gca())
    plt.xticks(years_w, rotation=90)
    plt.savefig(PLOTS_DIR / "pec_weighted_diff.png")
    plt.close()


# %%
def add_divider(ax):
    ax.axvline(DIVIDE_YEAR, color="red", linestyle="--")


def get_vals(df, col):
    return df.select(col).to_numpy().ravel()


# %%
if __name__ == "__main__":
    main()
