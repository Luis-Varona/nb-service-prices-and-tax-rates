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
PLOTS_DIR = WD / "plots"
DIVIDE_YEAR = 2012


# %%
def main() -> None:
    df = pl.read_excel(SRC).select(
        [
            "Year",
            "PolExpCapita",
            "AvgTaxRate",
            "Provider_PPSA",
            "LatestCensusPop",
        ]
    )

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    plot_polexpcapita(df)
    plot_avgtaxrate(df)


# %%
def plot_polexpcapita(df: pl.DataFrame) -> None:
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
def plot_avgtaxrate(df: pl.DataFrame) -> None:
    df_unweighted = (
        df.drop("LatestCensusPop")
        .group_by("Year")
        .agg(
            pl.col("AvgTaxRate")
            .filter(pl.col("Provider_PPSA"))
            .mean()
            .alias("AvgTaxRate_PPSA"),
            pl.col("AvgTaxRate")
            .filter(~pl.col("Provider_PPSA"))
            .mean()
            .alias("AvgTaxRate_nonPPSA"),
        )
        .sort("Year")
        .with_columns(
            (
                (pl.col("AvgTaxRate_PPSA") - pl.col("AvgTaxRate_PPSA").shift(1))
                / pl.col("AvgTaxRate_PPSA").shift(1)
                * 100
            ).alias("AvgTaxRate_PPSA_yoy"),
            (
                (pl.col("AvgTaxRate_nonPPSA") - pl.col("AvgTaxRate_nonPPSA").shift(1))
                / pl.col("AvgTaxRate_nonPPSA").shift(1)
                * 100
            ).alias("AvgTaxRate_nonPPSA_yoy"),
            (pl.col("AvgTaxRate_nonPPSA") - pl.col("AvgTaxRate_PPSA")).alias(
                "AvgTaxRate_diff"
            ),
        )
    )

    df_weighted = (
        df.group_by("Year")
        .agg(
            (
                (pl.col("AvgTaxRate") * pl.col("LatestCensusPop"))
                .filter(pl.col("Provider_PPSA"))
                .sum()
                / pl.col("LatestCensusPop").filter(pl.col("Provider_PPSA")).sum()
            ).alias("AvgTaxRate_PPSA"),
            (
                (pl.col("AvgTaxRate") * pl.col("LatestCensusPop"))
                .filter(~pl.col("Provider_PPSA"))
                .sum()
                / pl.col("LatestCensusPop").filter(~pl.col("Provider_PPSA")).sum()
            ).alias("AvgTaxRate_nonPPSA"),
        )
        .sort("Year")
        .with_columns(
            (
                (pl.col("AvgTaxRate_PPSA") - pl.col("AvgTaxRate_PPSA").shift(1))
                / pl.col("AvgTaxRate_PPSA").shift(1)
                * 100
            ).alias("AvgTaxRate_PPSA_yoy"),
            (
                (pl.col("AvgTaxRate_nonPPSA") - pl.col("AvgTaxRate_nonPPSA").shift(1))
                / pl.col("AvgTaxRate_nonPPSA").shift(1)
                * 100
            ).alias("AvgTaxRate_nonPPSA_yoy"),
            (pl.col("AvgTaxRate_nonPPSA") - pl.col("AvgTaxRate_PPSA")).alias(
                "AvgTaxRate_diff"
            ),
        )
    )

    years_unw = get_vals(df_unweighted, "Year").astype(int)
    years_w = get_vals(df_weighted, "Year").astype(int)

    plt.figure()
    sns.lineplot(
        x=years_unw, y=get_vals(df_unweighted, "AvgTaxRate_PPSA"), label="PPSA"
    )
    sns.lineplot(
        x=years_unw, y=get_vals(df_unweighted, "AvgTaxRate_nonPPSA"), label="non-PPSA"
    )
    plt.title("Unweighted AvgTaxRate")
    add_divider(plt.gca())
    plt.xticks(years_unw, rotation=90)
    plt.savefig(PLOTS_DIR / "unweighted_avgtaxrate.png")
    plt.close()

    plt.figure()
    sns.lineplot(
        x=years_unw, y=get_vals(df_unweighted, "AvgTaxRate_PPSA_yoy"), label="PPSA"
    )
    sns.lineplot(
        x=years_unw,
        y=get_vals(df_unweighted, "AvgTaxRate_nonPPSA_yoy"),
        label="non-PPSA",
    )
    plt.title("Unweighted YoY AvgTaxRate (%)")
    add_divider(plt.gca())
    plt.xticks(years_unw, rotation=90)
    plt.savefig(PLOTS_DIR / "unweighted_yoy.png")
    plt.close()

    plt.figure()
    sns.lineplot(x=years_unw, y=get_vals(df_unweighted, "AvgTaxRate_diff"))
    plt.title("Unweighted non-PPSA - PPSA Difference")
    add_divider(plt.gca())
    plt.xticks(years_unw, rotation=90)
    plt.savefig(PLOTS_DIR / "unweighted_diff.png")
    plt.close()

    plt.figure()
    sns.lineplot(x=years_w, y=get_vals(df_weighted, "AvgTaxRate_PPSA"), label="PPSA")
    sns.lineplot(
        x=years_w, y=get_vals(df_weighted, "AvgTaxRate_nonPPSA"), label="non-PPSA"
    )
    plt.title("Weighted AvgTaxRate")
    add_divider(plt.gca())
    plt.xticks(years_w, rotation=90)
    plt.savefig(PLOTS_DIR / "weighted_avgtaxrate.png")
    plt.close()

    plt.figure()
    sns.lineplot(
        x=years_w, y=get_vals(df_weighted, "AvgTaxRate_PPSA_yoy"), label="PPSA"
    )
    sns.lineplot(
        x=years_w, y=get_vals(df_weighted, "AvgTaxRate_nonPPSA_yoy"), label="non-PPSA"
    )
    plt.title("Weighted YoY AvgTaxRate (%)")
    add_divider(plt.gca())
    plt.xticks(years_w, rotation=90)
    plt.savefig(PLOTS_DIR / "weighted_yoy.png")
    plt.close()

    plt.figure()
    sns.lineplot(x=years_w, y=get_vals(df_weighted, "AvgTaxRate_diff"))
    plt.title("Weighted non-PPSA - PPSA Difference")
    add_divider(plt.gca())
    plt.xticks(years_w, rotation=90)
    plt.savefig(PLOTS_DIR / "weighted_diff.png")
    plt.close()


# %%
def add_divider(ax):
    ax.axvline(DIVIDE_YEAR, color="red", linestyle="--")


def get_vals(df, col):
    return df.select(col).to_numpy().ravel()


# %%
if __name__ == "__main__":
    main()
