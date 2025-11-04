# %%
import os

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns


# %%
WD = os.path.dirname(__file__)
SOURCE = os.path.join(WD, "..", "data_master.xlsx")
DIVIDE_YEAR = 2012


# %%
df = pl.read_excel(SOURCE).select(
    [
        "Year",
        "AvgTaxRate",
        "Provider_PPSA",
        "LatestCensusPop",
    ]
)


# %%
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


# %%
def add_divider(ax):
    ax.axvline(DIVIDE_YEAR, color="red", linestyle="--")


def get_vals(df, col):
    return df.select(col).to_numpy().ravel()


years_unw = get_vals(df_unweighted, "Year").astype(int)
years_w = get_vals(df_weighted, "Year").astype(int)


# %%
plt.figure()
sns.lineplot(x=years_unw, y=get_vals(df_unweighted, "AvgTaxRate_PPSA"), label="PPSA")
sns.lineplot(
    x=years_unw, y=get_vals(df_unweighted, "AvgTaxRate_nonPPSA"), label="non-PPSA"
)
plt.title("Unweighted AvgTaxRate")
add_divider(plt.gca())
plt.xticks(years_unw, rotation=90)
plt.savefig(os.path.join(WD, "unweighted_avgtaxrate.png"))
plt.close()

plt.figure()
sns.lineplot(
    x=years_unw, y=get_vals(df_unweighted, "AvgTaxRate_PPSA_yoy"), label="PPSA"
)
sns.lineplot(
    x=years_unw, y=get_vals(df_unweighted, "AvgTaxRate_nonPPSA_yoy"), label="non-PPSA"
)
plt.title("Unweighted YoY AvgTaxRate (%)")
add_divider(plt.gca())
plt.xticks(years_unw, rotation=90)
plt.savefig(os.path.join(WD, "unweighted_yoy.png"))
plt.close()

plt.figure()
sns.lineplot(x=years_unw, y=get_vals(df_unweighted, "AvgTaxRate_diff"))
plt.title("Unweighted non-PPSA - PPSA Difference")
add_divider(plt.gca())
plt.xticks(years_unw, rotation=90)
plt.savefig(os.path.join(WD, "unweighted_diff.png"))
plt.close()

plt.figure()
sns.lineplot(x=years_w, y=get_vals(df_weighted, "AvgTaxRate_PPSA"), label="PPSA")
sns.lineplot(x=years_w, y=get_vals(df_weighted, "AvgTaxRate_nonPPSA"), label="non-PPSA")
plt.title("Weighted AvgTaxRate")
add_divider(plt.gca())
plt.xticks(years_w, rotation=90)
plt.savefig(os.path.join(WD, "weighted_avgtaxrate.png"))
plt.close()

plt.figure()
sns.lineplot(x=years_w, y=get_vals(df_weighted, "AvgTaxRate_PPSA_yoy"), label="PPSA")
sns.lineplot(
    x=years_w, y=get_vals(df_weighted, "AvgTaxRate_nonPPSA_yoy"), label="non-PPSA"
)
plt.title("Weighted YoY AvgTaxRate (%)")
add_divider(plt.gca())
plt.xticks(years_w, rotation=90)
plt.savefig(os.path.join(WD, "weighted_yoy.png"))
plt.close()

plt.figure()
sns.lineplot(x=years_w, y=get_vals(df_weighted, "AvgTaxRate_diff"))
plt.title("Weighted non-PPSA - PPSA Difference")
add_divider(plt.gca())
plt.xticks(years_w, rotation=90)
plt.savefig(os.path.join(WD, "weighted_diff.png"))
plt.close()
