# %%
import os

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns

# %%
WD = os.path.dirname(__file__)
SOURCE = os.path.join(WD, "..", "..", "data_master.xlsx")
DIVIDE_YEAR = 2012
GROUP_SIZE = 13

# %%
df = (
    pl.read_excel(SOURCE)
    .select(
        [
            "Year",
            "Municipality",
            "AvgTaxRate",
            "PolExpCapita",
            "Provider_PPSA",
            "LatestCensusPop",
        ]
    )
    .filter(pl.col("Provider_PPSA"))
    .drop("Provider_PPSA")
)


# %%
all_muns = df.select("Municipality").unique()

polexp_diff = (
    df.filter(pl.col("Year").is_in([2011, 2012]))
    .select(["Municipality", "Year", "PolExpCapita"])
    .pivot(index="Municipality", columns="Year", values="PolExpCapita")
)

polexp_diff = (
    all_muns.join(polexp_diff, on="Municipality", how="left")
    .with_columns(
        pl.when((pl.col("2011").is_not_null()) & (pl.col("2012").is_not_null()))
        .then(pl.col("2012") - pl.col("2011"))
        .otherwise(999999)
        .alias("polexp_diff_2011_2012")
    )
    .select(["Municipality", "polexp_diff_2011_2012"])
    .sort("polexp_diff_2011_2012")
)

unique_muns = polexp_diff.select("Municipality").to_series().to_list()

# %%
group_assignments = {}

for i, mun in enumerate(unique_muns):
    group_id = i // GROUP_SIZE
    group_assignments[mun] = group_id + 1

df = df.with_columns(
    pl.col("Municipality")
    .map_elements(lambda x: group_assignments[x], pl.Int16)
    .alias("Group")
)

with open(os.path.join(WD, "group_assignments.txt"), "w") as f:
    groups_dict = {}

    for mun, group_id in group_assignments.items():
        if group_id not in groups_dict:
            groups_dict[group_id] = []

        groups_dict[group_id].append(mun)

    for group_id in sorted(groups_dict.keys()):
        f.write(f"Group {group_id} ({len(groups_dict[group_id])} municipalities):\n")

        for mun in sorted(groups_dict[group_id]):
            f.write(f"  - {mun}\n")

        f.write("\n")

# %%
df_unweighted = (
    df.group_by(["Year", "Group"])
    .agg(
        pl.col("AvgTaxRate").mean().alias("AvgTaxRate"),
        pl.col("PolExpCapita").mean().alias("PolExpCapita"),
        pl.col("LatestCensusPop").mean().alias("LatestCensusPop"),
    )
    .sort(["Group", "Year"])
    .with_columns(
        (
            (pl.col("AvgTaxRate") - pl.col("AvgTaxRate").shift(1))
            / pl.col("AvgTaxRate").shift(1)
            * 100
        )
        .over("Group")
        .alias("AvgTaxRate_yoy"),
        (
            (pl.col("PolExpCapita") - pl.col("PolExpCapita").shift(1))
            / pl.col("PolExpCapita").shift(1)
            * 100
        )
        .over("Group")
        .alias("PolExpCapita_yoy"),
    )
)

# %%
df_weighted = (
    df.group_by(["Year", "Group"])
    .agg(
        (
            (pl.col("AvgTaxRate") * pl.col("LatestCensusPop")).sum()
            / pl.col("LatestCensusPop").sum()
        ).alias("AvgTaxRate"),
        (
            (pl.col("PolExpCapita") * pl.col("LatestCensusPop")).sum()
            / pl.col("LatestCensusPop").sum()
        ).alias("PolExpCapita"),
    )
    .sort(["Group", "Year"])
    .with_columns(
        (
            (pl.col("AvgTaxRate") - pl.col("AvgTaxRate").shift(1))
            / pl.col("AvgTaxRate").shift(1)
            * 100
        )
        .over("Group")
        .alias("AvgTaxRate_yoy"),
        (
            (pl.col("PolExpCapita") - pl.col("PolExpCapita").shift(1))
            / pl.col("PolExpCapita").shift(1)
            * 100
        )
        .over("Group")
        .alias("PolExpCapita_yoy"),
    )
)


# %%
def add_divider(ax):
    ax.axvline(DIVIDE_YEAR - 1, color="red", linestyle="--")


# %%
def plot_by_group(df_data, metric, title_prefix, filename):
    plt.figure()
    for group_id in sorted(df_data.select("Group").unique().to_series().to_list()):
        if group_id in {3, 4}:
            continue

        group_data = df_data.filter(pl.col("Group") == group_id)
        years = group_data.select("Year").to_series().to_list()
        values = group_data.select(metric).to_series().to_list()
        sns.lineplot(x=years, y=values, label=f"Group {group_id}")

    plt.title(f"{title_prefix} {metric}")
    add_divider(plt.gca())
    all_years = sorted(df_data.select("Year").unique().to_series().to_list())
    plt.xticks(all_years, rotation=90)
    plt.legend()
    plt.savefig(os.path.join(WD, filename))
    plt.close()


# %%
plot_by_group(
    df_unweighted, "AvgTaxRate", "Unweighted", "unweighted_avgtaxrate_groups.png"
)
plot_by_group(
    df_unweighted,
    "AvgTaxRate_yoy",
    "Unweighted",
    "unweighted_avgtaxrate_yoy_groups.png",
)
plot_by_group(
    df_unweighted, "PolExpCapita", "Unweighted", "unweighted_polexpcapita_groups.png"
)
plot_by_group(
    df_unweighted,
    "PolExpCapita_yoy",
    "Unweighted",
    "unweighted_polexpcapita_yoy_groups.png",
)

plot_by_group(df_weighted, "AvgTaxRate", "Weighted", "weighted_avgtaxrate_groups.png")
plot_by_group(
    df_weighted, "AvgTaxRate_yoy", "Weighted YoY", "weighted_avgtaxrate_yoy_groups.png"
)
plot_by_group(
    df_weighted, "PolExpCapita", "Weighted", "weighted_polexpcapita_groups.png"
)
plot_by_group(
    df_weighted,
    "PolExpCapita_yoy",
    "Weighted YoY",
    "weighted_polexpcapita_yoy_groups.png",
)

plot_by_group(
    df_unweighted,
    "LatestCensusPop",
    "",
    "unweighted_latestcensuspop_groups.png",
)
