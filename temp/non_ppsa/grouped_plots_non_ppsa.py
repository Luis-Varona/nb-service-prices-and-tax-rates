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
DIVIDE_YEAR = 2012
GROUP_SIZE = 4


# %%
df = (
    pl.read_excel(SRC)
    .select(
        [
            "Year",
            "Municipality",
            "PolExpCapita",
            "OtherExpCapita",
            "Provider_PPSA",
        ]
    )
    .filter(~pl.col("Provider_PPSA"))
    .with_columns(
        (
            pl.col("PolExpCapita") / (pl.col("PolExpCapita") + pl.col("OtherExpCapita"))
        ).alias("PolExpShare")
    )
)


# %%
unique_muns = (
    df.group_by("Municipality")
    .agg(pl.col("PolExpShare").mean().alias("avg_polexpshare"))
    .sort("avg_polexpshare")
    .select("Municipality")
    .to_series()
    .to_list()
)


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

with open(WD / "group_assignments_non_ppsa.txt", "w") as f:
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
        pl.col("PolExpShare").mean().alias("PolExpShare"),
    )
    .sort(["Group", "Year"])
    .with_columns(
        (
            (pl.col("PolExpShare") - pl.col("PolExpShare").shift(1))
            / pl.col("PolExpShare").shift(1)
            * 100
        )
        .over("Group")
        .alias("PolExpShare_yoy"),
    )
)

# %%
df_weighted = (
    df.group_by(["Year", "Group"])
    .agg(
        (
            (pl.col("PolExpShare") * pl.col("PolExpCapita")).sum()
            / pl.col("PolExpCapita").sum()
        ).alias("PolExpShare"),
    )
    .sort(["Group", "Year"])
    .with_columns(
        (
            (pl.col("PolExpShare") - pl.col("PolExpShare").shift(1))
            / pl.col("PolExpShare").shift(1)
            * 100
        )
        .over("Group")
        .alias("PolExpShare_yoy"),
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
    plt.savefig(WD / filename)
    plt.close()


# %%
plot_by_group(
    df_unweighted,
    "PolExpShare",
    "Non-PPSA: Unweighted",
    "unweighted_polexpshare_groups_non_ppsa.png",
)
plot_by_group(
    df_unweighted,
    "PolExpShare_yoy",
    "Non-PPSA: Unweighted",
    "unweighted_polexpshare_yoy_groups_non_ppsa.png",
)

plot_by_group(
    df_weighted,
    "PolExpShare",
    "Non-PPSA: Weighted",
    "weighted_polexpshare_groups_non_ppsa.png",
)

plot_by_group(
    df_weighted,
    "PolExpShare_yoy",
    "Non-PPSA: Weighted",
    "weighted_polexpshare_yoy_groups_non_ppsa.png",
)
