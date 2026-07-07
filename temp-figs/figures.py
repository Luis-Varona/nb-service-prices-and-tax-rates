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
DATA_DIR = WD.parent / "data" / "data_final"
PLOTS_DIR = WD / "plots"


# %%
DIVIDE_YEAR = 2012
GROUP_SIZE = 13
METRICS = ["PolExpCapita", "OtherExpCapita", "PolFrac"]


# %%
def main() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data()
    plot_provider_trends(df)
    plot_shock_groups(df)


# %%
def load_data() -> pl.DataFrame:
    df_master = pl.read_excel(DATA_DIR / "data_master.xlsx").select(
        [
            "Year",
            "Municipality",
            "PolExpCapita",
            "OtherExpCapita",
            "Provider_PPSA",
            "Provider_MPSA",
            "LatestCensusPop",
        ]
    )
    df_pol = pl.read_excel(DATA_DIR / "data_pol_prov.xlsx")

    return df_master.join(df_pol, on="Municipality").with_columns(
        (
            pl.col("PolExpCapita") / (pl.col("PolExpCapita") + pl.col("OtherExpCapita"))
        ).alias("PolFrac")
    )


# %%
def plot_provider_trends(df: pl.DataFrame) -> None:
    for metric in METRICS:
        df_weighted = weighted_by_provider(df, metric)
        years = sorted(df_weighted.select("Year").unique().to_series().to_list())

        plt.figure()
        for provider in ["PPSA", "MPSA", "Municipal"]:
            group_data = df_weighted.filter(pl.col("Policing Provider") == provider)
            sns.lineplot(
                x=group_data.select("Year").to_series(),
                y=group_data.select(metric).to_series(),
                label=provider,
            )
        plt.title(f"Population-Weighted {metric}")
        add_divider(plt.gca())
        plt.xticks(years, rotation=90)
        plt.legend()
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / f"provider_{metric.lower()}.png", dpi=300)
        plt.close()


def weighted_by_provider(df: pl.DataFrame, metric: str) -> pl.DataFrame:
    if metric == "PolFrac":
        return (
            df.group_by(["Year", "Policing Provider"])
            .agg(
                (
                    (pl.col("PolExpCapita") * pl.col("LatestCensusPop")).sum()
                    / (
                        (pl.col("PolExpCapita") + pl.col("OtherExpCapita"))
                        * pl.col("LatestCensusPop")
                    ).sum()
                ).alias("PolFrac")
            )
            .sort(["Policing Provider", "Year"])
        )

    return (
        df.group_by(["Year", "Policing Provider"])
        .agg(
            (
                (pl.col(metric) * pl.col("LatestCensusPop")).sum()
                / pl.col("LatestCensusPop").sum()
            ).alias(metric)
        )
        .sort(["Policing Provider", "Year"])
    )


# %%
def plot_shock_groups(df: pl.DataFrame) -> None:
    df_ppsa = df.filter(pl.col("Provider_PPSA")).drop(
        ["Provider_PPSA", "Provider_MPSA", "Policing Provider"]
    )

    all_munis = df_ppsa.select("Municipality").unique()

    polexp_diff = (
        df_ppsa.filter(pl.col("Year").is_in([2011, 2012]))
        .select(["Municipality", "Year", "PolExpCapita"])
        .pivot(on="Year", index="Municipality", values="PolExpCapita")
    )

    polexp_diff = (
        all_munis.join(polexp_diff, on="Municipality", how="left")
        .with_columns(
            pl.when((pl.col("2011").is_not_null()) & (pl.col("2012").is_not_null()))
            .then(pl.col("2012") - pl.col("2011"))
            .otherwise(999999)
            .alias("polexp_diff")
        )
        .select(["Municipality", "polexp_diff"])
        .sort("polexp_diff")
    )

    munis = polexp_diff.select("Municipality").to_series().to_list()
    group_map = {mun: i // GROUP_SIZE + 1 for i, mun in enumerate(munis)}

    write_group_assignments(group_map)

    df_ppsa = df_ppsa.with_columns(
        pl.col("Municipality")
        .replace_strict(group_map, return_dtype=pl.Int16)
        .alias("Group")
    )

    for metric in METRICS:
        df_weighted = weighted_by_group(df_ppsa, metric)
        years = sorted(df_weighted.select("Year").unique().to_series().to_list())

        plt.figure()
        for group_id in sorted(
            df_weighted.select("Group").unique().to_series().to_list()
        ):
            group_data = df_weighted.filter(pl.col("Group") == group_id)
            sns.lineplot(
                x=group_data.select("Year").to_series(),
                y=group_data.select(metric).to_series(),
                label=f"Group {group_id}",
            )
        plt.title(f"PPSA Population-Weighted {metric} by Shock Group")
        add_divider(plt.gca())
        plt.xticks(years, rotation=90)
        plt.legend()
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / f"shock_{metric.lower()}.png", dpi=300)
        plt.close()


def weighted_by_group(df: pl.DataFrame, metric: str) -> pl.DataFrame:
    if metric == "PolFrac":
        return (
            df.group_by(["Year", "Group"])
            .agg(
                (
                    (pl.col("PolExpCapita") * pl.col("LatestCensusPop")).sum()
                    / (
                        (pl.col("PolExpCapita") + pl.col("OtherExpCapita"))
                        * pl.col("LatestCensusPop")
                    ).sum()
                ).alias("PolFrac")
            )
            .sort(["Group", "Year"])
        )

    return (
        df.group_by(["Year", "Group"])
        .agg(
            (
                (pl.col(metric) * pl.col("LatestCensusPop")).sum()
                / pl.col("LatestCensusPop").sum()
            ).alias(metric)
        )
        .sort(["Group", "Year"])
    )


# %%
def write_group_assignments(group_map: dict[str, int]) -> None:
    groups_dict: dict[int, list[str]] = {}

    for mun, group_id in group_map.items():
        if group_id not in groups_dict:
            groups_dict[group_id] = []
        groups_dict[group_id].append(mun)

    with open(WD / "ppsa_groups.txt", "w") as f:
        for group_id in sorted(groups_dict):
            f.write(
                f"Group {group_id} ({len(groups_dict[group_id])} municipalities):\n"
            )
            for mun in sorted(groups_dict[group_id]):
                f.write(f"  - {mun}\n")
            f.write("\n")


# %%
def add_divider(ax):
    ax.axvline(DIVIDE_YEAR, color="red", linestyle="--")


# %%
if __name__ == "__main__":
    main()
