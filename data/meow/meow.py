# Copyright 2025 Craig Brett and Luis M. B. Varona
#
# Licensed under the MIT license <LICENSE or
# http://opensource.org/licenses/MIT>. This file may not be copied, modified, or
# distributed except according to those terms.

from pathlib import Path

import polars as pl

WD = Path(__file__).parent
DATA_OLD = WD.parent / "data_final"

POL_PROV_RENAMES = {
    "Municipality": "municipality",
    "Policing Provider": "police_provider_2024",
}
TAX_BASE_RENAMES = {
    "Year": "year",
    "Municipality": "municipality",
    "Total Tax Base for Rate": "rateable_tax_base",
}
CMP_DEMO_RENAMES = {
    "Year": "year",
    "Municipality": "municipality",
    "Latest Census Population": "latest_census_pop",
    "Penultimate Census Population": "penult_census_pop",
    "Average Tax Rate": "average_tax_rate",
}
BGT_EXPS_RENAMES = {
    "Year": "year",
    "Municipality": "municipality",
    "General Government": "general_govt_exp",
    "Police": "police_exp",
    "Fire Protection": "fire_protection_exp",
    "Water Cost Transfer": "water_cost_exp",
    "Emergency Measures": "emergency_exp",
    "Other Protection Services": "other_protection_exp",
    "Transportation": "transportation_exp",
    "Environmental Health": "environ_health_exp",
    "Public Health": "public_health_exp",
    "Environmental Development": "environ_dev_exp",
    "Recreation & Cultural": "rec_and_culture_exp",
    "Debt Costs": "debt_exp",
    "Transfers": "transfers_exp",
    "Deficits": "deficits_exp",
    "Total Expenditures": "total_exp",
}
BGT_REVS_RENAMES = {
    "Year": "year",
    "Municipality": "municipality",
    "Warrant": "warrant_rev",
    "Unconditional Grant": "uncond_grant_rev",
    "Services to Other Governments": "other_govt_serv_rev",
    "Sale of Services": "sale_of_serv_rev",
    "Own-Source Revenue": "own_source_rev",
    "Conditional Transfers": "cond_transfer_rev",
    "Other Transfers": "other_transfer_rev",
    "Biennial Surplus": "biennial_surplus_rev",
    "Total Revenue": "total_rev",
}

POL_PROV_COLS = ["municipality", "police_provider_2024"]
CMP_DEMO_COLS = [
    "year",
    "municipality",
    "latest_census_pop",
    "penult_census_pop",
    "average_tax_rate",
    "rateable_tax_base",
]
BGT_EXPS_COLS = [
    "year",
    "municipality",
    "general_govt_exp",
    "police_exp",
    "fire_protection_exp",
    "water_cost_exp",
    "emergency_exp",
    "other_protection_exp",
    "transportation_exp",
    "environ_health_exp",
    "public_health_exp",
    "environ_dev_exp",
    "rec_and_culture_exp",
    "debt_exp",
    "transfers_exp",
    "deficits_exp",
    "total_exp",
]
BGT_REVS_COLS = [
    "year",
    "municipality",
    "warrant_rev",
    "uncond_grant_rev",
    "other_govt_serv_rev",
    "sale_of_serv_rev",
    "own_source_rev",
    "cond_transfer_rev",
    "other_transfer_rev",
    "biennial_surplus_rev",
    "total_rev",
]


def main() -> None:
    df_cpi_defl_2002 = pl.read_csv(WD / "cpi_defl_2002.csv")

    df_pol_prov_2024 = (
        pl.read_excel(DATA_OLD / "data_pol_prov.xlsx")
        .rename(POL_PROV_RENAMES)
        .with_columns(
            pl.col("police_provider_2024").str.replace("Municipal", "municipal")
        )
        .select(POL_PROV_COLS)
        .sort("municipality")
    )

    df_cmp_demo = (
        pl.read_excel(DATA_OLD / "data_cmp_data.xlsx")
        .rename(CMP_DEMO_RENAMES)
        .select(list(CMP_DEMO_RENAMES.values()))
    )

    df_tax_base = (
        pl.read_excel(DATA_OLD / "data_tax_base.xlsx")
        .select(["Year", "Municipality", "Total Tax Base for Rate"])
        .rename(TAX_BASE_RENAMES)
    )

    df_bgt_exps = (
        pl.read_excel(DATA_OLD / "data_bgt_exps.xlsx")
        .rename(BGT_EXPS_RENAMES)
        .select(BGT_EXPS_COLS)
    )

    df_bgt_revs = (
        pl.read_excel(DATA_OLD / "data_bgt_revs.xlsx")
        .rename(BGT_REVS_RENAMES)
        .select(BGT_REVS_COLS)
    )

    df_full = add_derived_cols(
        df_cmp_demo.join(df_tax_base, on=["year", "municipality"], how="left")
        .join(df_bgt_exps, on=["year", "municipality"], how="left")
        .join(df_bgt_revs, on=["year", "municipality"], how="left")
        .join(df_cpi_defl_2002, on="year", how="left")
        .join(df_pol_prov_2024, on="municipality", how="left")
    )

    main_cols = expand_col_list(
        list(
            dict.fromkeys(
                ["year", "cpi_2002", "cpi_deflator_2002"]
                + POL_PROV_COLS
                + CMP_DEMO_COLS
                + BGT_EXPS_COLS
                + BGT_REVS_COLS
            )
        ),
        "_base",
    )

    year_min = df_full["year"].min()
    year_max = df_full["year"].max()
    years = range(year_min, year_max + 1)

    df_main = df_full.select(main_cols).sort("year", "municipality")
    df_cmp_demo_out = df_full.select(expand_col_list(CMP_DEMO_COLS, "_base")).sort(
        "year", "municipality"
    )
    df_bgt_exps_out = df_full.select(expand_col_list(BGT_EXPS_COLS, "_exp")).sort(
        "year", "municipality"
    )
    df_bgt_revs_out = df_full.select(expand_col_list(BGT_REVS_COLS, "_rev")).sort(
        "year", "municipality"
    )

    df_main.write_csv(WD / "main.csv")
    df_pol_prov_2024.write_csv(WD / "pol_prov_2024.csv")
    df_cmp_demo_out.write_csv(WD / "cmp_demo.csv")
    df_bgt_exps_out.write_csv(WD / "bgt_exps.csv")
    df_bgt_revs_out.write_csv(WD / "bgt_revs.csv")

    for year in years:
        year_dir = WD / str(year)
        year_dir.mkdir(parents=True, exist_ok=True)

        df_main.filter(pl.col("year") == year).drop("year").write_csv(
            year_dir / f"main_{year}.csv"
        )
        df_cmp_demo_out.filter(pl.col("year") == year).drop("year").write_csv(
            year_dir / f"cmp_demo_{year}.csv"
        )
        df_bgt_exps_out.filter(pl.col("year") == year).drop("year").write_csv(
            year_dir / f"bgt_exps_{year}.csv"
        )
        df_bgt_revs_out.filter(pl.col("year") == year).drop("year").write_csv(
            year_dir / f"bgt_revs_{year}.csv"
        )


def expand_col_list(cols: list[str], suffix: str) -> list[str]:
    cols_new: list[str] = []

    for c in cols:
        cols_new.append(c)

        if c.endswith(suffix):
            cols_new.extend([f"{c}_adj", f"{c}_capita", f"{c}_capita_adj"])

    return cols_new


def add_derived_cols(df: pl.DataFrame) -> pl.DataFrame:
    cols_monetary = [c for c in df.columns if c.endswith(("_base", "_rev", "_exp"))]

    return df.with_columns(
        *[
            (pl.col(c) * pl.col("cpi_deflator_2002")).alias(f"{c}_adj")
            for c in cols_monetary
        ],
        *[
            (pl.col(c) / pl.col("latest_census_pop")).alias(f"{c}_capita")
            for c in cols_monetary
        ],
        *[
            (
                pl.col(c) * pl.col("cpi_deflator_2002") / pl.col("latest_census_pop")
            ).alias(f"{c}_capita_adj")
            for c in cols_monetary
        ],
    )


if __name__ == "__main__":
    main()
