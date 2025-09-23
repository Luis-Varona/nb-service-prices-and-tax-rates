# Copyright 2025 Craig Brett and Luis M. B. Varona
#
# Licensed under the MIT license <LICENSE or
# http://opensource.org/licenses/MIT>. This file may not be copied, modified, or
# distributed except according to those terms.


# %%
from pathlib import Path

import polars as pl
import polars.selectors as cs


# %%
WD = Path(__file__).parent
DATA_DIR = WD.parent.parent / "data"
SRC_DIR = DATA_DIR / "data_clean"
DST_DIR = DATA_DIR / "data_final"


# %%
YEARS = range(2000, 2021)

CATEGORIES = ["bgt_revs", "bgt_exps", "cmp_data", "tax_base"]

SCHEMAS_MASTER = {
    "bgt_exps": pl.Schema(
        {
            "Municipality": pl.Utf8,
            "General Government": pl.Float64,
            "Police": pl.Int64,
            "Fire Protection": pl.Int64,
            "Water Cost Transfer": pl.Int64,
            "Emergency Measures": pl.Int64,
            "Other Protection Services": pl.Int64,
            "Transportation": pl.Int64,
            "Environmental Health": pl.Int64,
            "Public Health": pl.Int64,
            "Environmental Development": pl.Int64,
            "Recreation & Cultural": pl.Int64,
            "Debt Costs": pl.Float64,
            "Transfers": pl.Int64,
            "Deficits": pl.Int64,
            "Total Expenditures": pl.Float64,
        }
    ),
    "bgt_revs": pl.Schema(
        {
            "Municipality": pl.Utf8,
            "Warrant": pl.Int64,
            "Unconditional Grant": pl.Int64,
            "Services to Other Governments": pl.Int64,
            "Sale of Services": pl.Int64,
            "Own-Source Revenue": pl.Int64,
            "Conditional Transfers": pl.Int64,
            "Other Transfers": pl.Int64,
            "Biennial Surplus": pl.Int64,
            "Total Revenue": pl.Int64,
        }
    ),
    "cmp_data": pl.Schema(
        {
            "Municipality": pl.Utf8,
            "Latest Census Population": pl.Int64,
            "Penultimate Census Population": pl.Int64,
            "Provincial Kilometrage": pl.Float64,
            "Regional Kilometrage": pl.Float64,
            "Municipal Kilometrage": pl.Float64,
            "Total Kilometrage": pl.Float64,
            "Population/Kilometrage": pl.Float64,
            "Tax Base": pl.Int64,
            "Tax Base/Capita": pl.Float64,
            "Tax Base/Kilometrage": pl.Float64,
            "Total Budget": pl.Int64,
            "Fiscal Capacity": pl.Float64,
            "Average Tax Rate": pl.Float64,
        }
    ),
    "tax_base": pl.Schema(
        {
            "Municipality": pl.Utf8,
            "General Residential Assessment": pl.Int64,
            "Federal Residential Assessment": pl.Int64,
            "Provincial Residential Assessment": pl.Int64,
            "Total Residential Assessment": pl.Int64,
            "General Non-Residential Assessment": pl.Int64,
            "Federal Non-Residential Assessment": pl.Int64,
            "Provincial Non-Residential Assessment": pl.Int64,
            "Total Non-Residential Assessment": pl.Int64,
            "Total Municipal Assessment Base": pl.Int64,
            "Total Municipal Tax Base": pl.Int64,
            "Total Tax Base for Rate": pl.Int64,
        }
    ),
    "pol_prov": pl.Schema(
        {
            "District": pl.Utf8,
            "Municipality": pl.Utf8,
            "Policing Provider": pl.Utf8,
        }
    ),
}

COLUMNS_MASTER = [
    "Year",
    "Municipality",
    "AvgTaxRate",
    "PolExpCapita",
    "OtherExpCapita",
    "OtherRevCapita",
    "TaxBaseCapita",
    "Provider_PPSA",
    "Provider_MPSA",
    "LatestCensusPop",
]

COLUMNS_SCALE = ["PolExp", "TotalExp", "TaxRev", "TotalRev", "TaxBase"]

SCALE_FACTOR = 1 / 1_000

MAPS_MASTER = {
    "bgt_exps": {
        "Year": "Year",
        "Municipality": "Municipality",
        "Police": "PolExp",
        "Total Expenditures": "TotalExp",
    },
    "bgt_revs": {
        "Year": "Year",
        "Municipality": "Municipality",
        "Warrant": "TaxRev",
        "Total Revenue": "TotalRev",
    },
    "cmp_data": {
        "Year": "Year",
        "Municipality": "Municipality",
        "Latest Census Population": "LatestCensusPop",
        "Average Tax Rate": "AvgTaxRate",
    },
    "tax_base": {
        "Year": "Year",
        "Municipality": "Municipality",
        "Total Tax Base for Rate": "TaxBase",
    },
}


# %%
def main() -> None:
    dfs_final = convert_clean_to_final()
    df_master = convert_final_to_master(dfs_final)

    DST_DIR.mkdir(parents=True, exist_ok=True)

    for cat, df in dfs_final.items():
        dst = DST_DIR / f"data_{cat}.xlsx"
        df.write_excel(dst, header_format={"bold": True}, autofit=True)

    dst_master = DST_DIR / "data_master.xlsx"
    df_master.write_excel(dst_master, header_format={"bold": True}, autofit=True)


# %%
def convert_clean_to_final() -> dict[str, pl.DataFrame]:
    dfs_concat = concat_panels_by_cat()
    muni_list = dfs_concat["cmp_data"].to_series(1).unique()
    df_pol_prov = melt_pol_prov_data(muni_list)

    return dfs_concat | {"pol_prov": df_pol_prov}


def concat_panels_by_cat() -> dict[str, pl.DataFrame]:
    src_year_dirs = {year: SRC_DIR / str(year) for year in YEARS}
    files = {
        cat: {year: next(src_year_dirs[year].glob(f"*_{cat}.xlsx")) for year in YEARS}
        for cat in CATEGORIES
    }

    return {
        cat: pl.concat(
            pl.read_excel(source, schema_overrides=SCHEMAS_MASTER[cat])
            .with_columns(pl.lit(year, pl.UInt32).alias("Year"))
            .select("Year", cs.exclude("Year"))
            for year, source in files[cat].items()
        )
        for cat in CATEGORIES
    }


def melt_pol_prov_data(muni_list: pl.Series) -> pl.DataFrame:
    src_pol_prov = next(SRC_DIR.glob("*_pol_prov.xlsx"))
    df = pl.read_excel(src_pol_prov, schema_overrides=SCHEMAS_MASTER["pol_prov"])
    prov_map = {}

    for dist in df.select("District").to_series().unique():
        if dist in muni_list:
            providers = df.filter(pl.col("District") == dist).to_series(2).unique()

            if len(providers) > 1:
                raise ValueError(
                    f"District {dist} has multiple policing providers: {providers}"
                )

            prov_map[dist] = providers[0]

    for muni in df.select("Municipality").to_series().unique():
        if muni in muni_list:
            providers = df.filter(pl.col("Municipality") == muni).to_series(2).unique()

            if len(providers) > 1:
                raise RuntimeError(
                    f"Municipality {muni} has multiple policing providers: {providers}"
                )

            if muni in prov_map and prov_map[muni] != providers[0]:
                raise RuntimeError(
                    f"Municipality {muni} has different policing providers: "
                    f"{prov_map[muni]} and {providers[0]}"
                )

            prov_map[muni] = providers[0]

    prov_map["Sussex Corner"] = prov_map["Sussex"]
    missing_munis = set(muni_list) - prov_map.keys()

    if missing_munis:
        raise RuntimeError(
            "Missing policing provider data for the following "
            f"municipalities: {missing_munis}"
        )

    muni_data, prov_data = zip(*prov_map.items())

    return pl.DataFrame(
        {
            "Municipality": muni_data,
            "Policing Provider": prov_data,
        }
    ).sort("Municipality")


# %%
def convert_final_to_master(dfs_final: dict[str, pl.DataFrame]) -> pl.DataFrame:
    dfs_inter = {
        cat: dfs_final[cat].select(MAPS_MASTER[cat].keys()).rename(MAPS_MASTER[cat])
        for cat in CATEGORIES
    }
    df_master = dfs_inter[CATEGORIES[0]]
    muni_list = set(df_master.to_series(1))

    if any(set(dfs_inter[cat].to_series(1)) != muni_list for cat in CATEGORIES[1:]):
        raise RuntimeError("Municipalities are not the same across datasets.")

    for cat in CATEGORIES[1:]:
        df_master = df_master.join(dfs_inter[cat], ["Year", "Municipality"])

    prov_map = dict(zip(*dfs_final["pol_prov"]))

    return (
        df_master.with_columns(pl.col("AvgTaxRate") / 100)
        .with_columns(pl.col(COLUMNS_SCALE) * SCALE_FACTOR)
        .with_columns(
            (pl.col("TaxBase") / pl.col("LatestCensusPop")).alias("TaxBaseCapita")
        )
        .with_columns(
            (pl.col("PolExp") / pl.col("LatestCensusPop")).alias("PolExpCapita")
        )
        .with_columns(
            ((pl.col("TotalExp") - pl.col("PolExp")) / pl.col("LatestCensusPop")).alias(
                "OtherExpCapita"
            )
        )
        .with_columns(
            ((pl.col("TotalRev") - pl.col("TaxRev")) / pl.col("LatestCensusPop")).alias(
                "OtherRevCapita"
            )
        )
        .with_columns(pl.col("Municipality").replace(prov_map).alias("Provider"))
        .to_dummies("Provider")
        .with_columns(cs.matches("Provider_*").cast(pl.Boolean))
        .select(COLUMNS_MASTER)
    )


# %%
if __name__ == "__main__":
    main()
