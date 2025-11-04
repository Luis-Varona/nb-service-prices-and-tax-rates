# Copyright 2025 Craig Brett and Luis M. B. Varona
#
# Licensed under the MIT license <LICENSE or
# http://opensource.org/licenses/MIT>. This file may not be copied, modified, or
# distributed except according to those terms.


# %%
import re

from io import BytesIO
from pathlib import Path
from sys import path

import polars as pl
import polars.selectors as cs


# %%
WD = Path(__file__).parent
path.append(str(WD.parent))

from utils import suppress_fastexcel_logging  # noqa: E402


# %%
DATA_DIR = WD.parent.parent / "data"
SRC_DIR = DATA_DIR / "data_xlsx"
DST_DIR = DATA_DIR / "data_clean"


# %%
@suppress_fastexcel_logging
def main() -> None:
    write_clean_pol_prov_data()
    write_clean_bgt_exps_data()
    write_clean_bgt_revs_data()
    write_clean_cmp_data()
    write_clean_tax_base_data()


# %%
def clean_munis(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        pl.col("Municipality")
        .str.to_titlecase()
        .str.replace_all(r"\s[-\(].*", "")
        .str.strip_chars()
        .str.replace_all(r"\n\s*", "")
        .str.replace_all(r"_X000(D|d)_", "")
        .str.replace_all(r"\\", "/")
        .str.replace_all(r"-\s*", "-")
        .str.replace_all(" De ", " de ")
        .str.replace_all("-De-", "-de-")
        .str.replace(r"^Aroostock$", "Aroostook")
        .str.replace(r"^Baker Brook$", "Baker-Brook")
        .str.replace(r"^Grande Anse$", "Grande-Anse")
        .str.replace(r"^Grand Bay/Westfield$", "Grand Bay-Westfield")
        .str.replace(r"^Grand-Falls/Grand-Sault$", "Grand Falls/Grand-Sault")
        .str.replace(r"^Grand-Sault\s*/\s*Grand(\s|-)Falls$", "Grand Falls/Grand-Sault")
        .str.replace(r"^Lameque$", "Lamèque")
        .str.replace(r"^Mcadam$", "McAdam")
        .str.replace(r"^Neguac$", "Néguac")
        .str.replace(r"^Saint-Francois-de-Madawaska$", "Saint-François-de-Madawaska")
        .str.replace(r"^Saint-Louis de Kent$", "Saint-Louis-de-Kent")
        .str.replace(r"^Sainte-Marie-Saint-Rapha(e|ê)l$", "Sainte-Marie-Saint-Raphaël")
        .str.replace(r"^Shédiac$", "Shediac")
        .str.replace(r"^St-Hilaire$", "Saint-Hilaire")
        .str.replace(r"^St-Isidore$", "Saint-Isidore")
        .str.replace(r"^St. Andrews$", "Saint Andrews")
        .str.replace(r"^St. André$", "Saint-André")
        .str.replace(r"^St. George$", "Saint George")
        .str.replace(r"^St. Hilaire$", "Saint-Hilaire")
        .str.replace(r"^St. Léonard$", "Saint-Léonard")
        .str.replace(r"^Ste-Anne-de-Madawaska$", "Sainte-Anne-de-Madawaska")
        .str.replace(r"^Town (O|o)f Rothesay$", "Rothesay")
        .str.replace(r"^(Village de )?Lac(\s|-)Baker$", "Lac Baker")
    )


# %%
def write_clean_pol_prov_data() -> None:
    file = next(SRC_DIR.rglob("*_pol_prov.xlsx"))
    dst = DST_DIR / file.relative_to(SRC_DIR)
    dst.parent.mkdir(parents=True, exist_ok=True)

    df = clean_pol_prov_data(file)
    df.write_excel(dst, header_format={"bold": True}, autofit=True)


def clean_pol_prov_data(file: Path) -> pl.DataFrame:
    df_init = (
        pl.read_excel(file, columns=[0, 8])
        .filter(pl.nth(0).map_elements(len, pl.Int64) < 81)
        .with_columns(
            pl.nth(1)
            .str.slice(0, 4)
            .str.to_uppercase()
            .str.replace(r"MUNI|BNPP|KVPF", "Municipal")
        )
    )

    muni_map = _get_pol_prov_muni_map(df_init)

    municipalities = [muni for muni in muni_map for _ in muni_map[muni]]
    districts, providers = zip(
        *(token for value in muni_map.values() for token in value)
    )

    return (
        pl.DataFrame(
            {
                "District": districts,
                "Municipality": municipalities,
                "Policing Provider": providers,
            }
        )
        .filter(pl.col("District").str.contains(r"\s(C|TV|V)$"))
        .with_columns(
            pl.col("District")
            .str.replace(r"\s(C|TV|V)$", "")
            .str.replace(r"^Baker Brook$", "Baker-Brook")
            .str.replace(r"^Cambridge Narrows$", "Cambridge-Narrows")
            .str.replace(r"^Neguac", "Néguac")
            .str.replace(r"^Plaster Rocker$", "Plaster Rock")
            .str.replace(r"^Saint-Anne$", "Sainte-Anne-de-Madawaska")
            .str.replace(r"^Saint-François$", "Saint-François-de-Madawaska")
            .str.replace(r"^Ste-Marie-St-Raphael$", "Sainte-Marie-Saint-Raphaël")
            .str.replace(r"^Tracadie$", "Tracadie-Sheila")
            .str.replace(r"^Eel Riv.*", "Eel River Crossing")
            .str.replace(r"^Grand-Sault.*", "Grand Falls/Grand-Sault")
            .str.replace(r"^Nackawic.*", "Nackawic")
        )
        .with_columns(
            pl.col("Municipality")
            .str.replace(r"^Tracadie$", "Tracadie-Sheila")
            .str.replace(r"^Grand-Sault.*", "Grand Falls/Grand-Sault")
            .str.replace(r"^Nackawic.*", "Nackawic")
        )
    )


def _get_pol_prov_muni_map(df_init: pl.DataFrame) -> dict[str, list[tuple]]:
    muni_iter = zip(*df_init)
    curr_muni, curr_prov = next(muni_iter)
    muni_map: dict[str, list[tuple]] = {curr_muni: []}

    for muni, prov in muni_iter:
        if muni == "Florenceville-Bristol TV":
            muni_map[curr_muni].extend(
                [
                    (muni, curr_prov),
                    ("Florenceville TV", curr_prov),
                    ("Bristol TV", curr_prov),
                ]
            )
        elif muni == "Fundy Shores":
            curr_muni, curr_prov = muni, prov
            muni_map[curr_muni] = []
        elif curr_muni == "Fundy Shores" and muni != "Fundy-St. Martins":
            muni_map[curr_muni].append((muni, prov))
        elif muni == "Woodstock":
            curr_muni, curr_prov = muni, prov
            muni_map[curr_muni] = []
        elif curr_muni == "Woodstock":
            muni_map[curr_muni].append((muni, prov))
        elif prov is None:
            muni_map[curr_muni].append((muni, curr_prov))
        else:
            curr_muni, curr_prov = muni, prov
            muni_map[curr_muni] = []

    return muni_map


# %%
def write_clean_bgt_exps_data() -> None:
    for file in SRC_DIR.rglob("*_bgt_exps.xlsx"):
        dst = DST_DIR / file.relative_to(SRC_DIR)
        dst.parent.mkdir(parents=True, exist_ok=True)

        df = clean_bgt_exps_data(file)
        df.write_excel(dst, header_format={"bold": True}, autofit=True)


def clean_bgt_exps_data(file: Path) -> pl.DataFrame:
    df_init = pl.read_excel(file, has_header=False)
    skip = next(
        i
        for i, row in enumerate(df_init.iter_rows(), 1)
        if row[1] and re.match(r"(Fredericton|Bathurst)", row[1].title())
    )
    columns = [
        "Index",
        "Municipality",
        "General Government",
        "Police",
        "Fire Protection",
        "Water Cost Transfer",
        "Emergency Measures",
        "Other Protection Services",
        "Transportation",
        "Environmental Health",
        "Public Health",
        "Environmental Development",
        "Recreation & Cultural",
        "Debt Costs",
        "Transfers",
        "Deficits",
        "Total Expenditures",
    ]
    columns_float = ["General Government", "Debt Costs", "Total Expenditures"]

    with BytesIO() as buffer:
        df_init.write_excel(buffer)
        df = pl.read_excel(buffer, read_options={"skip_rows": skip}, has_header=False)

    df = df.select(
        col for col in df if col.drop_nulls().len() >= df.height / 10
    ).select(pl.nth(list(range(17))))

    df = (
        df.rename(dict(zip(df.columns, columns)))
        .filter(~pl.col("Index").str.contains(r"\D"))
        .drop("Index")
        .with_columns(cs.exclude("Municipality", *columns_float).cast(pl.Int64))
        .with_columns(cs.by_name(columns_float).cast(pl.Float64))
        .fill_null(0)
    )

    return clean_munis(df)


# %%
def write_clean_bgt_revs_data() -> None:
    for file in SRC_DIR.rglob("*_bgt_revs.xlsx"):
        dst = DST_DIR / file.relative_to(SRC_DIR)
        dst.parent.mkdir(parents=True, exist_ok=True)

        df = clean_bgt_revs_data(file)
        df.write_excel(dst, header_format={"bold": True}, autofit=True)


def clean_bgt_revs_data(file: Path) -> pl.DataFrame:
    df_init = pl.read_excel(file, has_header=False)
    skip = next(
        i
        for i, row in enumerate(df_init.iter_rows(), 1)
        if row[1] and re.match(r"(Fredericton|Bathurst)", row[1].title())
    )
    columns = [
        "Index",
        "Municipality",
        "Warrant",
        "Unconditional Grant",
        "Services to Other Governments",
        "Sale of Services",
        "Own-Source Revenue",
        "Conditional Transfers",
        "Other Transfers",
        "Biennial Surplus",
        "Total Revenue",
    ]

    with BytesIO() as buffer:
        df_init.write_excel(buffer)
        df = pl.read_excel(buffer, read_options={"skip_rows": skip}, has_header=False)

    df = df.select(
        col for col in df if col.drop_nulls().len() >= df.height / 10
    ).select(pl.nth(list(range(11))))

    df = (
        df.rename(dict(zip(df.columns, columns)))
        .filter(~pl.col("Index").str.contains(r"\D"))
        .drop("Index")
        .with_columns(cs.exclude("Municipality").cast(pl.Int64))
        .fill_null(0)
    )

    return clean_munis(df)


# %%
def write_clean_cmp_data() -> None:
    for file in SRC_DIR.rglob("*_cmp_data.xlsx"):
        dst = DST_DIR / file.relative_to(SRC_DIR)
        dst.parent.mkdir(parents=True, exist_ok=True)

        df = clean_cmp_data(file)
        df.write_excel(dst, header_format={"bold": True}, autofit=True)


def clean_cmp_data(file: Path) -> pl.DataFrame:
    df_init = pl.read_excel(file, has_header=False)
    skip = next(
        i
        for i, row in enumerate(df_init.iter_rows(), 1)
        if row[1] and re.match(r"(Fredericton|Bathurst)", row[1].title())
    )
    columns = [
        "Index",
        "Municipality",
        "Latest Census Population",
        "Penultimate Census Population",
        "Provincial Kilometrage",
        "Regional Kilometrage",
        "Municipal Kilometrage",
        "Total Kilometrage",
        "Population/Kilometrage",
        "Tax Base",
        "Tax Base/Capita",
        "Tax Base/Kilometrage",
        "Total Budget",
        "Fiscal Capacity",
        "Average Tax Rate",
    ]
    columns_float = [
        "Provincial Kilometrage",
        "Regional Kilometrage",
        "Municipal Kilometrage",
        "Total Kilometrage",
        "Population/Kilometrage",
        "Tax Base/Capita",
        "Tax Base/Kilometrage",
        "Fiscal Capacity",
        "Average Tax Rate",
    ]
    columns_int = [
        "Latest Census Population",
        "Penultimate Census Population",
        "Tax Base",
        "Total Budget",
    ]

    with BytesIO() as buffer:
        df_init.write_excel(buffer)
        df = pl.read_excel(buffer, read_options={"skip_rows": skip}, has_header=False)

    df = df.select(
        col for col in df if col.drop_nulls().len() >= df.height / 10
    ).select(pl.nth(list(range(15))))

    df = (
        df.rename(dict(zip(df.columns, columns)))
        .filter(~pl.col("Index").str.contains(r"\D"))
        .drop("Index")
        .with_columns(cs.by_name(columns_float).cast(pl.Float64))
        .with_columns(cs.by_name(columns_int).cast(pl.Int64))
        .fill_null(0)
    )

    return clean_munis(df)


# %%
def write_clean_tax_base_data() -> None:
    for file in SRC_DIR.rglob("*_tax_base.xlsx"):
        dst = DST_DIR / file.relative_to(SRC_DIR)
        dst.parent.mkdir(parents=True, exist_ok=True)

        df = clean_tax_base_data(file)
        df.write_excel(dst, header_format={"bold": True}, autofit=True)


def clean_tax_base_data(file: Path) -> pl.DataFrame:
    df_init = pl.read_excel(file, has_header=False)
    skip = next(
        i
        for i, row in enumerate(df_init.iter_rows(), 1)
        if row[1] and re.match(r"(Fredericton|Bathurst)", row[1].title())
    )
    columns = [
        "Index",
        "Municipality",
        "General Residential Assessment",
        "Federal Residential Assessment",
        "Provincial Residential Assessment",
        "Total Residential Assessment",
        "General Non-Residential Assessment",
        "Federal Non-Residential Assessment",
        "Provincial Non-Residential Assessment",
        "Total Non-Residential Assessment",
        "Total Municipal Assessment Base",
        "Total Municipal Tax Base",
        "Total Tax Base for Rate",
    ]

    with BytesIO() as buffer:
        df_init.write_excel(buffer)
        df = pl.read_excel(buffer, read_options={"skip_rows": skip}, has_header=False)

    df = df.select(
        col for col in df if col.drop_nulls().len() >= df.height / 10
    ).select(pl.nth(list(range(13))))

    df = (
        df.rename(dict(zip(df.columns, columns)))
        .filter(~pl.col("Municipality").str.contains(r"^(GROUP|TOTAL|of|\*)"))
        .with_columns(cs.exclude("Municipality").cast(pl.Float64, strict=False))
        .fill_null(0)
    )

    return clean_munis(_combine_tax_base_districts(df))


def _combine_tax_base_districts(df: pl.DataFrame) -> pl.DataFrame:
    row_iter = df.iter_rows()
    base_idx = 0

    for i, row in enumerate(row_iter):
        if row[0] == 0:
            for j in range(2, df.width):
                df[base_idx, j] += row[j]
        else:
            base_idx = i

    return (
        df.filter(pl.col("Index") != 0)
        .drop("Index")
        .with_columns(cs.exclude("Municipality").cast(pl.Int64))
    )


# %%
if __name__ == "__main__":
    main()
