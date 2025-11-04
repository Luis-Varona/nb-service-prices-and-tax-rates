# Copyright 2025 Craig Brett and Luis M. B. Varona
#
# Licensed under the MIT license <LICENSE or
# http://opensource.org/licenses/MIT>. This file may not be copied, modified, or
# distributed except according to those terms.


# %%
from io import BytesIO
from pathlib import Path

import polars as pl

from openpyxl import load_workbook


# %%
WD = Path(__file__).parent
DATA_DIR = WD.parent.parent / "data"
SRC_DIR = DATA_DIR / "data_final"
SRC = SRC_DIR / "data_master.xlsx"
DST_DIR = DATA_DIR
DST = DST_DIR / "inconsistent_munis.xlsx"


# %%
def main() -> None:
    df = pl.read_excel(SRC).select(["Year", "Municipality"])
    inconsistent_munis = (
        df.group_by("Municipality")
        .agg(pl.col("Year").n_unique().alias("YearCount"))
        .filter(pl.col("YearCount") < df.select("Year").n_unique())
        .select("Municipality")
        .to_series()
        .to_list()
    )

    year_min = df.select(pl.col("Year").min()).item()
    year_max = df.select(pl.col("Year").max()).item()
    years_all = set(range(year_min, year_max + 1))

    df_inconsistent = (
        df.filter(pl.col("Municipality").is_in(inconsistent_munis))
        .group_by("Municipality")
        .agg([pl.col("Year")])
        .with_columns(
            pl.col("Year")
            .map_elements(
                lambda years: sorted(years_all - set(years)),
                pl.List(pl.Int64),
            )
            .alias("MissingYears")
        )
        .select(["Municipality", "MissingYears"])
        .sort("Municipality")
    )

    with BytesIO() as buffer:
        df_inconsistent.write_excel(buffer)
        wb = load_workbook(buffer)

    ws = wb.active

    for column_cells in ws.columns:
        max_length = max(
            len(str(cell.value)) if cell.value is not None else 0
            for cell in column_cells
        )
        ws.column_dimensions[column_cells[0].column_letter].width = max_length

    wb.save(DST)


# %%
if __name__ == "__main__":
    main()
