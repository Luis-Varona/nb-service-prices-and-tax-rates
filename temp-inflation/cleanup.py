# Copyright 2025 Craig Brett and Luis M. B. Varona
#
# Licensed under the MIT license <LICENSE or
# http://opensource.org/licenses/MIT>. This file may not be copied, modified, or
# distributed except according to those terms.


# %%
import csv

from pathlib import Path

import polars as pl


# %%
WD = Path(__file__).parent
SRC = WD / "1810000501-eng.csv"
DST = WD / "cpi_deflator.csv"

PANEL_YEARS = range(2000, 2021)
BASE_YEAR = 2020


# %%
def main() -> None:
    df = build_deflator(SRC, PANEL_YEARS, BASE_YEAR)
    df.write_csv(DST)
    print(df)
    print(f"\nWritten to {DST}")


# %%
def build_deflator(
    src: Path,
    panel_years: range,
    base_year: int,
) -> pl.DataFrame:
    years, cpi_values = parse_allitems_cpi(src)

    df = pl.DataFrame({"Year": years, "CPI": cpi_values}).filter(
        pl.col("Year").is_between(panel_years.start, panel_years.stop - 1)
    )

    base_cpi = df.filter(pl.col("Year") == base_year).select("CPI").item()

    return df.with_columns((pl.lit(base_cpi) / pl.col("CPI")).alias("Deflator"))


def parse_allitems_cpi(src: Path) -> tuple[list[int], list[float]]:
    with open(src, encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        rows = list(reader)

    year_row = next(r for r in rows if r and r[0].startswith("Products"))
    allitems_row = next(r for r in rows if r and r[0].strip() == "All-items")

    years = [int(v) for v in year_row[1:] if v.strip().isdigit()]
    cpi_values = [float(v) for v in allitems_row[1:] if v.strip()]

    if len(years) != len(cpi_values):
        raise ValueError(
            f"Year count ({len(years)}) != CPI value count ({len(cpi_values)})"
        )

    return years, cpi_values


# %%
if __name__ == "__main__":
    main()
