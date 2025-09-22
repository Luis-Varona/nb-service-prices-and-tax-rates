# Copyright 2025 Craig Brett and Luis M. B. Varona
#
# Licensed under the MIT license <LICENSE or
# http://opensource.org/licenses/MIT>. This file may not be copied, modified, or
# distributed except according to those terms.


# %%
from io import BytesIO
from pathlib import Path
from shutil import copy2
from sys import path

import pandas as pd
import polars as pl


# %%
WD = Path(__file__).parent
path.append(str(WD.parent))

from utils import suppress_fastexcel_logging  # noqa: E402


# %%
DATA_DIR = WD.parent.parent / "data"
SRC_DIR = DATA_DIR / "data_raw"
DST_DIR = DATA_DIR / "data_xlsx"


# %%
@suppress_fastexcel_logging
def main() -> None:
    for suffix in (".xlsx", ".xls", ".xlw"):
        for file in SRC_DIR.rglob(f"*{suffix}"):
            cp_excel_as_xlsx(file, SRC_DIR, DST_DIR)


# %%
def cp_excel_as_xlsx(file: Path, src_dir: Path, dst_dir: Path) -> None:
    suffix = file.suffix.lower()
    dst = dst_dir / file.relative_to(src_dir).with_suffix(".xlsx")

    if suffix == ".xlsx":
        dst.parent.mkdir(parents=True, exist_ok=True)
        copy2(file, dst)
    elif suffix == ".xls":
        dst.parent.mkdir(parents=True, exist_ok=True)
        pl.read_excel(file, has_header=False).write_excel(
            dst, include_header=False, autofit=True
        )
    elif suffix == ".xlw":
        dst.parent.mkdir(parents=True, exist_ok=True)

        with BytesIO() as buffer:
            pd.read_excel(file, engine="xlrd").to_excel(
                buffer, header=False, index=False
            )
            pl.read_excel(buffer, has_header=False).write_excel(
                dst, include_header=False, autofit=True
            )
    else:
        raise ValueError(f"File `{file}` is not an Excel file.")


# %%
if __name__ == "__main__":
    main()
