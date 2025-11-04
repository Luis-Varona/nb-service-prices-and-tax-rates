# Copyright 2025 Craig Brett and Luis M. B. Varona
#
# Licensed under the MIT license <LICENSE or
# http://opensource.org/licenses/MIT>. This file may not be copied, modified, or
# distributed except according to those terms.


# %%
from pathlib import Path

import polars as pl

from linearmodels.panel.model import PooledOLS


# %%
WD = Path(__file__).parent
DATA_DIR = WD.parent.parent / "data"
SRC = DATA_DIR / "data_final" / "data_master.xlsx"


# %%
DEP_VAR = "AvgTaxRate"
INDEP_VARS = ["PolExpCapita", "Provider_PPSA"]
TIME_VAR = "Year"


# %%
def main() -> None:
    formula = f"{DEP_VAR} ~ 1 + {' + '.join(INDEP_VARS)}"
    columns = (
        [TIME_VAR]
        + [DEP_VAR]
        + list(dict.fromkeys(var for item in INDEP_VARS for var in item.split(":")))
    )

    df = pl.read_excel(SRC).select(columns).to_pandas()
    df["entity"] = 1
    df = df.set_index(["entity", TIME_VAR])

    model = PooledOLS.from_formula(formula, df)
    result = model.fit()
    print(result.summary)


# %%
if __name__ == "__main__":
    main()
