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
SRC = WD.parent.parent / "data" / "data_final" / "data_master.xlsx"
TXT_DIR = WD / "txt"


# %%
DEP_VAR = "AvgTaxRate"
TIME_VAR = "Year"
INDIC_2011 = "Post2011"

MODELS = {
    "basic_ols_naive": [
        "PolExpCapita",
        "Provider_PPSA",
    ],
    "basic_ols_post2011": [
        "PolExpCapita",
        "Provider_PPSA",
        INDIC_2011,
        f"{INDIC_2011}:Provider_PPSA",
    ],
    "basic_ols_full": [
        "PolExpCapita",
        "Provider_PPSA",
        INDIC_2011,
        "PolExpCapita:Provider_PPSA",
        f"{INDIC_2011}:PolExpCapita",
        f"{INDIC_2011}:Provider_PPSA",
        f"{INDIC_2011}:PolExpCapita:Provider_PPSA",
    ],
}


# %%
def main() -> None:
    TXT_DIR.mkdir(parents=True, exist_ok=True)

    for model_name, indep_vars in MODELS.items():
        run_ols(indep_vars, TXT_DIR / f"{model_name}.txt")


# %%
def run_ols(indep_vars: list[str], dst: Path) -> None:
    formula = f"{DEP_VAR} ~ 1 + {' + '.join(indep_vars)}"
    columns = (
        [TIME_VAR]
        + [DEP_VAR]
        + list(dict.fromkeys(var for item in indep_vars for var in item.split(":")))
    )

    df = (
        pl.read_excel(SRC)
        .with_columns((pl.col(TIME_VAR) > 2011).alias(INDIC_2011))
        .select(columns)
        .to_pandas()
    )
    df["entity"] = 1
    df = df.set_index(["entity", TIME_VAR])

    model = PooledOLS.from_formula(formula, df)
    result = model.fit()

    dst.write_text(str(result.summary))


# %%
if __name__ == "__main__":
    main()
