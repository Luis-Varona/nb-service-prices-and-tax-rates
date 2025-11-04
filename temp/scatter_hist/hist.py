# Copyright 2025 Craig Brett and Luis M. B. Varona
#
# Licensed under the MIT license <LICENSE or
# http://opensource.org/licenses/MIT>. This file may not be copied, modified, or
# distributed except according to those terms.


# %%
from pathlib import Path

import polars as pl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from statsmodels.regression.linear_model import OLS, RegressionResultsWrapper


# %%
WD = Path(__file__).parent
DATA_DIR = WD.parent.parent / "data"
DST_DIR = WD / "hist"
SRC = DATA_DIR / "data_final" / "data_master.xlsx"


# %%
DEP_VARS = {
    "PolExpCapita": "pol_exp",
    "AvgTaxRate": "avg_tax",
    "TaxBaseCapita": "tax_base",
    "OtherExpCapita": "other_exp",
}
TIME_VAR = "Year"
CUTOFF = 2011
INDIC_POST = f"Post{CUTOFF}"
INDEP_VARS = [
    TIME_VAR,
    INDIC_POST,
    f"{TIME_VAR}:{INDIC_POST}",
]
ENTITY_COL = "Municipality"
DIVIDE_COL = {
    "Provider_PPSA": ["PPSA", "Non-PPSA"],
}


# %%
def main() -> None:
    DST_DIR.mkdir(parents=True, exist_ok=True)

    columns = (
        list(DEP_VARS.keys())
        + list(dict.fromkeys(var for item in INDEP_VARS for var in item.split(":")))
        + [ENTITY_COL]
        + list(DIVIDE_COL.keys())
    )
    df = (
        pl.read_excel(SRC)
        .with_columns((pl.col(TIME_VAR) > CUTOFF).alias(INDIC_POST))
        .select(columns)
    )

    transition_munis = get_transition_munis(df)
    df = df.filter(pl.col(ENTITY_COL).is_in(transition_munis)).with_columns(
        pl.col(TIME_VAR) - pl.col(TIME_VAR).min()
    )

    divide_col_name = list(DIVIDE_COL.keys())[0]
    df_ppsa = df.filter(pl.col(divide_col_name))
    df_non_ppsa = df.filter(~pl.col(divide_col_name))
    ppsa_munis = df_ppsa.select(ENTITY_COL).to_series().unique()
    non_ppsa_munis = df_non_ppsa.select(ENTITY_COL).to_series().unique()

    for dep_var, short_name in DEP_VARS.items():
        results_ppsa = {
            muni: get_entity_regression(dep_var, muni, df_ppsa) for muni in ppsa_munis
        }
        results_non_ppsa = {
            muni: get_entity_regression(dep_var, muni, df_non_ppsa)
            for muni in non_ppsa_munis
        }

        indicator_coeffs_ppsa = np.array(
            [results_ppsa[muni].params[f"{INDIC_POST}[T.True]"] for muni in ppsa_munis]
        )
        interaction_coeffs_ppsa = np.array(
            [
                results_ppsa[muni].params[f"{TIME_VAR}:{INDIC_POST}[T.True]"]
                for muni in ppsa_munis
            ]
        )
        indicator_coeffs_non_ppsa = np.array(
            [
                results_non_ppsa[muni].params[f"{INDIC_POST}[T.True]"]
                for muni in non_ppsa_munis
            ]
        )
        interaction_coeffs_non_ppsa = np.array(
            [
                results_non_ppsa[muni].params[f"{TIME_VAR}:{INDIC_POST}[T.True]"]
                for muni in non_ppsa_munis
            ]
        )

        df_plot_ppsa = pl.DataFrame(
            {
                "interaction": interaction_coeffs_ppsa,
                "indicator": indicator_coeffs_ppsa,
                "Policing Provider": DIVIDE_COL[divide_col_name][0],
            }
        )
        df_plot_non_ppsa = pl.DataFrame(
            {
                "interaction": interaction_coeffs_non_ppsa,
                "indicator": indicator_coeffs_non_ppsa,
                "Policing Provider": DIVIDE_COL[divide_col_name][1],
            }
        )
        df_plot = pl.concat([df_plot_ppsa, df_plot_non_ppsa])

        for param, name in {
            "indicator": INDIC_POST,
            "interaction": f"{TIME_VAR}:{INDIC_POST}",
        }.items():
            plot = sns.histplot(
                data=df_plot,
                x=param,
                hue="Policing Provider",
                element="step",
                stat="probability",
                common_norm=False,
            )
            plot.set_title(f"Distribution of {name} Coefficients — {dep_var}")
            plot.set_xlabel(f"{name} Coefficient")
            plt.axvline(0, color="grey", linestyle="--", linewidth=1)
            plt.tight_layout()
            plt.savefig(DST_DIR / f"hist_{short_name}_{param}.png", dpi=300)
            plt.clf()

        if dep_var == "TaxBaseCapita":
            df_plot = df_plot.filter(pl.col("indicator") > -1000).filter(
                pl.col("interaction") < 100
            )

            for param, name in {
                "indicator": INDIC_POST,
                "interaction": f"{TIME_VAR}:{INDIC_POST}",
            }.items():
                plot = sns.histplot(
                    data=df_plot,
                    x=param,
                    hue="Policing Provider",
                    element="step",
                    stat="probability",
                    common_norm=False,
                )
                plot.set_title(f"Distribution of {name} Coefficients — {dep_var}")
                plot.set_xlabel(f"{name} Coefficient (No Outliers)")
                plt.axvline(0, color="grey", linestyle="--", linewidth=1)
                plt.tight_layout()
                plt.savefig(
                    DST_DIR / f"hist_{short_name}_{param}_no_outliers.png", dpi=300
                )
                plt.clf()


# %%
def get_transition_munis(df: pl.DataFrame) -> set[str]:
    munis_pre = set(
        df.filter(pl.col(TIME_VAR) <= CUTOFF).select(ENTITY_COL).to_series().unique()
    )
    munis_post = set(
        df.filter(pl.col(TIME_VAR) > CUTOFF).select(ENTITY_COL).to_series().unique()
    )

    return munis_pre.intersection(munis_post)


# %%
def get_entity_regression(
    dep_var: str, municipality: str, df: pl.DataFrame
) -> RegressionResultsWrapper:
    formula = f"{dep_var} ~ 1 + {' + '.join(INDEP_VARS)}"
    df_entity = df.filter(pl.col(ENTITY_COL) == municipality).to_pandas()
    model = OLS.from_formula(formula, df_entity)

    return model.fit()


# %%
if __name__ == "__main__":
    main()
