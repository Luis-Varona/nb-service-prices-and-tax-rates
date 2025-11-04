# Copyright 2025 Craig Brett and Luis M. B. Varona
#
# Licensed under the MIT license <LICENSE or
# http://opensource.org/licenses/MIT>. This file may not be copied, modified, or
# distributed except according to those terms.


# %%
from io import StringIO
from pathlib import Path

import polars as pl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn.cluster import KMeans
from statsmodels.regression.linear_model import OLS, RegressionResultsWrapper


# %%
WD = Path(__file__).parent
DATA_DIR = WD.parent.parent / "data"
SRC = DATA_DIR / "data_final" / "data_master.xlsx"
DST = WD / "clustering.txt"


# %%
DEP_VAR = "PolExpCapita"
TIME_VAR = "Year"
# CUTOFF = 2011
CUTOFF = 11
INDIC_POST = f"Post{CUTOFF}"
INDEP_VARS = [
    TIME_VAR,
    INDIC_POST,
    f"{TIME_VAR}:{INDIC_POST}",
]
ENTITY_COL = "Municipality"
FILTER_COL = "Provider_PPSA"


# %%
N_CLUSTERS = 2
N_INIT = 10
RANDOM_STATE = 87


# %%
def main() -> None:
    columns = (
        [DEP_VAR]
        + list(dict.fromkeys(var for item in INDEP_VARS for var in item.split(":")))
        + [ENTITY_COL]
    )
    df = (
        pl.read_excel(SRC)
        .filter(pl.col(FILTER_COL))
        .with_columns(pl.col(TIME_VAR) - pl.col(TIME_VAR).min())
        .with_columns((pl.col(TIME_VAR) > CUTOFF).alias(INDIC_POST))
        .select(columns)
    )
    transition_munis = get_transition_munis(df)
    results = {muni: get_entity_regression(muni, df) for muni in transition_munis}

    indicator_coeffs = np.array(
        [results[muni].params[f"{INDIC_POST}[T.True]"] for muni in transition_munis]
    )
    interaction_coeffs = np.array(
        [
            results[muni].params[f"{TIME_VAR}:{INDIC_POST}[T.True]"]
            for muni in transition_munis
        ]
    ).reshape(-1, 1)

    sns.scatterplot(
        x=interaction_coeffs.flatten(),
        y=indicator_coeffs.flatten(),
    )
    plt.show()
    kmeans = KMeans(N_CLUSTERS, n_init=N_INIT, random_state=RANDOM_STATE)
    cluster_labels = kmeans.fit_predict(interaction_coeffs)

    muni_clusters = dict(zip(transition_munis, cluster_labels))
    sorted_munis = sorted(
        transition_munis,
        key=lambda muni: (
            muni_clusters[muni],
            results[muni].params[f"{TIME_VAR}:{INDIC_POST}[T.True]"],
        ),
    )

    out = StringIO()
    muni_width = max(len(m) for m in transition_munis)

    for cluster in [0, 1]:
        out.write(f"\n=== CLUSTER {cluster} ===\n")
        cluster_munis = [m for m in sorted_munis if muni_clusters[m] == cluster]

        for muni in cluster_munis:
            result = results[muni]
            post = result.params[f"{INDIC_POST}[T.True]"]
            inter = result.params[f"{TIME_VAR}:{INDIC_POST}[T.True]"]

            out.write(f"- {muni}:{' ' * (muni_width - len(muni) + 2)}")
            out.write(f"{INDIC_POST} = {post:8.3f},  ")
            out.write(f"{TIME_VAR}:{INDIC_POST} = {inter:8.5f}\n")

        cluster_coeffs = [
            results[m].params[f"{TIME_VAR}:{INDIC_POST}[T.True]"] for m in cluster_munis
        ]
        out.write(f"  Cluster mean: {np.mean(cluster_coeffs):.5f}\n")
        out.write(f"  Cluster size: {len(cluster_munis)}\n")

    DST.write_text(out.getvalue())


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
    municipality: str, df: pl.DataFrame
) -> RegressionResultsWrapper:
    formula = f"{DEP_VAR} ~ 1 + {' + '.join(INDEP_VARS)}"
    df_entity = df.filter(pl.col(ENTITY_COL) == municipality).to_pandas()
    model = OLS.from_formula(formula, df_entity)

    return model.fit()


# %%
if __name__ == "__main__":
    main()
