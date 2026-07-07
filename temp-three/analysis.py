# Copyright 2025 Craig Brett and Luis M. B. Varona
#
# Licensed under the MIT license <LICENSE or
# http://opensource.org/licenses/MIT>. This file may not be copied, modified, or
# distributed except according to those terms.


# %%
import pickle
from pathlib import Path

import polars as pl
from linearmodels.panel.model import PanelOLS, PooledOLS


# %%
WD = Path(__file__).parent
DATA_DIR = WD.parent / "data" / "data_final"
CPI_SRC = DATA_DIR / "cpi_deflator.csv"
RESULTS_DIR = WD / "results"
TXT_DIR = RESULTS_DIR / "txt"
TEX_DIR = RESULTS_DIR / "tex"
PKL_DIR = RESULTS_DIR / "pkl"


# %%
DEP_VAR = "AvgTaxRate"
ENTITY_COL = "Municipality"
TIME_COL = "Year"
POST_YEAR = 2012

DEFLATE_COLS = ["PolExpCapita", "ControlRevCapita", "UncondGrantCapita"]
DIFF_COLS = ["AvgTaxRate", "PolExpCapita", "ControlRevCapita", "UncondGrantCapita"]

MODES = [
    ("reg_rate", False),
    ("reg_diff", True),
]


# %%
DDD_CORE = [
    "PolExpCapita",
    "PolExpCapita:Provider_PPSA",
    "PolExpCapita:Provider_Municipal",
    "PolExpCapita:Post2012",
    "Provider_PPSA:Post2012",
    "Provider_Municipal:Post2012",
    "PolExpCapita:Provider_PPSA:Post2012",
    "PolExpCapita:Provider_Municipal:Post2012",
]

POOLED_EXTRA = ["Provider_PPSA", "Provider_Municipal", "Post2012"]
FE_EXTRA = ["Post2012"]


# %%
CONTROL_SETS = [
    ("no_ctrl", []),
    ("ctrl_rev", ["ControlRevCapita"]),
    ("full", ["ControlRevCapita", "UncondGrantCapita"]),
]


# %%
SPECS = [
    ("pooled_no_ctrl", "OLS", "None"),
    ("pooled_ctrl_rev", "OLS", "CtrlRev"),
    ("pooled_full", "OLS", "Full"),
    ("fe_no_ctrl", "FE", "None"),
    ("fe_ctrl_rev", "FE", "CtrlRev"),
    ("fe_full", "FE", "Full"),
]

COEF_ORDER = [
    "PolExpCapita",
    "Provider_PPSA",
    "Provider_Municipal",
    "Post2012",
    "PolExpCapita:Provider_PPSA",
    "PolExpCapita:Provider_Municipal",
    "PolExpCapita:Post2012",
    "Provider_PPSA:Post2012",
    "Provider_Municipal:Post2012",
    "PolExpCapita:Provider_PPSA:Post2012",
    "PolExpCapita:Provider_Municipal:Post2012",
    "ControlRevCapita",
    "UncondGrantCapita",
]

COEF_LABELS = {
    "PolExpCapita": "PolExpCapita",
    "Provider_PPSA": "PPSA",
    "Provider_Municipal": "Municipal",
    "Post2012": "Post2012",
    "PolExpCapita:Provider_PPSA": "PolExpCapita * PPSA",
    "PolExpCapita:Provider_Municipal": "PolExpCapita * Municipal",
    "PolExpCapita:Post2012": "PolExpCapita * Post2012",
    "Provider_PPSA:Post2012": "PPSA * Post2012",
    "Provider_Municipal:Post2012": "Municipal * Post2012",
    "PolExpCapita:Provider_PPSA:Post2012": "PolExpCapita * PPSA * Post2012",
    "PolExpCapita:Provider_Municipal:Post2012": "PolExpCapita * Municipal * Post2012",
    "ControlRevCapita": "ControlRevCapita",
    "UncondGrantCapita": "UncondGrantCapita",
}


# %%
def main() -> None:
    TXT_DIR.mkdir(parents=True, exist_ok=True)
    TEX_DIR.mkdir(parents=True, exist_ok=True)
    PKL_DIR.mkdir(parents=True, exist_ok=True)

    for prefix, diff in MODES:
        title = f"YearDiff={diff}"
        df = load_data(diff)
        results = {}

        for ctrl_name, ctrl_vars in CONTROL_SETS:
            key = f"pooled_{ctrl_name}"
            terms = POOLED_EXTRA + DDD_CORE + ctrl_vars
            results[key] = run_pooled(df, terms)
            save_summary(f"{prefix}_{key}", results[key])

            key = f"fe_{ctrl_name}"
            terms = FE_EXTRA + DDD_CORE + ctrl_vars
            results[key] = run_panel(df, terms)
            save_summary(f"{prefix}_{key}", results[key])

        with open(PKL_DIR / f"{prefix}.pkl", "wb") as f:
            pickle.dump(results, f)

        table = build_table(results, title)
        print(f"\n{'=' * 60}")
        print(f"  {prefix}")
        print(f"{'=' * 60}")
        print(table)

        (TEX_DIR / f"{prefix}_table.tex").write_text(table)
        (RESULTS_DIR / f"{prefix}_table_standalone.tex").write_text(
            standalone_wrap(table)
        )


# %%
def save_summary(key: str, result) -> None:
    (TXT_DIR / f"{key}.txt").write_text(result.summary.as_text())
    (TEX_DIR / f"{key}.tex").write_text(
        result.summary.as_latex().replace("\\begin{table}", "\\begin{table}[H]", 1)
    )


# %%
def load_data(diff: bool) -> pl.DataFrame:
    df_master = pl.read_excel(DATA_DIR / "data_master.xlsx").select(
        [
            TIME_COL,
            ENTITY_COL,
            "AvgTaxRate",
            "PolExpCapita",
            "OtherRevCapita",
            "Provider_PPSA",
            "Provider_MPSA",
            "LatestCensusPop",
        ]
    )
    df_revs = pl.read_excel(
        DATA_DIR / "data_bgt_revs.xlsx",
        columns=[TIME_COL, ENTITY_COL, "Unconditional Grant"],
    )
    df_cpi = pl.read_csv(CPI_SRC).with_columns(pl.col(TIME_COL).cast(pl.UInt32))

    df = (
        df_master.join(df_revs, on=[TIME_COL, ENTITY_COL], how="left")
        .with_columns(
            (~pl.col("Provider_PPSA") & ~pl.col("Provider_MPSA"))
            .cast(pl.Int8)
            .alias("Provider_Municipal"),
        )
        .drop("Provider_MPSA")
        .with_columns(
            (pl.col("Unconditional Grant") / (1000 * pl.col("LatestCensusPop"))).alias(
                "UncondGrantCapita"
            ),
        )
        .with_columns(
            (pl.col("OtherRevCapita") - pl.col("UncondGrantCapita")).alias(
                "ControlRevCapita"
            ),
        )
        .join(df_cpi.select(TIME_COL, "Deflator"), on=TIME_COL, how="left")
        .with_columns(pl.col(DEFLATE_COLS) * pl.col("Deflator"))
        .drop("Deflator")
    )

    if diff:
        df = (
            df.sort([ENTITY_COL, TIME_COL])
            .with_columns(
                (pl.col(TIME_COL) - pl.col(TIME_COL).shift(1).over(ENTITY_COL)).alias(
                    "_year_gap"
                ),
                *(
                    (pl.col(c) - pl.col(c).shift(1).over(ENTITY_COL)).alias(c)
                    for c in DIFF_COLS
                ),
            )
            .filter(pl.col("_year_gap") == 1)
            .drop("_year_gap")
        )

    return df.with_columns(
        (pl.col(TIME_COL) >= POST_YEAR).cast(pl.Int8).alias("Post2012"),
        pl.col("Provider_PPSA").cast(pl.Int8),
    ).drop(["Unconditional Grant", "LatestCensusPop", "OtherRevCapita"])


# %%
def run_pooled(df: pl.DataFrame, terms: list[str], dep_var: str = DEP_VAR):
    formula = f"{dep_var} ~ 1 + {' + '.join(terms)}"
    df_pd = df.to_pandas().set_index([ENTITY_COL, TIME_COL])
    model = PooledOLS.from_formula(formula, df_pd)
    return model.fit(cov_type="clustered", cluster_entity=True)


def run_panel(df: pl.DataFrame, terms: list[str], dep_var: str = DEP_VAR):
    formula = f"{dep_var} ~ 1 + {' + '.join(terms)} + EntityEffects"
    df_pd = df.to_pandas().set_index([ENTITY_COL, TIME_COL])
    model = PanelOLS.from_formula(formula, df_pd)
    return model.fit(cov_type="clustered", cluster_entity=True)


# %%
def stars(pval: float) -> str:
    if pval < 0.01:
        return "^{***}"
    elif pval < 0.05:
        return "^{**}"
    elif pval < 0.1:
        return "^{*}"
    return ""


def fmt(val: float) -> str:
    if abs(val) < 0.00005:
        s = f"{val:.3e}"
        i = s.index("e")
        sign = s[i + 1]
        digits = s[i + 2 :]
        return s[:i] + r"\text{e" + sign + "}" + digits
    return f"{val:.4f}"


# %%
def build_table(results: dict, title: str) -> str:
    ncols = len(SPECS)
    t1 = "    "
    t2 = "        "
    lines = []

    lines.append(r"\begin{table}[H]")
    lines.append(t1 + r"\centering")
    lines.append(t1 + r"\scriptsize")
    lines.append(t1 + r"\begin{minipage}{\textwidth}")
    lines.append(t1 + r"\centering")
    lines.append(t1 + r"{\normalsize " + title + "}")
    lines.append(t1 + r"\vspace{6pt}")
    lines.append(t1 + r"\begin{tabular}{l" + "c" * ncols + "}")
    lines.append(t2 + r"\toprule")

    nums = " & ".join(f"({i + 1})" for i in range(ncols))
    lines.append(t2 + f" & {nums} \\\\")

    ests = " & ".join(spec[1] for spec in SPECS)
    lines.append(t2 + f" & {ests} \\\\")
    lines.append(t2 + r"\midrule")

    for coef_name in COEF_ORDER:
        label = COEF_LABELS[coef_name]
        cells = []
        se_cells = []

        for spec_key, _, _ in SPECS:
            result = results[spec_key]

            if coef_name in result.params.index:
                coef = result.params[coef_name]
                se = result.std_errors[coef_name]
                pval = result.pvalues[coef_name]
                cells.append(f"${fmt(coef)}{stars(pval)}$")
                se_cells.append(f"$({fmt(se)})$")
            else:
                cells.append("")
                se_cells.append("")

        lines.append(t2 + f"{label} & {' & '.join(cells)} \\\\")
        lines.append(t2 + f" & {' & '.join(se_cells)} \\\\[4pt]")

    lines.append(t2 + r"\midrule")

    ctrl_cells = " & ".join(spec[2] for spec in SPECS)
    lines.append(t2 + f"Controls & {ctrl_cells} \\\\")

    efe = " & ".join("Yes" if est == "FE" else "No" for _, est, _ in SPECS)
    lines.append(t2 + f"Entity FE & {efe} \\\\")

    ns = " & ".join(str(int(results[k].nobs)) for k, _, _ in SPECS)
    lines.append(t2 + f"$N$ & {ns} \\\\")

    r2s = " & ".join(f"${results[k].rsquared:.4f}$" for k, _, _ in SPECS)
    lines.append(t2 + f"$R^2$ & {r2s} \\\\")

    lines.append(t2 + r"\bottomrule")
    lines.append(t1 + r"\end{tabular}")
    lines.append("")
    lines.append(t1 + r"\vspace{6pt}")
    lines.append(
        t1 + r"{\scriptsize Standard errors clustered by municipality in parentheses. "
        r"$^{***}p<0.01$, $^{**}p<0.05$, $^{*}p<0.1$.}"
    )
    lines.append("")
    lines.append(
        t1 + r"{\scriptsize ControlRevCapita = "
        r"(Total Revenue $-$ Warrant $-$ Unconditional Grant) / capita.}"
    )
    lines.append(t1 + r"\end{minipage}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def standalone_wrap(table: str) -> str:
    return (
        "\\documentclass[border=10pt,varwidth=25cm]{standalone}\n"
        "\\usepackage{amsmath}\n"
        "\\usepackage{booktabs}\n"
        "\\renewenvironment{table}[1][]{}{}\n"
        "\\begin{document}\n"
        f"{table}\n"
        "\\end{document}\n"
    )


# %%
if __name__ == "__main__":
    main()
