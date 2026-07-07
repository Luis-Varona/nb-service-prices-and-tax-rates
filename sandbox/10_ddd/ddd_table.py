# Copyright 2025 Craig Brett and Luis M. B. Varona
#
# Licensed under the MIT license <LICENSE or
# http://opensource.org/licenses/MIT>. This file may not be copied, modified, or
# distributed except according to those terms.


# %%
import pickle
from pathlib import Path

# %%
WD = Path(__file__).parent
PKL_DIR = WD / "pkl"
TEX_DIR = WD / "tex"


# %%
SPECS = [
    ("pooled_no_ctrl", "OLS", "None"),
    ("pooled_exp", "OLS", "OtherExp"),
    ("pooled_exp_rev", "OLS", "+OtherRev"),
    ("pooled_full", "OLS", "Full"),
    ("fe_no_ctrl", "FE", "None"),
    ("fe_exp", "FE", "OtherExp"),
    ("fe_exp_rev", "FE", "+OtherRev"),
    ("fe_full", "FE", "Full"),
]


# %%
COEF_ORDER = [
    "PolExpCapita",
    "Provider_PPSA",
    "Post2012",
    "PolExpCapita:Provider_PPSA",
    "PolExpCapita:Post2012",
    "Provider_PPSA:Post2012",
    "PolExpCapita:Provider_PPSA:Post2012",
    "OtherExpCapita",
    "OtherRevCapita",
    "UnconditionalGrantCapita",
]

COEF_LABELS = {
    "PolExpCapita": "PolExpCapita",
    "Provider_PPSA": "PPSA",
    "Post2012": "Post2012",
    "PolExpCapita:Provider_PPSA": "PolExpCapita * PPSA",
    "PolExpCapita:Post2012": "PolExpCapita * Post2012",
    "Provider_PPSA:Post2012": "PPSA * Post2012",
    "PolExpCapita:Provider_PPSA:Post2012": "PolExpCapita * PPSA * Post2012",
    "OtherExpCapita": "OtherExpCapita",
    "OtherRevCapita": "OtherRevCapita",
    "UnconditionalGrantCapita": "UncondGrantCapita",
}


# %%
def main() -> None:
    TEX_DIR.mkdir(parents=True, exist_ok=True)

    with open(PKL_DIR / "results.pkl", "rb") as f:
        results = pickle.load(f)

    table = build_table(results)
    print(table)

    with open(TEX_DIR / "ddd_table.tex", "w") as f:
        f.write(table)

    standalone = (
        "\\documentclass[border=10pt,varwidth=25cm]{standalone}\n"
        "\\usepackage{amsmath}\n"
        "\\usepackage{booktabs}\n"
        "\\renewenvironment{table}[1][]{}{}\n"
        "\\begin{document}\n"
        f"{table}\n"
        "\\end{document}\n"
    )

    with open(WD / "ddd_table_standalone.tex", "w") as f:
        f.write(standalone)


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
def build_table(results: dict) -> str:
    ncols = len(SPECS)
    t1 = "    "
    t2 = "        "
    lines = []

    lines.append(r"\begin{table}[H]")
    lines.append(t1 + r"\centering")
    lines.append(t1 + r"\scriptsize")
    lines.append(t1 + r"\begin{minipage}{\textwidth}")
    lines.append(t1 + r"\centering")
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
    lines.append(t1 + r"\end{minipage}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


# %%
if __name__ == "__main__":
    main()
