# Overview of Data

The `main.csv` spreadsheet contains all data from all years, combined, except for per-capita and inflation-adjusted figures (which can be easily derived from the data it does contain).

For convenience, the spreadsheets `cpi_defl_2002.csv`, `pol_prov_2024.csv`, `cmp_demo.csv` `bgt_exps.csv`, and `bgt_revs.csv` are also presented - the first two use strict subsets of the columns in `main.csv`, whereas the last three include columns suffixed with `adj`, `capita`, and `capita_adj` for the per-capita and inflation-adjusted monetary figures. (Refraining from suffixing the monetary columns in `main_csv` with these derived figures is done to avoid an unreadably large combined spreadsheet.)

For further convenience, each of `main.csv`, `cmp_demo.csv`, `bgt_exps.csv`, and `bgt_revs.csv` also has a corresponding `[year]/[name]_[year].csv` sheet (e.g., `2004/main_2004.csv`) containing all data from a given year (without the `year` column, of course).

Finally, `meow.py` is just a script showing how this data was processed into its final form. (We will be rewriting the whole data processing pipeline later anyway as we near the paper-writing stage.)
