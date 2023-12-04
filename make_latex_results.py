import os

import pandas as pd


def df_to_latex_table(df):
    table = "\\begin{table}[h!]\n"
    table += "\\begin{tabular}{|l|l|l|l|l|l|l|l|l|l|l|l|}\\hline\n"
    table += " & ".join([str(i) for i in df.columns]) + "\\\\\\hline\n"
    # add multirow for each 4 rows
    for i, row in enumerate(df.itertuples()):
        if i % 4 == 0:
            table += "\\multirow{4}{*}{\\rotatebox[origin=c]{90}{\\textbf{Model}}}"
        table += " & ".join([str(round(x, 2)) for x in row[1:]]) + "\\\\\\hline\n"
    table += "\\end{tabular}\n"
    table += "\\end{table}\n"
    return table

if __name__ == "__main__":
    dfs = {}
    for filename in os.listdir("out/eval/cross-validation"):
        if not filename.endswith(".csv"):
            continue
        df = pd.read_csv(f"out/eval/cross-validation/{filename}")
        dfs[filename.removesuffix(".csv")] = df
    tables = {}
    for metric in ["precision", "recall", "f1"]:
        df = pd.DataFrame()
        for name, df_ in dfs.items():
            df[name] = df_[metric]
        tables[metric] = df.transpose()
    for metric, df in tables.items():
        print(metric)
        print(df_to_latex_table(df))
    