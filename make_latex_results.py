import os

import pandas as pd

def circular_iterator(iterable):
    while True:
        for x in iterable:
            yield x

def df_to_latex_table(df, models, vectorizers, selections, caption="", label=""):
    column_config = "|l|l|l|c|c|c|c|c|c|c|c|c|"
    table = "\\begin{table}[h!]\n"
    if caption:
        table += f"\\caption{{{caption}}}\n"
    table += f"\\begin{{tabular}}{{{column_config}}}\\hline\n"
    table += " & ".join([""] *3  + [str(i) for i in df.columns]) + "\\\\\\hline\n"
    m_gen = circular_iterator(models)
    v_gen = circular_iterator(vectorizers)
    s_gen = circular_iterator(selections)
    for i, row in enumerate(df.itertuples()):
        multirow = []
        if i % 4 == 0:
            multirow.append(f"\\multirow{{4}}{{*}}{{\\textbf{{{next(m_gen)}}}}}")
        else:
            multirow.append("")
        if i % 2 == 0:
            multirow.append(f"\\multirow{{2}}{{*}}{{\\textbf{{{next(v_gen)}}}}}")
        else:
            multirow.append("")
        multirow.append(f"\\multirow{{1}}{{*}}{{\\textbf{{{next(s_gen)}}}}}")
        table += (
            " & ".join(multirow + [str(round(x, 2)) for x in row[1:]])
            + "\\\\"
            + ("\\hline\n" if i % 4 == 3 else "\\cline{2-12}\n" if i % 2 == 1 else "\\cline{3-12}\n")
        )
    table += "\\end{tabular}\n"
    if label:
        table += f"\\label{{{label}}}\n"
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
    print(tables)
    models =[]
    vectorizers = []
    selections = []
    for name in dfs.keys():
        model, vectorizer, selection = name.split("_")
        models.append(model)
        vectorizers.append(vectorizer)
        selections.append(selection)
    for metric, df in tables.items():
        print(metric)
        print(df_to_latex_table(df, [
            "\\gls{cnn}",
            "Decision Tree",
            "Gaussian Naive Bayes",
            "\\gls{lstm}",
            "Dense Neural Network",
            "\\gls{svm}",
            ], ["Count", "\\gls{tfidf}"], ["chi2", "\\gls{anova}"]))
