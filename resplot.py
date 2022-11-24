# Neural Abstractions
# Copyright (c) 2022  Alessandro Abate, Alec Edwards, Mirco Giacobbe

# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from utils import interpolate_error

import benchmarks


def sort_results(f1: str = "results/results.csv"):
    df = pd.read_csv(f1)
    sorted_df = df.sort_values(by=["Benchmark", "Method", "Partitions", "Seed"])
    sorted_df.to_csv(f1, index=False)


def get_successes(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[df["Result"] == "S"]


def get_best_results(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[df.groupby(["Benchmark", "Partitions"]).Error_1_norm.idxmin()]


def append_normalisation(df: pd.DataFrame) -> pd.DataFrame:
    mode = []
    for index, row in df.iterrows():
        if "normalise" in row["Benchmark"]:
            df.at[index, "Benchmark"] = row["Benchmark"][:-11]
            mode.append("normalised")
        else:
            mode.append("unnormalised")
    df["Mode"] = mode
    return df


def combine_results(
    f1: str = "hybridisation-results.csv", f2: str = "results.csv"
) -> pd.DataFrame:
    df1 = pd.read_csv(f"results/{f1}")
    df2 = pd.read_csv(f"results/{f2}")
    # df2 = df2[df2["Method"].str.contains("ISFNA", na=False)]
    # df2['Method'] = 'NA'
    # df2 = check_successes(df2)
    # df1 = df1[df1.Benchmark.isin(benchmarks)]
    # df2 = df2[df2.Benchmark.isin(benchmarks)]
    df2 = get_successes(df2)
    # df2 = get_best_results(df2)
    df = pd.concat([df1, df2], ignore_index=True)
    return df


def plot_combined_results(
    f1: str = "hybridisation-results.csv", f2: str = "results.csv"
):
    df = combine_results(f1=f1, f2=f2)
    palette = "rocket_r"

    g = sns.FacetGrid(df, col="Benchmark", col_wrap=3)
    g.map(
        sns.scatterplot,
        "Partitions",
        "Error_1_norm",
        hue=df["Method"],
        style=df["Method"],
        # markers=True,
        palette=palette,
        # linewidth=1.5,
        # markersize=8,
        hue_order=["RA", "ASM", "IFNA"],
    )
    g.add_legend()
    # splot = sns.lmplot(x='Partitions', y='Error_1_norm', data=df, col='Benchmark', col_wrap=3, hue='Method')
    g.set(xlim=[1, 200], ylim=[1e-2, 10], yscale="log", xscale="log")


def plot_hybridisation_results(f: str = "hybridisation-results.csv"):
    df = pd.read_csv(f"results/{f}")
    df = append_normalisation(df)
    splot = sns.relplot(
        x="Partitions",
        y="Error",
        data=df,
        kind="line",
        linewidth=2.5,
        hue="Method",
        col="Benchmark",
        style="Mode",
        col_wrap=3,
        palette="rocket_r",
    )
    splot.set(yscale="log", xscale="log")


def plot_robustness_results(f: str = "robustness-results.csv"):
    df = pd.read_csv(f"results/{f}")
    df = get_successes(df)
    splot = sns.countplot(
        x="Width",
        data=df,
        palette="rocket_r",
    )
    splot.set_title(
        "Number of successful Lokta-Volterra runs (out of 11) for different NN structures; error=0.01."
    )


def check_successes(df):
    for i, row in df.iterrows():
        np = row["Partitions"]
        e = interpolate_error(row["Benchmark"], np)
        if row["Error"] < e:
            df.loc[i, "Result"] = "S"
        else:
            df.loc[i, "Result"] = "F"
    return df  # df.to_csv('results/results-corrected.csv')


def plot_successes(f: str = "results/iter-results/iter-results.csv"):
    df = pd.read_csv(f)
    df = get_successes(df)
    g = sns.FacetGrid(df, col="Benchmark", col_wrap=3)
    g.map(
        sns.countplot, "Width", palette="mako", order=df["Width"].value_counts().index
    )


def plot_results(f: str = "results.csv"):
    df = pd.read_csv(f"results/{f}")
    df = df.loc[df["Benchmark"] != "spring-pendulum-normalised"]
    df = df[~df["Benchmark"].str.contains("normalised", na=False)]
    palette = sns.cubehelix_palette(light=0.75, n_colors=4, rot=-0.5)
    splot = sns.relplot(
        x="Partitions",
        y="Est-Max-se",
        data=df,
        style="Result",
        hue="Error",
        palette=palette,
        col="Benchmark",
        col_wrap=3,
        s=75,
    )
    splot.set(yscale="log", xscale="log")


def is_succ(X):
    c = 0
    for x in X:
        if x == "S":
            c += 1
    return c


def table_2():
    data = pd.read_csv("results/results.csv")
    data = data.loc[data["Result"] == "S"]
    # data = data.replace("exponential", "Exponential")  # Neurips papers has different namings
    data["Error_2_norm"] = pd.to_numeric(data["Error_2_norm"], downcast="float")
    data["Partitions"] = pd.to_numeric(data["Partitions"], downcast="float")
    G = data.groupby(["Benchmark", "Width"])
    cols = [G.Error_2_norm, G.Partitions]
    tab = []
    n_succ = G["Partitions"].count() / 10  # Total 10 experiment repeats
    n_succ.name = "Success Ratio"
    col_df = []
    for col in cols:
        row_mean = col.mean()
        row_mean.name += " mean"
        row_min = col.min()
        row_min.name += " min"
        row_max = col.max()
        row_max.name += " max"
        col_df.append(pd.concat([row_mean, row_max, row_min], axis=1))
    col_df.append(n_succ)
    df = pd.concat(col_df, axis=1)

    df.to_latex("results/table2.tex")

    hyb_data = pd.read_csv("results/hybridisation-results.csv")
    # hyb_data = hyb_data.replace(
    #     "exponential", "Exponential"
    # )  # Neurips papers has different namings
    hyb_data = hyb_data.drop(["Method", "h"], axis=1)
    hyb_data.to_latex("results/table2-asm.tex")


if __name__ == "__main__":
    table_2()
