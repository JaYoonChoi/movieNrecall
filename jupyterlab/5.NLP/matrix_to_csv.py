import json
import csv
import os, glob
from tqdm import tqdm
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

files = sorted(glob.glob("matrices/*.json"))
csv_dir = "csvs"
xlsx_dir = "xlsx"
plots_dir = "plots"
rank_plots_dir = "rank_plots"

os.makedirs(csv_dir, exist_ok=True)
os.makedirs(xlsx_dir, exist_ok=True)
os.makedirs(plots_dir, exist_ok=True)
os.makedirs(rank_plots_dir, exist_ok=True)

for file in tqdm(files):
    with open(file, "r") as f:
        data = json.load(f)
        filename = os.path.basename(file).split(".")[0]

    for program in ["RunningMan"]:
        dfs = {}
        for recall in ["Recall1", "Recall2"]:
            header = [""] + data[program][recall]

            # Write to CSV
            output_path = f"{csv_dir}/{filename}-{program}-{recall}.csv"
            with open(output_path, "w") as f:
                writer = csv.writer(f)
                writer.writerow(header)

                for refer, numbers in zip(
                    data[program]["references"], data[program][recall + "-Matrix"]
                ):
                    numbers = [round(x, 4) for x in numbers]
                    writer.writerow([refer] + numbers)
            dfs[recall] = pd.read_csv(output_path)

            # Save data heatmap plot
            matrix = dfs[recall].iloc[:, 1:].to_numpy()
            rank = matrix.argsort(axis=0).argsort(axis=0)
            rank[rank > 3] = 3
            rank = (3 - rank) / 3

            fig = plt.figure(figsize=(10, 10))
            sns.heatmap(
                matrix,
                cmap="YlGnBu",
                xticklabels=False,
                yticklabels=False,
            )
            plt.title(f"{filename}-{program}-{recall}")
            plt.xlabel("Recall Sentences", fontsize=16)
            plt.ylabel("Reference Annotation Sentences", fontsize=16)
            plt.savefig(f"{plots_dir}/{filename}-{program}-{recall}.png")
            plt.close(fig)

            fig = plt.figure(figsize=(10, 10))
            sns.heatmap(
                rank,
                cmap="YlGnBu",
                xticklabels=False,
                yticklabels=False,
            )
            plt.title(f"{filename}-{program}-{recall}-rank")
            plt.xlabel("Recall Sentences", fontsize=16)
            plt.ylabel("Reference Annotation Sentences", fontsize=16)
            plt.savefig(f"{rank_plots_dir}/{filename}-{program}-{recall}-rank.png")
            plt.close(fig)

        # Write to XLSX
        with pd.ExcelWriter(f"{xlsx_dir}/{filename}.xlsx") as writer:
            for sheet, df in dfs.items():
                df.to_excel(writer, sheet_name=sheet, index=False)
