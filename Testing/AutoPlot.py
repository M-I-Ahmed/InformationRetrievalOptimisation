# This script will take the results from the recall evaluation and generate plots to visualise the performance of the different models across different values of k.

import argparse
import csv
import matplotlib.pyplot as plt

def load_results(file_path):
    results = {}
    with open(file_path, "r", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        for row in reader:
            model_name = row[0]
            model_results = {int(k): float(v) for k, v in zip(header[1:], row[1:])}
            results[model_name] = model_results
    return results

def plot_results(results, ks, title, ylabel):
    plt.figure(figsize=(10, 6))
    markers = ["o", "s", "D", "^", "v", "<", ">", "P", "X", "*", "h", "H", "d", "p", "8"]
    colors = plt.cm.tab20.colors
    for idx, (model_name, model_results) in enumerate(results.items()):
        recalls = [model_results.get(k, 0.0) for k in ks]
        marker = markers[idx % len(markers)]
        color = colors[idx % len(colors)]
        plt.plot(ks, recalls, marker=marker, color=color, label=model_name)
    plt.xlabel("k")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(ks)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot recall results from CSV file")
    parser.add_argument("--input", type=str, required=True, help="Path to the input CSV file with recall results")
    parser.add_argument(
        "--precision-input",
        type=str,
        default=None,
        help="Optional CSV file with precision@k results.",
    )
    args = parser.parse_args()

    recall_results = load_results(args.input)
    recall_ks = sorted({k for model_results in recall_results.values() for k in model_results.keys()})
    plot_results(recall_results, recall_ks, "Recall@k for Different Models", "Recall@k")

    if args.precision_input:
        precision_results = load_results(args.precision_input)
        precision_ks = sorted({k for model_results in precision_results.values() for k in model_results.keys()})
        plot_results(precision_results, precision_ks, "Precision@k for Different Models", "Precision@k")


    