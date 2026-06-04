# visualize_pareto.py

import json
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import mplcursors

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def identify_pareto_front(trials):
    """
    Given trials: list of dicts with 'values' = [f1, time].
    Returns a list of (time, f1) tuples that belong to the Pareto front.
    """
    pareto_front = []
    for i, trial_i in enumerate(trials):
        f1_i, time_i = trial_i["values"]
        dominated = False
        for j, trial_j in enumerate(trials):
            if i == j:
                continue
            f1_j, time_j = trial_j["values"]
            # trial_j dominates trial_i if f1_j >= f1_i and time_j <= time_i
            # with at least one strict inequality
            if (f1_j >= f1_i and time_j <= time_i) and (f1_j > f1_i or time_j < time_i):
                dominated = True
                break
        if not dominated:
            pareto_front.append((time_i, f1_i))
    return pareto_front

def visualize_pareto(json_file):
    """
    Visualises the Pareto front from a JSON file containing
    trials: [ {number, values:[f1, time], params, best_threshold, state}, ... ]
    """
    # Load data
    data = load_json(json_file)

    # Extract F1/time for all trials and build tooltip labels
    all_time = []
    all_f1 = []
    labels = []  # Tooltip text for each point

    for trial in data:
        f1_val, time_val = trial["values"]
        all_time.append(time_val)
        all_f1.append(f1_val)

        # Build tooltip label with key parameters
        trial_number = trial["number"]
        params = trial.get("params", {})
        aggregator = params.get("aggregator", "NA")
        hidden_dim = params.get("hidden_dim", "NA")
        dropout = params.get("dropout_rate", "NA")
        lr = params.get("lr", "NA")

        label_str = (
            f"Trial #{trial_number}\n"
            f"F1 = {f1_val:.3f}, Time = {time_val:.2f}s\n"
            f"Aggregator = {aggregator}\n"
            f"Hidden Dim = {hidden_dim}\n"
            f"Dropout = {dropout:.3f}\n"
            f"LR = {lr:.5f}"
        )
        labels.append(label_str)

    # Compute Pareto front
    pareto_front = identify_pareto_front(data)
    if pareto_front:
        pareto_front_sorted = sorted(pareto_front, key=lambda x: x[0])
        pareto_time, pareto_f1 = zip(*pareto_front_sorted)
    else:
        pareto_time, pareto_f1 = [], []

    # Configure seaborn style
    sns.set_theme(style="whitegrid", context="talk", font_scale=1.2)
    plt.figure(figsize=(12,8))

    # Plot all trials
    scatter_all = plt.scatter(
        all_time, all_f1,
        c='blue', alpha=0.6, s=120, edgecolors='w', label='Trials'
    )

    # Plot Pareto front
    if pareto_front:
        scatter_pareto = plt.scatter(
            pareto_time, pareto_f1,
            c='red', marker='*', s=300, edgecolors='k', label='Pareto Front'
        )
        plt.plot(
            pareto_time, pareto_f1,
            'r--', linewidth=2
        )

    # Interactive hover tooltips for all trials
    cursor = mplcursors.cursor(scatter_all, hover=True)

    @cursor.connect("add")
    def on_add(sel):
        i = sel.index
        sel.annotation.set_text(labels[i])
        sel.annotation.set_fontsize(11)
        sel.annotation.xy = (all_time[i], all_f1[i])

    plt.xlabel("Training Time (s)", fontsize=16)
    plt.ylabel("F1 Score (Validation)", fontsize=16)
    plt.title("Pareto Front: F1 vs. Time (hover for details)", fontsize=18, fontweight='bold')
    plt.legend(loc='lower right', frameon=False, fontsize=14)
    plt.xlim(left=0)
    plt.ylim(bottom=0, top=1)
    plt.tight_layout()
    plt.show()

def main():
    if len(sys.argv) < 2:
        print("Usage: python visualize_pareto.py <nas_results.json>")
        sys.exit(1)
    json_file = sys.argv[1]
    visualize_pareto(json_file)

if __name__ == "__main__":
    main()
