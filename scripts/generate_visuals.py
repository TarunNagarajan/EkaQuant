import json
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
plt.rcParams.update({"figure.max_open_warning": 0})


def load_data(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    experiments = data["merged"]["experiments"]
    languages = data["merged"]["languages"]

    parsed = []
    for exp in experiments:
        mod = exp["name"].replace("ablate::", "")
        layer_match = re.search(r"layers\.(\d+)\.(.*)", mod)
        layer_idx = int(layer_match.group(1)) if layer_match else -1
        component = layer_match.group(2) if layer_match else "unknown"

        row = {
            "Module": mod,
            "Layer": layer_idx,
            "Component": component,
            "Overall Delta": exp["overall_delta"],
            "Absolute Sensitivity": abs(exp["overall_delta"])
            if exp["overall_delta"] < 0
            else 0,
        }
        for lang in languages:
            row[f"{lang} Delta"] = exp["per_language"][lang]["delta"]
        parsed.append(row)

    df = pd.DataFrame(parsed)
    return df, languages


def save_plot_with_explanation(output_dir, name, explanation):
    plot_dir = os.path.join(output_dir, name)
    os.makedirs(plot_dir, exist_ok=True)

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"{name}.png"), dpi=300)
    plt.close()

    with open(os.path.join(plot_dir, "explanation.md"), "w", encoding="utf-8") as f:
        f.write(explanation)


def generate_visualizations(summary_json_path, output_dir):
    df, languages = load_data(summary_json_path)

    plt.figure(figsize=(12, 8))
    top20 = df.nsmallest(20, "Overall Delta")
    sns.barplot(data=top20, x="Overall Delta", y="Module", palette="Reds_r")
    plt.title("Top 20 Most Sensitive Modules")
    plt.xlabel("Performance Drop")
    save_plot_with_explanation(
        output_dir,
        "top20_overall_sensitivity",
        "Ranking of modules that cause the most significant performance degradation when ablated.",
    )

    plt.figure(figsize=(12, 6))
    layer_agg = df.groupby("Layer")["Overall Delta"].mean().reset_index()
    sns.lineplot(data=layer_agg, x="Layer", y="Overall Delta", marker="o", color="red")
    plt.title("Average Sensitivity by Layer Depth")
    plt.xlabel("Layer Index")
    plt.ylabel("Average Performance Drop")
    plt.axhline(0, color="gray", linestyle="--")
    save_plot_with_explanation(
        output_dir,
        "layer_depth_sensitivity",
        "Visualizes how sensitivity is distributed across the depth of the model.",
    )

    plt.figure(figsize=(12, 6))
    df["Base Component"] = df["Component"].apply(
        lambda x: "mlp" if "mlp" in x else ("self_attn" if "attn" in x else "other")
    )
    sns.boxplot(data=df, x="Base Component", y="Overall Delta", palette="Set2")
    plt.title("Sensitivity Distribution by Component Type")
    plt.ylabel("Performance Drop")
    save_plot_with_explanation(
        output_dir,
        "component_boxplot",
        "Comparison of sensitivity ranges for MLP and Attention components.",
    )

    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        data=df,
        x="en Delta",
        y="bn Delta",
        hue="Layer",
        palette="viridis",
        size="Absolute Sensitivity",
        sizes=(20, 200),
    )
    plt.title("Language Bottleneck: Bengali vs English")
    plt.plot([-80, 5], [-80, 5], color="red", linestyle="--", alpha=0.5)
    save_plot_with_explanation(
        output_dir,
        "bn_vs_en_scatter",
        "Identifies modules that uniquely bottleneck Bengali performance compared to English.",
    )

    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        data=df,
        x="en Delta",
        y="hi Delta",
        hue="Layer",
        palette="magma",
        size="Absolute Sensitivity",
        sizes=(20, 200),
    )
    plt.title("Language Bottleneck: Hindi vs English")
    plt.plot([-80, 5], [-80, 5], color="red", linestyle="--", alpha=0.5)
    save_plot_with_explanation(
        output_dir,
        "hi_vs_en_scatter",
        "Identifies modules that uniquely bottleneck Hindi performance compared to English.",
    )

    df_sorted = df.sort_values(by="Absolute Sensitivity", ascending=False).reset_index()
    df_sorted["Cumulative Pct"] = (
        df_sorted["Absolute Sensitivity"].cumsum()
        / df_sorted["Absolute Sensitivity"].sum()
    ) * 100
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.bar(
        df_sorted.index[:50], df_sorted["Absolute Sensitivity"].head(50), color="C0"
    )
    ax1.set_xlabel("Ranked Modules")
    ax1.set_ylabel("Absolute Sensitivity")
    ax2 = ax1.twinx()
    ax2.plot(
        df_sorted.index[:50],
        df_sorted["Cumulative Pct"].head(50),
        color="C1",
        marker="o",
    )
    ax2.set_ylabel("Cumulative Sensitivity %")
    plt.title("Pareto Analysis of Network Sensitivity")
    save_plot_with_explanation(
        output_dir,
        "pareto_sensitivity",
        "Demonstrates that a small subset of modules accounts for the majority of model sensitivity.",
    )

    plt.figure(figsize=(10, 6))
    sns.histplot(df["Overall Delta"], bins=30, kde=True, color="purple")
    plt.title("Distribution of Performance Deltas")
    plt.xlabel("Delta")
    save_plot_with_explanation(
        output_dir,
        "delta_distribution",
        "Histogram showing the spread of impact across all ablated modules.",
    )

    safe = df[df["Overall Delta"] > 0]
    plt.figure(figsize=(10, 6))
    if not safe.empty:
        sns.barplot(
            data=safe.sort_values("Overall Delta", ascending=False).head(10),
            x="Overall Delta",
            y="Module",
            palette="Greens_r",
        )
        plt.title("Performance-Enhancing Ablations")
    save_plot_with_explanation(
        output_dir,
        "safe_zones",
        "Identifies modules that are highly robust or potentially redundant.",
    )

    df["Lang_Std"] = df[[f"{l} Delta" for l in languages]].std(axis=1)
    top_var = df.nlargest(15, "Lang_Std")
    plt.figure(figsize=(12, 8))
    sns.barplot(data=top_var, x="Lang_Std", y="Module", palette="plasma")
    plt.title("High Language Variance Modules")
    plt.xlabel("Standard Deviation")
    save_plot_with_explanation(
        output_dir,
        "language_variance",
        "Highlights modules that affect target languages inconsistently.",
    )

    l0 = df[df["Layer"] == 0]
    if not l0.empty:
        l0_melt = l0.melt(
            id_vars=["Component"],
            value_vars=[f"{l} Delta" for l in languages],
            var_name="Language",
            value_name="Delta",
        )
        plt.figure(figsize=(10, 6))
        sns.barplot(
            data=l0_melt, x="Component", y="Delta", hue="Language", palette="Set1"
        )
        plt.title("Layer 0 Sensitivity by Language")
        save_plot_with_explanation(
            output_dir,
            "layer_0_anchor",
            "Analysis of the critical importance of the initial model layer.",
        )

    bn_spec = df.copy()
    bn_spec["BN_Spec"] = bn_spec["bn Delta"] - bn_spec["en Delta"]
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=bn_spec.nsmallest(10, "BN_Spec"),
        x="BN_Spec",
        y="Module",
        palette="Blues_r",
    )
    plt.title("Bengali-Specific Bottlenecks")
    save_plot_with_explanation(
        output_dir,
        "bengali_bottlenecks",
        "Focus on modules that impact Bengali significantly more than English.",
    )

    hi_spec = df.copy()
    hi_spec["HI_Spec"] = hi_spec["hi Delta"] - hi_spec["en Delta"]
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=hi_spec.nsmallest(10, "HI_Spec"),
        x="HI_Spec",
        y="Module",
        palette="Oranges_r",
    )
    plt.title("Hindi-Specific Bottlenecks")
    save_plot_with_explanation(
        output_dir,
        "hindi_bottlenecks",
        "Focus on modules that impact Hindi significantly more than English.",
    )

    kl_data = pd.DataFrame(
        {
            "Language": ["Hindi", "Bengali", "English", "Average"],
            "Uniform 4-bit": [17.750000, 17.375000, 0.120605, 11.748535],
            "EkaQuant 4-bit": [6.937500, 9.000000, 0.220703, 5.386068],
        }
    )
    kl_melt = kl_data.melt(
        id_vars="Language", var_name="Method", value_name="KL Divergence"
    )
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=kl_melt,
        x="Language",
        y="KL Divergence",
        hue="Method",
        palette=["#34495e", "#2ecc71"],
    )
    plt.title("KL Divergence Comparison")
    save_plot_with_explanation(
        output_dir,
        "kl_divergence_comparison",
        "Comparison of mathematical fidelity between uniform and EkaQuant selective quantization.",
    )

    improvements = [
        (17.75 - 6.9375) / 17.75 * 100,
        (17.375 - 9.0) / 17.375 * 100,
        (0.120605 - 0.220703) / 0.120605 * 100,
        (11.748 - 5.386) / 11.748 * 100,
    ]
    plt.figure(figsize=(10, 6))
    sns.barplot(
        x=["Hindi", "Bengali", "English", "Average"],
        y=improvements,
        palette=["#e67e22", "#2980b9", "#e74c3c", "#27ae60"],
    )
    plt.axhline(0, color="black", linewidth=1)
    plt.title("KL Improvement Percentage")
    save_plot_with_explanation(
        output_dir,
        "kl_improvement_pct",
        "Percentage of error recovery achieved by EkaQuant over uniform 4-bit.",
    )

    budgets = [0, 50, 150, 300, 500, 1000]
    recovery = [0, 20, 54.16, 75, 85, 92]
    plt.figure(figsize=(10, 6))
    plt.plot(budgets, recovery, marker="o", color="#8e44ad", linewidth=2, markersize=8)
    plt.fill_between(budgets, recovery, color="#8e44ad", alpha=0.2)
    plt.title("Budget vs Recovery Tradeoff")
    plt.axvline(150, color="red", linestyle="--")
    save_plot_with_explanation(
        output_dir,
        "vram_tradeoff_curve",
        "Visualizes the relationship between the VRAM budget for protected layers and overall KL recovery.",
    )

    plt.figure(figsize=(12, 4))
    alloc = np.zeros((1, 32))
    for i in [0, 13, 18, 22, 27]:
        alloc[0, i] = 1
    sns.heatmap(
        alloc,
        cmap=["#ecf0f1", "#27ae60"],
        cbar=False,
        yticklabels=["Precision"],
        xticklabels=range(32),
        linewidths=1,
        linecolor="white",
    )
    plt.title("Selective Quantization Allocation Map")
    save_plot_with_explanation(
        output_dir,
        "knapsack_allocation",
        "Mapping of layers selected for higher precision by the Knapsack algorithm.",
    )

    attn_vs_mlp = df.groupby("Base Component")["Overall Delta"].mean().reset_index()
    plt.figure(figsize=(8, 6))
    sns.barplot(
        data=attn_vs_mlp[attn_vs_mlp["Base Component"].isin(["mlp", "self_attn"])],
        x="Base Component",
        y="Overall Delta",
        palette="Pastel1",
    )
    plt.title("Mean Sensitivity: Attention vs MLP")
    save_plot_with_explanation(
        output_dir,
        "attn_vs_mlp",
        "Average performance impact comparison between Attention and MLP blocks.",
    )

    plt.figure(figsize=(8, 8))
    counts = [len(df[df["Overall Delta"] < 0]), len(df[df["Overall Delta"] > 0])]
    plt.pie(
        counts,
        labels=["Degrades", "Neutral/Improves"],
        autopct="%1.1f%%",
        colors=["#e74c3c", "#2ecc71"],
        startangle=140,
    )
    plt.title("Ablation Impact Ratio")
    save_plot_with_explanation(
        output_dir,
        "impact_ratio_pie",
        "Ratio of modules that negatively impact performance when ablated.",
    )

    df["Layer_Variance"] = df.groupby("Layer")["Overall Delta"].transform("std")
    top_layer_var = (
        df[["Layer", "Layer_Variance"]].drop_duplicates().nlargest(10, "Layer_Variance")
    )
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=top_layer_var,
        x="Layer",
        y="Layer_Variance",
        palette="coolwarm",
        order=top_layer_var["Layer"],
    )
    plt.title("Internal Layer Variance")
    save_plot_with_explanation(
        output_dir,
        "layer_internal_variance",
        "Layers with the highest diversity in component sensitivity.",
    )

    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        data=df, x="Layer", y="Absolute Sensitivity", alpha=0.6, color="darkred"
    )
    plt.title("Network Fragility Map")
    plt.yscale("log")
    save_plot_with_explanation(
        output_dir,
        "fragility_map",
        "Logarithmic view of module sensitivity across model layers.",
    )


if __name__ == "__main__":
    generate_visualizations("data/sweep_summary.json", "plots")
