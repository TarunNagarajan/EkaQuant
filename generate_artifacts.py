import json
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def load_summary(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def generate_artifacts(summary_json_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    summary = load_summary(summary_json_path)
    
    experiments = summary["merged"]["experiments"]
    languages = summary["merged"]["languages"]
    
    # 1. Parse Data
    data = []
    for exp in experiments:
        module_name = exp["name"].replace("ablate::", "")
        row = {
            "Module": module_name,
            "Overall Delta": exp["overall_delta"]
        }
        for lang in languages:
            row[f"{lang} Delta"] = exp["per_language"][lang]["delta"]
        data.append(row)
        
    # Sort by Overall Delta (most negative first)
    data.sort(key=lambda x: x["Overall Delta"])
    df = pd.DataFrame(data)
    
    # 2. Write CSV
    csv_path = os.path.join(output_dir, "sensitivity_ranking.csv")
    df.to_csv(csv_path, index=False)
    print(f"Generated: {csv_path}")
    
    # 3. Write Markdown Heatmap
    md_path = os.path.join(output_dir, "sensitivity_heatmap.md")
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("# Task-Aware Sensitivity Heatmap\n\n")
        f.write("This table shows the performance drop (delta) when specific modules are ablated. **More negative = Higher Sensitivity**.\n\n")
        
        # Headers
        headers = ["Module", "Overall Delta"] + [f"{lang.upper()} Delta" for lang in languages]
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("|"+"|".join(["---:"] * len(headers))+"|\n")
        
        for _, row in df.iterrows():
            def format_score(val):
                if val < -10: return f"**🔴 {val:.2f}**"
                if val < -2: return f"🟠 {val:.2f}"
                if val < -0.5: return f"🟡 {val:.2f}"
                if val > 0: return f"🟢 +{val:.2f}"
                return f"{val:.2f}"
            
            cols = [
                f"`{row['Module']}`",
                format_score(row['Overall Delta'])
            ]
            for lang in languages:
                cols.append(format_score(row[f"{lang} Delta"]))
            
            f.write("| " + " | ".join(cols) + " |\n")
    print(f"Generated: {md_path}")
    
    # 4. Generate Visual Heatmap (PNG)
    try:
        # Filter top 20 most sensitive for visualization
        top_df = df.head(20).set_index("Module")
        
        # Keep only language columns
        lang_cols = [f"{lang} Delta" for lang in languages]
        heat_data = top_df[lang_cols]
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(heat_data, annot=True, cmap="Reds_r", fmt=".2f", linewidths=.5, 
                    cbar_kws={'label': 'Performance Delta (Drop)'})
        plt.title("Top 20 Most Sensitive Modules by Language")
        plt.tight_layout()
        
        png_path = os.path.join(output_dir, "sensitivity_heatmap.png")
        plt.savefig(png_path, dpi=300)
        print(f"Generated: {png_path}")
    except Exception as e:
        print(f"Could not generate PNG heatmap (matplotlib/seaborn might be missing): {e}")

if __name__ == "__main__":
    generate_artifacts("data/sweep_summary.json", "artifacts")
