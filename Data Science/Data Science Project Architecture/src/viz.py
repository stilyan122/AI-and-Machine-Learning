"""
Exploratory visualization helpers.

Matplotlib-only plots for class balance, histograms, boxplots, correlations,
target-rate by categories, decile trends, and co-occurrence/risk heatmaps.
Each function optionally saves to file (no I/O side effects otherwise).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_target_distribution(y, title="Target distribution", save=None):
    """
    Plot Target (chosen) Distribution Function
    """
    
    y = pd.Series(y).astype(int)
    plt.figure()
    
    y.value_counts().sort_index().plot(kind="bar")

    plt.gca().tick_params(axis='x', labelrotation=0)
    
    plt.xlabel(y.name if y.name else "target")
    plt.ylabel("Count")
    plt.title(title)
    plt.tight_layout()
    
    if save:
        plt.savefig(save)
        plt.close()
    else:
        plt.show()

def plot_hist(df, col, title=None, save=None, bins=8):
    """
    Plot Histogram Function
    """
    
    if title is None:
        title = f"{col} distribution"
        
    plt.figure()
    df[col].dropna().plot(kind="hist", bins=bins)
    
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.title(title)
    plt.tight_layout()
    
    if save:
        plt.savefig(save)
        plt.close()
    else:
        plt.show()
    
def boxplot_by_target(df, feature, target, labels=("Class 0","Class 1"), save=None):
    """
    Plot Boxplot By a Given Target Function
    """
    
    data0 = df.loc[df[target]==0, feature].dropna().values
    data1 = df.loc[df[target]==1, feature].dropna().values
    
    plt.figure()
    plt.boxplot([data0, data1], labels=list(labels))
    plt.ylabel(feature)
    plt.title(f"{feature} by {target}")
    plt.tight_layout()
    
    if save:
        plt.savefig(save)
        plt.close()
    else:
        plt.show()

def top_corr_with_target(df, target, k=15, save=None):
    """
    Function to Build Correlation Plot By a Given Target
    """
    
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if target in num_cols:
        num_cols.remove(target)
        
    corr = df[num_cols].corrwith(df[target]).abs().sort_values(ascending=False).head(k)
    plt.figure()
    corr.plot(kind="bar")
    
    plt.gca().tick_params(axis='x', labelrotation=70)
    plt.ylabel("|Pearson correlation|")
    plt.title(f"Top {k} numeric features correlated with {target}")
    plt.tight_layout()
    
    if save:
        plt.savefig(save)
        plt.close()
    else:
        plt.show()
        
    return corr

def positive_rate_by_binary(df, cols, target, top=15, save=None):
    """
    Function to Plot a + Rate by Binary Columns
    """
    
    rates = []
    for c in cols:
        vals = df[c].dropna().unique()
        if set(vals).issubset({0,1}):
            if (df[c]==1).any():
                r = df.loc[df[c]==1, target].mean()
                rates.append((c, float(r)))
                
    rates = sorted(rates, key=lambda t: t[1], reverse=True)[:top]
    
    if rates:
        plt.figure()
        names = [r[0] for r in rates]
        vals  = [r[1] for r in rates]
        
        plt.bar(names, vals)
        plt.xticks(rotation=80)
        plt.ylabel(f"P({target}=1 | feature=1)")
        plt.title("Positive rate by binary feature")
        plt.tight_layout()
        
        if save:
            plt.savefig(save)
            plt.close()
        else:
            plt.show()
            
    return pd.DataFrame(rates, columns=["feature","pos_rate"])

def decile_trend_plot(df, col, target="Diagnosis", q=10, save=None):
    """
    Function to plot target-rate across deciles
    """
    
    qbins = pd.qcut(df[col], q=q, duplicates="drop")
    grp = df.groupby(qbins, observed=False)[target]
    rate = grp.mean()
    n = grp.size()
    
    plt.figure()
    rate.reset_index(drop=True).plot(marker="o")
    plt.xlabel(f"{col} deciles (low -> high)")
    plt.ylabel(f"Mean {target}")
    plt.title(f"Asthma rate across {col} deciles")
    
    for i, v in enumerate(rate.values):
        plt.text(i, v, str(int(n.iloc[i])), ha="center", va="bottom", fontsize=8)
        
    plt.tight_layout()
    
    if save:
        plt.savefig(save)
        plt.close()
    else:
        plt.show()

def plot_target_rate_by_category(df, col, target="Diagnosis", order="index", observed=False, save=None):
    """
    Bar chart of P(target=1) by categorical code column function
    """
    
    tbl = df.groupby(col, observed=observed)[target].agg(["mean","size"])
    
    if order == "rate":
        tbl = tbl.sort_values("mean")
    else:
        tbl = tbl.sort_index()

    plt.figure()
    plt.bar(tbl.index.astype(str), tbl["mean"].values)
    
    for i, v in enumerate(tbl["mean"].values):
        plt.text(i, v, str(int(tbl["size"].iloc[i])), ha="center", va="bottom", fontsize=8)  
    
    plt.xlabel(col)
    plt.ylabel(f"P({target}=1)")
    plt.title(f"Asthma rate by {col}")
    plt.tight_layout()
    
    if save:
        plt.savefig(save)
        plt.close()
    else:
        plt.show()

def plot_symptom_count_vs_rate(df, symptom_cols, target="Diagnosis", save=None):
    """
    Sum a set of 0/1 symptom flags per row, then plot P(target=1) vs that count (with counts annotated)
    """
    
    tmp = df.copy()
    tmp["symptom_count"] = tmp[symptom_cols].sum(axis=1)
    rate = tmp.groupby("symptom_count", observed=False)[target].mean()
    n = tmp.groupby("symptom_count", observed=False)[target].size()

    plt.figure()
    plt.bar(rate.index.astype(int), rate.values)
    
    for i, v in enumerate(rate.values):
        plt.text(i, v, str(int(n.iloc[i])), ha="center", va="bottom", fontsize=8)
        
    plt.xlabel("Symptom count (sum of 0/1 flags)")
    plt.ylabel(f"P({target}=1)")
    plt.title("Asthma rate vs symptom count")
    plt.tight_layout()
    
    if save:
        plt.savefig(save)
        plt.close()
    else:
        plt.show()

def plot_risk_heatmap_2x2(df, a, b, target="Diagnosis", observed=False, save=None):
    """
    2x2 heatmap of P(target=1) for a pair of binary features (a,b) function
    """
    
    g = df.groupby([a, b], observed=observed)[target].mean().unstack()
    
    plt.figure()
    plt.imshow(g.values, interpolation="nearest")
    plt.xticks([0,1], [f"{b}=0", f"{b}=1"])
    plt.yticks([0,1], [f"{a}=0", f"{a}=1"])
    plt.xlabel(b); plt.ylabel(a)
    plt.title(f"P({target}=1) by {a} & {b}")
    
    for i in range(2):
        for j in range(2):
            try:
                val = g.iloc[i, j]
                if pd.notna(val):
                    plt.text(j, i, f"{val:.2f}", ha="center", va="center")
            except Exception:
                pass
                
    plt.tight_layout()
    
    if save:
        plt.savefig(save)
        plt.close()
    else:
        plt.show()

def plot_symptom_cooccurrence(df, cols, save=None):
    """
    Heatmap of pairwise co-occurrence rates: P(a=1 AND b=1) function
    """
    
    cols = [c for c in cols if c in df.columns]
    m = np.zeros((len(cols), len(cols)))
    for i, a in enumerate(cols):
        ai = df[a].eq(1)
        for j, b in enumerate(cols):
            m[i, j] = (ai & df[b].eq(1)).mean()

    plt.figure()
    plt.imshow(m, interpolation="nearest")
    plt.xticks(range(len(cols)), cols, rotation=90)
    plt.yticks(range(len(cols)), cols)
    
    for i in range(len(cols)):
        for j in range(len(cols)):
            plt.text(j, i, f"{m[i,j]:.2f}", ha="center", va="center", fontsize=7)
   
    plt.title("Symptom co-occurrence rate")
    plt.tight_layout()
    
    if save:
        plt.savefig(save)
        plt.close()
    else:
        plt.show()