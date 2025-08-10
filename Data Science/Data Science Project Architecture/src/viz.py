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

    plt.show()

def plot_hist(df, col, title=None, save=None):
    """
    Plot Histogram Function
    """
    
    if title is None:
        title = f"{col} distribution"
        
    plt.figure()
    df[col].dropna().plot(kind="hist")
    
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.title(title)
    plt.tight_layout()
    
    if save:
        plt.savefig(save)
        
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

    plt.show()
        
    return corr

def positive_rate_by_binary(df, cols, target, top=15, save=None):
    """
    Funciton to Plot a + Rate by Binary Columns
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

    plt.show()
            
    return pd.DataFrame(rates, columns=["feature","pos_rate"])