"""Shared plot style + the report figures. One house style, applied everywhere."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd


def style_axes(ax, title=None, subtitle=None, ylabel=None, xlabel=None):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#888888")
    ax.tick_params(colors="#444444", labelsize=10)
    ax.grid(axis="y", color="#dddddd", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, fontsize=14, fontweight="bold", color="#222222", loc="left", pad=28)
    if subtitle:
        ax.text(0, 1.03, subtitle, transform=ax.transAxes, fontsize=10.5, color="#666666")
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10.5, color="#444444")
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=10.5, color="#444444")
    return ax


def _fmt_count(x, _=None):
    return f"{x/1e6:.1f}M" if x >= 1e6 else f"{x/1e3:.0f}K"


def stock_pivot(wide: pd.DataFrame) -> pd.DataFrame:
    s = wide.dropna(subset=["Total_Indian_Migrants"])
    return s.pivot(index="Year", columns="Destination_Country", values="Total_Indian_Migrants")


def plot_stock_trajectories(wide, colors, path):
    piv = stock_pivot(wide)
    order = piv.iloc[-1].sort_values(ascending=False).index
    fig, ax = plt.subplots(figsize=(10, 6.5))
    for c in order:
        s = piv[c].dropna()
        ax.plot(s.index, s.values, marker="o", ms=5, lw=2.2, color=colors[c], zorder=3)
        ax.annotate(f"{c}  {_fmt_count(s.values[-1])}", (s.index[-1], s.values[-1]),
                    xytext=(6, 0), textcoords="offset points", fontsize=9.5,
                    color=colors[c], fontweight="bold", va="center")
    ax.set_yscale("log")
    ax.set_xlim(1999, 2029)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_fmt_count))
    style_axes(ax, "Diaspora growth is steepest outside the USA",
               "Indian-born population by destination, log scale (UN DESA quinquennial stock)",
               "Indian-born residents (log scale)", "Year")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_growth(wide, colors, path):
    piv = stock_pivot(wide)
    g = ((piv.loc[2024] / piv.loc[2000] - 1) * 100).dropna().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars = ax.barh(g.index, g.values, color=[colors[c] for c in g.index], zorder=3)
    for b, v in zip(bars, g.values):
        ax.text(v + 8, b.get_y() + b.get_height() / 2, f"+{v:.0f}%", va="center",
                fontsize=10, fontweight="bold", color="#333333")
    ax.invert_yaxis(); ax.set_xlim(0, g.max() * 1.22)
    style_axes(ax, "Canada and Australia out-grew the US diaspora more than 2-to-1",
               "% growth in Indian-born population, 2000 to 2024 (both endpoints observed)",
               xlabel="Growth 2000-2024 (%)")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_uae(wide, color, path):
    uae = stock_pivot(wide)["UAE"].dropna()
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(uae.index, uae / 1e6, marker="o", ms=7, lw=2.5, color=color, zorder=3)
    ax.fill_between(uae.index, uae / 1e6, alpha=0.08, color=color)
    pk = uae.idxmax()
    ax.annotate(f"peak: {uae.max()/1e6:.2f}M ({pk})", (pk, uae.max() / 1e6),
                xytext=(pk - 5, uae.max() / 1e6 + 0.25), fontsize=10, color=color,
                fontweight="bold", arrowprops=dict(arrowstyle="->", color=color))
    d = (uae.iloc[-1] - uae.max()) / 1e3
    ax.annotate(f"{d:+.0f}K by {uae.index[-1]}", (uae.index[-1], uae.iloc[-1] / 1e6),
                xytext=(uae.index[-1] - 3.5, uae.iloc[-1] / 1e6 - 0.4), fontsize=10,
                color=color, fontweight="bold", arrowprops=dict(arrowstyle="->", color=color))
    style_axes(ax, "The UAE is the only destination where the diaspora shrank",
               "Indian-born population in the UAE (UN DESA)", "Millions", "Year")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_macro_collinearity(wide, path):
    m = (wide.drop_duplicates("Year").sort_values("Year")
             [["Year", "GDP_India", "Remittance_USD_Billion", "Unemployment_India"]]
             .dropna(subset=["GDP_India"]))
    norm = lambda s: (s - s.min()) / (s.max() - s.min())
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    ax.plot(m.Year, norm(m.Year), color="#999999", ls=":", lw=2, label="Year (reference)")
    ax.plot(m.Year, norm(m.GDP_India), color="#2C5F8A", lw=2.4, label=f"GDP  (r={m.Year.corr(m.GDP_India):.2f} with Year)")
    ax.plot(m.Year, norm(m.Remittance_USD_Billion), color="#D4A017", lw=2.4,
            label=f"Remittances  (r={m.Year.corr(m.Remittance_USD_Billion):.2f})")
    ax.plot(m.Year, norm(m.Unemployment_India), color="#E8734A", lw=2.2, ls="--",
            label=f"Unemployment  (r={m.Year.corr(m.Unemployment_India):.2f})")
    ax.legend(frameon=False, fontsize=9.5, loc="upper left")
    style_axes(ax, "GDP and remittances are time trends in disguise",
               "Each series min-max scaled to [0, 1]; unemployment is the only one not tightly bound to time",
               "Scaled value", "Year")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_leaderboard(lb: pd.DataFrame, path):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    y = np.arange(len(lb))
    ax.barh(y + 0.2, lb["train_R2"], 0.38, color="#B8C7D6", label="Train R2", zorder=3)
    ax.barh(y - 0.2, lb["LOOCV_R2"], 0.38, color="#2C5F8A", label="LOOCV R2", zorder=3)
    for i, r in lb.iterrows():
        ax.text(r["LOOCV_R2"] + 0.005, i - 0.2, f"{r['LOOCV_R2']:.3f}", va="center", fontsize=9.5, fontweight="bold")
    ax.set_yticks(y); ax.set_yticklabels(lb["model"]); ax.invert_yaxis()
    ax.set_xlim(0.7, 1.02); ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, fontsize=9)
    style_axes(ax, "Simplest model wins under leave-one-out CV",
               "Train vs held-out R2 on log(migrant stock); n=35", xlabel="R2")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_residuals(meta: pd.DataFrame, resid, colors, path):
    d = meta.assign(residual=resid)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for c, sub in d.groupby("Destination_Country"):
        ax.scatter(sub.Year, sub.residual, s=70, color=colors[c], label=c, zorder=3,
                   edgecolor="white", linewidth=0.6)
    ax.axhline(0, color="#888888", lw=1)
    ax.legend(frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    style_axes(ax, "Where the baseline misses: Germany and the UK's endpoints",
               "LOOCV residuals (actual - predicted, log scale), linear baseline",
               "Residual (log scale)", "Year")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
