#!/usr/bin/env python3
"""
10_hypothesis_analysis.py — Test all 13 hypotheses using NER outputs.

A. Core football (H1-H5)
B. Men vs Women portrayal (H9-H15)

Produces plots + a summary report.

Usage:
    python final_codes/10_hypothesis_analysis.py
    python final_codes/10_hypothesis_analysis.py --ner-dir data/kaggle_processed/test/ner_outputs/lg
"""

import argparse
import re
import textwrap
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from scipy.stats import chi2_contingency, mannwhitneyu

# ─── Defaults ────────────────────────────────────────────────────────
DEFAULT_NER_DIR = Path("data/kaggle_processed/test/ner_outputs/lg")
DEFAULT_ART_DIR = Path("data/kaggle_processed/test")
DEFAULT_OUT = Path("data/kaggle_processed/test/analysis")

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
})

COLORS = {
    "other": "#2196F3",
    "women": "#E91E63",
    "accent": "#FF9800",
    "grey": "#9E9E9E",
}

RNG_SEED = 42
TEST_LOG = []


# ═══════════════════════════════════════════════════════════════════════
# STATISTICAL HELPERS
# ═══════════════════════════════════════════════════════════════════════

def cohens_d(group1, group2):
    """Compute Cohen's d effect size between two groups."""
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return 0.0
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return (np.mean(group1) - np.mean(group2)) / pooled_std


def cramers_v(contingency_table):
    """Compute Cramér's V from a contingency table."""
    chi2, _, _, _ = chi2_contingency(contingency_table)
    n = contingency_table.sum().sum()
    min_dim = min(contingency_table.shape) - 1
    if min_dim == 0 or n == 0:
        return 0.0
    return np.sqrt(chi2 / (n * min_dim))


def proportion_ztest(count1, nobs1, count2, nobs2):
    """Two-proportion z-test. Returns z-statistic and p-value."""
    p1 = count1 / max(nobs1, 1)
    p2 = count2 / max(nobs2, 1)
    p_pool = (count1 + count2) / max(nobs1 + nobs2, 1)
    se = np.sqrt(p_pool * (1 - p_pool) * (1/max(nobs1, 1) + 1/max(nobs2, 1)))
    if se == 0:
        return 0.0, 1.0
    z = (p1 - p2) / se
    p_val = 2 * (1 - sp_stats.norm.cdf(abs(z)))
    return z, p_val


def effect_label(d):
    """Interpret Cohen's d magnitude."""
    d = abs(d)
    if d < 0.2:
        return "negligible"
    elif d < 0.5:
        return "small"
    elif d < 0.8:
        return "medium"
    else:
        return "large"


def verdict(p_value, effect_size, threshold=0.05):
    """Return structured verdict from p-value + effect size."""
    sig = p_value < threshold
    eff = effect_label(effect_size)
    if sig and eff in ("medium", "large"):
        return "✅ Supported"
    elif sig and eff == "small":
        return "⚠️ Partially Supported"
    elif sig:
        return "⚠️ Weakly Supported (negligible effect)"
    else:
        return "❌ Not Supported"


def print_stat(test_name, stat_val, p_val, eff_size=None, eff_type="Cohen's d"):
    """Print a formatted statistical test result."""
    line = f"    {test_name}: stat={stat_val:.3f}, p={p_val:.6f}"
    if p_val < 0.001:
        line += " (***)"
    elif p_val < 0.01:
        line += " (**)"
    elif p_val < 0.05:
        line += " (*)"
    else:
        line += " (n.s.)"
    if eff_size is not None:
        line += f", {eff_type}={eff_size:.3f} ({effect_label(eff_size)})"
    print(line)
    if eff_size is not None:
        print(f"    → Verdict: {verdict(p_val, eff_size)}")


def compile_term_pattern(terms):
    """
    Compile robust case-insensitive regex for phrase matching.
    - Uses word boundaries to avoid substring artifacts.
    - Allows flexible whitespace for multi-word terms.
    """
    escaped = [re.escape(t).replace(r"\ ", r"\s+") for t in sorted(terms, key=len, reverse=True)]
    return re.compile(r"(?<!\w)(?:" + "|".join(escaped) + r")(?!\w)", re.I)


def contains_term(text, pattern):
    if text is None:
        return False
    return bool(pattern.search(str(text)))


def count_term_hits(text, pattern):
    if text is None:
        return 0
    return len(pattern.findall(str(text)))


def bootstrap_mean_diff_ci(group1, group2, n_boot=2000, alpha=0.05, seed=RNG_SEED):
    """Bootstrap CI for mean(group1) - mean(group2)."""
    g1 = np.asarray(group1, dtype=float)
    g2 = np.asarray(group2, dtype=float)
    if len(g1) == 0 or len(g2) == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        b1 = rng.choice(g1, size=len(g1), replace=True)
        b2 = rng.choice(g2, size=len(g2), replace=True)
        diffs[i] = b1.mean() - b2.mean()
    lo = np.quantile(diffs, alpha / 2)
    hi = np.quantile(diffs, 1 - alpha / 2)
    return (float(lo), float(hi))


def log_test(hypothesis, test_name, stat_value, p_value, effect_size=None, effect_type="", note=""):
    """Collect statistical tests for final multiple-testing correction."""
    TEST_LOG.append(
        {
            "hypothesis": hypothesis,
            "test_name": test_name,
            "statistic": float(stat_value) if stat_value is not None else np.nan,
            "p_value": float(p_value) if p_value is not None else np.nan,
            "effect_size": float(effect_size) if effect_size is not None else np.nan,
            "effect_type": effect_type,
            "note": note,
        }
    )


def benjamini_hochberg(pvals):
    """Benjamini-Hochberg FDR adjustment."""
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    order = np.argsort(pvals)
    ranks = np.arange(1, n + 1)
    sorted_p = pvals[order]
    adjusted = sorted_p * n / ranks
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0, 1)
    q = np.empty_like(adjusted)
    q[order] = adjusted
    return q


def export_test_summary(out_dir: Path):
    """Save a final table of tests with FDR-corrected p-values."""
    if not TEST_LOG:
        return

    df = pd.DataFrame(TEST_LOG)
    valid = df["p_value"].notna()
    if valid.any():
        df.loc[valid, "q_value_bh"] = benjamini_hochberg(df.loc[valid, "p_value"].values)
        df["significant_0_05"] = df["p_value"] < 0.05
        df["significant_fdr_0_05"] = df["q_value_bh"] < 0.05
    else:
        df["q_value_bh"] = np.nan
        df["significant_0_05"] = False
        df["significant_fdr_0_05"] = False

    df.to_csv(out_dir / "hypothesis_tests_summary.csv", index=False)

    with open(out_dir / "hypothesis_tests_summary.txt", "w", encoding="utf-8") as f:
        f.write("Hypothesis Test Summary (with BH-FDR)\n")
        f.write("=" * 72 + "\n")
        for h, sub in df.groupby("hypothesis", sort=False):
            f.write(f"\n{h}\n")
            for _, r in sub.iterrows():
                f.write(
                    f"  - {r['test_name']}: stat={r['statistic']:.3f}, "
                    f"p={r['p_value']:.6f}, q={r.get('q_value_bh', np.nan):.6f}, "
                    f"effect={r['effect_size']:.3f} {r['effect_type']}\n"
                )
                if isinstance(r["note"], str) and r["note"]:
                    f.write(f"    note: {r['note']}\n")


# ═══════════════════════════════════════════════════════════════════════
# LOAD DATA
# ═══════════════════════════════════════════════════════════════════════

def load_data(ner_dir: Path, art_dir: Path):
    other_ents = pd.read_csv(ner_dir / "other_entities.csv")
    women_ents = pd.read_csv(ner_dir / "women_entities.csv")
    other_arts = pd.read_csv(art_dir / "other_articles.csv")
    women_arts = pd.read_csv(art_dir / "women_articles.csv")
    return other_ents, women_ents, other_arts, women_arts


# ═══════════════════════════════════════════════════════════════════════
# H1: Prominence — heavy-tail distribution of PLAYER mentions
# ═══════════════════════════════════════════════════════════════════════

def h1_prominence(other_ents, women_ents, out_dir):
    print("\n" + "="*70)
    print("H1: PLAYER prominence — heavy-tail distribution")
    print("="*70)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    results = {}
    mentions_per_article = {}
    for ax, (label, ents) in zip(axes, [("Men", other_ents), ("Women's", women_ents)]):
        players = ents[ents["entity_label"] == "PLAYER"]
        freq = players["entity_text_norm"].value_counts()
        total = freq.sum()
        top10 = freq.head(10).sum()
        top10_pct = top10 / total * 100 if total > 0 else 0
        top50 = freq.head(50).sum()
        top50_pct = top50 / total * 100 if total > 0 else 0

        results[label] = {
            "unique_players": len(freq),
            "total_mentions": total,
            "top10_share": top10_pct,
            "top50_share": top50_pct,
        }

        # Per-article player mention count for statistical test
        mentions_per_article[label] = players.groupby("article_id").size().values

        # Rank-frequency plot (log-log)
        ranks = np.arange(1, len(freq) + 1)
        ax.loglog(
            ranks,
            freq.values,
            "o-",
            markersize=2,
            color=COLORS["other"] if label == "Men" else COLORS["women"],
        )
        # Annotate a few key ranks so names are visible for quick manual validation.
        ann_ranks = [1, 2, 3, 5, 10, 20, 50]
        ann_ranks = [r for r in ann_ranks if r <= len(freq)]
        for idx, r in enumerate(ann_ranks):
            player_name = str(freq.index[r - 1])
            if len(player_name) > 20:
                player_name = player_name[:20] + "..."
            y_val = float(freq.iloc[r - 1])
            offset_y = 8 if idx % 2 == 0 else -10
            ax.scatter([r], [y_val], s=16, color=COLORS["accent"], zorder=3)
            ax.annotate(
                f"{r}. {player_name}",
                xy=(r, y_val),
                xytext=(6, offset_y),
                textcoords="offset points",
                fontsize=7,
                color="black",
                alpha=0.9,
            )
        ax.set_xlabel("Rank (log)")
        ax.set_ylabel("Frequency (log)")
        ax.set_title(f"{label}: PLAYER Zipf Plot")
        ax.axvline(10, color="red", ls="--", alpha=0.5, label=f"Top-10 = {top10_pct:.1f}%")
        ax.legend()

        print(f"\n  {label}:")
        print(f"    Unique players: {len(freq)}")
        print(f"    Total mentions: {total}")
        print(f"    Top-10 share:   {top10_pct:.1f}%")
        print(f"    Top-50 share:   {top50_pct:.1f}%")
        print(f"    Top-10 players: {', '.join(freq.head(10).index.tolist())}")

    # Statistical test: compare per-article PLAYER mention counts
    u_stat, p_val = mannwhitneyu(
        mentions_per_article["Men"],
        mentions_per_article["Women's"],
        alternative="two-sided",
    )
    d = cohens_d(mentions_per_article["Men"], mentions_per_article["Women's"])
    ci_lo, ci_hi = bootstrap_mean_diff_ci(
        mentions_per_article["Men"],
        mentions_per_article["Women's"],
    )
    print("\n  Statistical test (PLAYER mentions per article):")
    print_stat("Mann-Whitney U", u_stat, p_val, d)
    print(f"    95% bootstrap CI for mean difference (Men - Women): [{ci_lo:.3f}, {ci_hi:.3f}]")
    log_test(
        "H1",
        "Mann-Whitney U (PLAYER mentions per article)",
        u_stat,
        p_val,
        d,
        "Cohen's d",
        note=f"95% CI mean diff Men-Women: [{ci_lo:.3f}, {ci_hi:.3f}]",
    )

    fig.suptitle("H1: PLAYER Prominence — Heavy-Tail Distribution", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "H1_player_prominence.png")
    plt.close(fig)
    return results


# ═══════════════════════════════════════════════════════════════════════
# H2: Club centrality — CLUB co-occurrence network clusters
# ═══════════════════════════════════════════════════════════════════════

def h2_club_centrality(other_ents, women_ents, out_dir):
    print("\n" + "="*70)
    print("H2: CLUB co-occurrence clustering by competition")
    print("="*70)

    def build_cooccurrence(ents):
        clubs_per_article = ents[ents["entity_label"] == "CLUB"].groupby("article_id")["entity_text_norm"].apply(set)
        cooc = Counter()
        for clubs in clubs_per_article:
            clubs_list = sorted(clubs)
            for i in range(len(clubs_list)):
                for j in range(i + 1, len(clubs_list)):
                    cooc[(clubs_list[i], clubs_list[j])] += 1
        return cooc

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    purity_rows = []
    pair_purities = {}

    def competition_purity_metrics(ents):
        """Measure how concentrated club-pair mentions are within single competitions."""
        clubs_per_article = (
            ents[ents["entity_label"] == "CLUB"]
            .groupby("article_id")["entity_text_norm"]
            .apply(lambda x: sorted(set(x)))
        )
        comps_per_article = (
            ents[ents["entity_label"] == "COMPETITION"]
            .groupby("article_id")["entity_text_norm"]
            .apply(lambda x: sorted(set(x)))
        )
        article_ids = sorted(set(clubs_per_article.index) & set(comps_per_article.index))

        pair_comp_counts = defaultdict(Counter)
        for aid in article_ids:
            clubs = clubs_per_article.get(aid, [])
            comps = comps_per_article.get(aid, [])
            if len(clubs) < 2 or len(comps) == 0:
                continue
            for i in range(len(clubs)):
                for j in range(i + 1, len(clubs)):
                    pair = (clubs[i], clubs[j])
                    for comp in comps:
                        pair_comp_counts[pair][comp] += 1

        if not pair_comp_counts:
            return pd.DataFrame(columns=["club_pair", "pair_count", "purity"])

        rows = []
        for pair, comp_counter in pair_comp_counts.items():
            total = sum(comp_counter.values())
            purity = max(comp_counter.values()) / total if total > 0 else 0.0
            rows.append(
                {
                    "club_pair": f"{pair[0]} ↔ {pair[1]}",
                    "pair_count": total,
                    "purity": purity,
                }
            )
        return pd.DataFrame(rows).sort_values("pair_count", ascending=False)

    for ax, (label, ents, col) in zip(axes, [
        ("Men", other_ents, COLORS["other"]),
        ("Women's", women_ents, COLORS["women"]),
    ]):
        cooc = build_cooccurrence(ents)
        top20 = cooc.most_common(20)
        if top20:
            labels_bars = [f"{a} ↔ {b}" for (a, b), _ in top20]
            vals = [v for _, v in top20]
            y_pos = range(len(labels_bars))
            ax.barh(y_pos, vals, color=col, alpha=0.8)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(labels_bars, fontsize=7)
            ax.invert_yaxis()
            ax.set_xlabel("Co-occurrence count")
            ax.set_title(f"{label}: Top-20 Club Co-occurrences")

        print(f"\n  {label} top-10 club pairs:")
        for (a, b), cnt in cooc.most_common(10):
            print(f"    {a} ↔ {b}: {cnt}")

        purity_df = competition_purity_metrics(ents)
        if len(purity_df) > 0:
            weighted_purity = np.average(purity_df["purity"], weights=purity_df["pair_count"])
            median_purity = purity_df["purity"].median()
            pair_purities[label] = purity_df["purity"].values
            print(f"  {label} competition-purity (weighted mean): {weighted_purity:.3f}")
            print(f"  {label} competition-purity (median):       {median_purity:.3f}")
            top_pure = purity_df.sort_values(["purity", "pair_count"], ascending=[False, False]).head(5)
            print("  Top-5 highest-purity pairs:")
            for _, row in top_pure.iterrows():
                print(f"    {row['club_pair']}: purity={row['purity']:.2f}, n={int(row['pair_count'])}")
            purity_rows.append(
                pd.DataFrame(
                    {
                        "group": label,
                        "club_pair": purity_df["club_pair"],
                        "pair_count": purity_df["pair_count"],
                        "purity": purity_df["purity"],
                    }
                )
            )

    # Also check competition-club co-occurrence
    comps_per_article_other = other_ents[other_ents["entity_label"] == "COMPETITION"].groupby("article_id")["entity_text_norm"].apply(set)
    clubs_per_article_other = other_ents[other_ents["entity_label"] == "CLUB"].groupby("article_id")["entity_text_norm"].apply(set)

    comp_club = Counter()
    for art_id in comps_per_article_other.index.intersection(clubs_per_article_other.index):
        for comp in comps_per_article_other[art_id]:
            for club in clubs_per_article_other[art_id]:
                comp_club[(comp, club)] += 1

    print(f"\n  Top competition-club associations (Men):")
    for (comp, club), cnt in comp_club.most_common(15):
        print(f"    {comp} → {club}: {cnt}")

    if pair_purities.get("Men") is not None and pair_purities.get("Women's") is not None:
        u_stat, p_val = mannwhitneyu(
            pair_purities["Men"],
            pair_purities["Women's"],
            alternative="two-sided",
        )
        d = cohens_d(pair_purities["Men"], pair_purities["Women's"])
        ci_lo, ci_hi = bootstrap_mean_diff_ci(
            pair_purities["Men"],
            pair_purities["Women's"],
        )
        print("\n  Statistical test (competition-purity of club pairs):")
        print_stat("Mann-Whitney U", u_stat, p_val, d)
        print(f"    95% bootstrap CI for purity mean diff (Men - Women): [{ci_lo:.3f}, {ci_hi:.3f}]")
        log_test(
            "H2",
            "Mann-Whitney U (competition-purity of club pairs)",
            u_stat,
            p_val,
            d,
            "Cohen's d",
            note=f"95% CI mean diff Men-Women: [{ci_lo:.3f}, {ci_hi:.3f}]",
        )

    if purity_rows:
        pd.concat(purity_rows, ignore_index=True).to_csv(out_dir / "H2_club_pair_competition_purity.csv", index=False)
        print(f"\n  Saved: {out_dir / 'H2_club_pair_competition_purity.csv'}")

    fig.suptitle("H2: Club Co-occurrence Networks", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "H2_club_cooccurrence.png")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
# H3: Manager-focus weeks — MANAGER + negative terms
# ═══════════════════════════════════════════════════════════════════════

def h3_manager_focus(other_ents, women_ents, out_dir):
    print("\n" + "="*70)
    print("H3: Manager/Coach mentions & negative event terms")
    print("="*70)

    negative_terms = {
        "sacked", "fired", "dismissed", "resign", "under pressure",
        "pressure", "crisis", "defeat", "loss", "poor", "disappointing",
        "struggle", "sack", "axe", "axed", "replaced", "interim",
        "out of job", "parted ways", "mutual consent", "relegation",
    }
    manager_pattern = re.compile(r"(?<!\w)(?:manager|coach|head\s*coach|boss|gaffer)(?!\w)", re.I)
    negative_pattern = compile_term_pattern(negative_terms)

    # Collect data for cross-group test
    h3_data = {}
    for label, ents in [("Men", other_ents), ("Women's", women_ents)]:
        # Article-level counts to avoid entity-row duplication artifacts
        article_rows = []
        for aid, group in ents.groupby("article_id"):
            contexts = group["sentence_context"].fillna("").astype(str).str.lower()
            manager_hits = int(sum(count_term_hits(c, manager_pattern) for c in contexts))
            negative_hits = int(sum(count_term_hits(c, negative_pattern) for c in contexts))
            article_rows.append(
                {
                    "article_id": aid,
                    "manager_hits": manager_hits,
                    "negative_hits": negative_hits,
                    "has_manager": manager_hits > 0,
                    "has_manager_negative": manager_hits > 0 and negative_hits > 0,
                }
            )
        art_df = pd.DataFrame(article_rows)
        articles_with_managers = set()
        manager_negative_articles = set()
        if not art_df.empty:
            articles_with_managers = set(art_df.loc[art_df["has_manager"], "article_id"])
            manager_negative_articles = set(art_df.loc[art_df["has_manager_negative"], "article_id"])

        total_articles = ents["article_id"].nunique()
        pct_manager = len(articles_with_managers) / total_articles * 100 if total_articles > 0 else 0
        pct_neg = len(manager_negative_articles) / total_articles * 100 if total_articles > 0 else 0
        h3_data[label] = (len(manager_negative_articles), len(articles_with_managers), total_articles)

        print(f"\n  {label}:")
        print(f"    Articles with manager/coach context: {len(articles_with_managers)} ({pct_manager:.1f}%)")
        print(f"    Articles with manager + negative terms: {len(manager_negative_articles)} ({pct_neg:.1f}%)")

        # Correlation: manager mention intensity vs negative mention intensity
        if len(art_df) > 3:
            rho, p_rho = sp_stats.spearmanr(art_df["manager_hits"], art_df["negative_hits"])
            print(f"    Spearman(manager_hits, negative_hits): rho={rho:.3f}, p={p_rho:.6f}")
            log_test(
                "H3",
                f"Spearman correlation manager-vs-negative ({label})",
                rho,
                p_rho,
                abs(rho),
                "Spearman rho",
            )

    # Statistical test: compare negative-manager proportions
    neg_o, mgr_o, _ = h3_data["Men"]
    neg_w, mgr_w, _ = h3_data["Women's"]
    z, p = proportion_ztest(neg_o, mgr_o, neg_w, mgr_w)
    # Effect size: difference in proportions
    p1 = neg_o / max(mgr_o, 1)
    p2 = neg_w / max(mgr_w, 1)
    h_eff = 2 * np.arcsin(np.sqrt(p1)) - 2 * np.arcsin(np.sqrt(p2))  # Cohen's h
    print("\n  Statistical test (negative rate among manager articles):")
    print_stat("Proportion z-test", z, p, abs(h_eff), "Cohen's h")
    log_test(
        "H3",
        "Proportion z-test (manager-negative rate Men vs Women)",
        z,
        p,
        abs(h_eff),
        "Cohen's h",
    )

    # Visualization: negative terms near manager mentions
    fig, ax = plt.subplots(figsize=(10, 5))
    neg_counts = {"Other": defaultdict(int), "Women": defaultdict(int)}
    neg_term_patterns = {neg: compile_term_pattern({neg}) for neg in negative_terms}
    for label, ents, key in [("Other", other_ents, "Other"), ("Women", women_ents, "Women")]:
        for _, row in ents.iterrows():
            ctx = str(row.get("sentence_context", "")).lower()
            if manager_pattern.search(ctx):
                for neg in negative_terms:
                    if contains_term(ctx, neg_term_patterns[neg]):
                        neg_counts[key][neg] += 1

    all_terms = sorted(
        set(neg_counts["Other"].keys()) | set(neg_counts["Women"].keys()),
        key=lambda t: neg_counts["Other"].get(t, 0) + neg_counts["Women"].get(t, 0),
        reverse=True,
    )[:15]

    x = np.arange(len(all_terms))
    w = 0.35
    ax.bar(x - w/2, [neg_counts["Other"].get(t, 0) for t in all_terms], w,
           label="Men", color=COLORS["other"], alpha=0.8)
    ax.bar(x + w/2, [neg_counts["Women"].get(t, 0) for t in all_terms], w,
           label="Women's", color=COLORS["women"], alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(all_terms, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Frequency")
    ax.set_title("H3: Negative Terms Near Manager/Coach Mentions")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "H3_manager_negative.png")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
# H4: Transfer windows — PLAYER ↔ CLUB co-occurrence + MONEY
# ═══════════════════════════════════════════════════════════════════════

def h4_transfer_windows(other_ents, women_ents, out_dir):
    print("\n" + "="*70)
    print("H4: Transfer signals — PLAYER↔CLUB co-occurrence & MONEY mentions")
    print("="*70)

    transfer_terms = {
        "transfer", "sign", "signing", "signed", "deal", "fee",
        "loan", "buy", "sell", "bid", "offer", "contract",
        "release clause", "free agent", "permanent",
        "swap", "negotiate", "agree", "personal terms",
    }
    transfer_pattern = compile_term_pattern(transfer_terms)

    group_stats = {}
    for label, ents in [("Men", other_ents), ("Women's", women_ents)]:
        art_rows = []
        for aid, group in ents.groupby("article_id"):
            contexts = group["sentence_context"].fillna("").astype(str).str.lower()
            titles = group["article_title"].fillna("").astype(str).str.lower()
            transfer_hits = int(
                sum(count_term_hits(c, transfer_pattern) for c in contexts)
                + sum(count_term_hits(t, transfer_pattern) for t in titles)
            )
            n_players = group.loc[group["entity_label"] == "PLAYER", "entity_text_norm"].nunique()
            n_clubs = group.loc[group["entity_label"] == "CLUB", "entity_text_norm"].nunique()
            money_mentions = int((group["entity_label"] == "MONEY").sum())
            art_rows.append(
                {
                    "article_id": aid,
                    "is_transfer": transfer_hits > 0,
                    "player_club_cooc": n_players * n_clubs,
                    "money_mentions": money_mentions,
                }
            )
        art_df = pd.DataFrame(art_rows)
        transfer_df = art_df[art_df["is_transfer"]]
        non_transfer_df = art_df[~art_df["is_transfer"]]
        group_stats[label] = {"transfer": transfer_df, "non_transfer": non_transfer_df}

        transfer_cooc = transfer_df["player_club_cooc"].mean() if len(transfer_df) else 0.0
        non_transfer_cooc = non_transfer_df["player_club_cooc"].mean() if len(non_transfer_df) else 0.0
        money_rate_transfer = transfer_df["money_mentions"].mean() if len(transfer_df) else 0.0
        money_rate_other = non_transfer_df["money_mentions"].mean() if len(non_transfer_df) else 0.0

        print(f"\n  {label}:")
        print(f"    Transfer articles: {len(transfer_df)}, Non-transfer: {len(non_transfer_df)}")
        print(f"    PLAYER×CLUB co-occurrence — transfer: {transfer_cooc:.2f}, non-transfer: {non_transfer_cooc:.2f}")
        print(f"    MONEY mentions/article — transfer: {money_rate_transfer:.3f}, non-transfer: {money_rate_other:.3f}")

        if len(transfer_df) > 2 and len(non_transfer_df) > 2:
            # Co-occurrence test
            u1, p1 = mannwhitneyu(
                transfer_df["player_club_cooc"],
                non_transfer_df["player_club_cooc"],
                alternative="two-sided",
            )
            d1 = cohens_d(transfer_df["player_club_cooc"], non_transfer_df["player_club_cooc"])
            ci1 = bootstrap_mean_diff_ci(transfer_df["player_club_cooc"], non_transfer_df["player_club_cooc"])
            print(f"    {label} co-occurrence test:")
            print_stat("Mann-Whitney U", u1, p1, d1)
            print(f"    95% CI mean diff (transfer - non-transfer): [{ci1[0]:.3f}, {ci1[1]:.3f}]")
            log_test(
                "H4",
                f"Mann-Whitney U PLAYER×CLUB cooc transfer vs non-transfer ({label})",
                u1,
                p1,
                d1,
                "Cohen's d",
                note=f"95% CI mean diff transfer-non: [{ci1[0]:.3f}, {ci1[1]:.3f}]",
            )

            # MONEY intensity test
            u2, p2 = mannwhitneyu(
                transfer_df["money_mentions"],
                non_transfer_df["money_mentions"],
                alternative="two-sided",
            )
            d2 = cohens_d(transfer_df["money_mentions"], non_transfer_df["money_mentions"])
            ci2 = bootstrap_mean_diff_ci(transfer_df["money_mentions"], non_transfer_df["money_mentions"])
            print(f"    {label} MONEY-rate test:")
            print_stat("Mann-Whitney U", u2, p2, d2)
            print(f"    95% CI mean diff (transfer - non-transfer): [{ci2[0]:.3f}, {ci2[1]:.3f}]")
            log_test(
                "H4",
                f"Mann-Whitney U MONEY mentions transfer vs non-transfer ({label})",
                u2,
                p2,
                d2,
                "Cohen's d",
                note=f"95% CI mean diff transfer-non: [{ci2[0]:.3f}, {ci2[1]:.3f}]",
            )

    # Bar chart
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    categories = ["Transfer\nArticles", "Non-Transfer\nArticles"]
    for ax_idx, (label, col) in enumerate([
        ("Men", COLORS["other"]),
        ("Women's", COLORS["women"]),
    ]):
        transfer_df = group_stats[label]["transfer"]
        non_transfer_df = group_stats[label]["non_transfer"]
        money_tr = transfer_df["money_mentions"].mean() if len(transfer_df) else 0.0
        money_nt = non_transfer_df["money_mentions"].mean() if len(non_transfer_df) else 0.0

        axes[ax_idx].bar(categories, [money_tr, money_nt], color=[col, COLORS["grey"]], alpha=0.8)
        axes[ax_idx].set_ylabel("MONEY mentions per article")
        axes[ax_idx].set_title(f"{label}")

    fig.suptitle("H4: Transfer Window — MONEY Mention Rate", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "H4_transfer_money.png")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
# H5: Injury narrative
# ═══════════════════════════════════════════════════════════════════════

def h5_injury_narrative(other_ents, women_ents, out_dir):
    print("\n" + "="*70)
    print("H5: Injury & fatigue narrative")
    print("="*70)

    injury_terms = {
        "injury", "injured", "hamstring", "knee", "ankle", "muscle",
        "strain", "sprain", "tear", "ligament", "acl", "mcl",
        "concussion", "fracture", "surgery", "operation", "rehab",
        "rehabilitation", "fitness", "fit", "unfit", "doubt",
        "sideline", "sidelined", "ruled out", "miss", "missed",
        "fatigue", "tired", "congested", "fixture pile-up",
        "knock", "setback", "blow", "crocked",
    }
    injury_pattern = compile_term_pattern(injury_terms)

    fig, ax = plt.subplots(figsize=(12, 5))
    per_article_injury = {}
    for label, ents, col in [("Men", other_ents, COLORS["other"]), ("Women's", women_ents, COLORS["women"])]:
        injury_freq = Counter()
        art_hits = []
        term_patterns = {t: compile_term_pattern({t}) for t in injury_terms}
        for _, row in ents.iterrows():
            ctx = str(row.get("sentence_context", "")).lower()
            for term, pat in term_patterns.items():
                if contains_term(ctx, pat):
                    injury_freq[term] += 1
        for aid, group in ents.groupby("article_id"):
            contexts = group["sentence_context"].fillna("").astype(str).str.lower()
            hits = int(sum(count_term_hits(c, injury_pattern) for c in contexts))
            art_hits.append(hits)
        per_article_injury[label] = np.array(art_hits)

        top_terms = injury_freq.most_common(15)
        print(f"\n  {label} — top injury/fatigue terms:")
        for t, c in top_terms:
            print(f"    {t}: {c}")

    # Combined comparison
    terms_union = set()
    counts = {"Other": Counter(), "Women": Counter()}
    term_patterns = {t: compile_term_pattern({t}) for t in injury_terms}
    for key, ents in [("Other", other_ents), ("Women", women_ents)]:
        for _, row in ents.iterrows():
            ctx = str(row.get("sentence_context", "")).lower()
            for term in injury_terms:
                if contains_term(ctx, term_patterns[term]):
                    counts[key][term] += 1
                    terms_union.add(term)

    top_terms = sorted(terms_union, key=lambda t: counts["Other"].get(t, 0) + counts["Women"].get(t, 0), reverse=True)[:15]
    x = np.arange(len(top_terms))
    w = 0.35
    ax.bar(x - w/2, [counts["Other"].get(t, 0) for t in top_terms], w,
           label="Men", color=COLORS["other"], alpha=0.8)
    ax.bar(x + w/2, [counts["Women"].get(t, 0) for t in top_terms], w,
           label="Women's", color=COLORS["women"], alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(top_terms, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Frequency in entity contexts")
    ax.set_title("H5: Injury & Fatigue Narrative Terms")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "H5_injury_narrative.png")
    plt.close(fig)

    # Statistical comparison by article-level injury term intensity
    if len(per_article_injury["Men"]) > 2 and len(per_article_injury["Women's"]) > 2:
        u_stat, p_val = mannwhitneyu(
            per_article_injury["Men"],
            per_article_injury["Women's"],
            alternative="two-sided",
        )
        d = cohens_d(per_article_injury["Men"], per_article_injury["Women's"])
        ci_lo, ci_hi = bootstrap_mean_diff_ci(
            per_article_injury["Men"],
            per_article_injury["Women's"],
        )
        print("\n  Statistical test (injury-term hits per article):")
        print_stat("Mann-Whitney U", u_stat, p_val, d)
        print(f"    95% bootstrap CI for mean diff (Men - Women): [{ci_lo:.3f}, {ci_hi:.3f}]")
        log_test(
            "H5",
            "Mann-Whitney U (injury-term hits per article)",
            u_stat,
            p_val,
            d,
            "Cohen's d",
            note=f"95% CI mean diff Men-Women: [{ci_lo:.3f}, {ci_hi:.3f}]",
        )


# ═══════════════════════════════════════════════════════════════════════
# H9: Individual vs Team emphasis
# ═══════════════════════════════════════════════════════════════════════

def h9_individual_vs_team(other_ents, women_ents, out_dir):
    print("\n" + "="*70)
    print("H9: Individual vs Team emphasis (PLAYER count vs CLUB count per article)")
    print("="*70)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    stats_data = {}
    for label, ents, col in [("Men", other_ents, COLORS["other"]),
                              ("Women's", women_ents, COLORS["women"])]:
        per_art = ents.groupby("article_id").apply(
            lambda g: pd.Series({
                "n_players": g[g["entity_label"] == "PLAYER"]["entity_text_norm"].nunique(),
                "n_clubs": g[g["entity_label"] == "CLUB"]["entity_text_norm"].nunique(),
                "player_mentions": len(g[g["entity_label"] == "PLAYER"]),
                "club_mentions": len(g[g["entity_label"] == "CLUB"]),
                "total_entities": len(g),
            })
        )
        per_art["player_to_club_ratio"] = per_art["n_players"] / per_art["n_clubs"].replace(0, np.nan)
        per_art["player_to_club_ratio"] = per_art["player_to_club_ratio"].fillna(0)

        stats_data[label] = per_art

        avg_players = per_art["n_players"].mean()
        avg_clubs = per_art["n_clubs"].mean()
        player_ratio = per_art["player_mentions"].sum() / max(per_art["total_entities"].sum(), 1)
        club_ratio = per_art["club_mentions"].sum() / max(per_art["total_entities"].sum(), 1)

        print(f"\n  {label}:")
        print(f"    Avg distinct PLAYERs per article: {avg_players:.1f}")
        print(f"    Avg distinct CLUBs per article:   {avg_clubs:.1f}")
        print(f"    PLAYER share of total entities:   {player_ratio*100:.1f}%")
        print(f"    CLUB share of total entities:     {club_ratio*100:.1f}%")
        print(f"    Player-to-Club ratio:             {avg_players/max(avg_clubs, 0.01):.2f}")

    # Plot: distributions
    for ax_idx, metric, title in [
        (0, "n_players", "Distinct PLAYERs per article"),
        (1, "n_clubs", "Distinct CLUBs per article"),
    ]:
        for label, col in [("Men", COLORS["other"]), ("Women's", COLORS["women"])]:
            data = stats_data[label][metric].clip(upper=30)
            axes[ax_idx].hist(data, bins=30, alpha=0.5, color=col, label=label, density=True)
        axes[ax_idx].set_xlabel(title)
        axes[ax_idx].set_ylabel("Density")
        axes[ax_idx].legend()

    fig.suptitle("H9: Individual vs Team Emphasis", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "H9_individual_vs_team.png")
    plt.close(fig)

    # Statistical tests
    t_stat, p_val = sp_stats.mannwhitneyu(
        stats_data["Men"]["n_players"],
        stats_data["Women's"]["n_players"],
        alternative="two-sided",
    )
    d_players = cohens_d(stats_data["Men"]["n_players"], stats_data["Women's"]["n_players"])
    ci_players = bootstrap_mean_diff_ci(
        stats_data["Men"]["n_players"],
        stats_data["Women's"]["n_players"],
    )
    print("\n  Mann-Whitney U test (distinct PLAYERs):")
    print_stat("Mann-Whitney U", t_stat, p_val, d_players)
    print(f"    95% bootstrap CI mean diff (Men - Women): [{ci_players[0]:.3f}, {ci_players[1]:.3f}]")
    log_test(
        "H9",
        "Mann-Whitney U (distinct PLAYERs per article)",
        t_stat,
        p_val,
        d_players,
        "Cohen's d",
        note=f"95% CI mean diff Men-Women: [{ci_players[0]:.3f}, {ci_players[1]:.3f}]",
    )

    u_clubs, p_clubs = sp_stats.mannwhitneyu(
        stats_data["Men"]["n_clubs"],
        stats_data["Women's"]["n_clubs"],
        alternative="two-sided",
    )
    d_clubs = cohens_d(stats_data["Men"]["n_clubs"], stats_data["Women's"]["n_clubs"])
    print("  Mann-Whitney U test (distinct CLUBs):")
    print_stat("Mann-Whitney U", u_clubs, p_clubs, d_clubs)
    log_test("H9", "Mann-Whitney U (distinct CLUBs per article)", u_clubs, p_clubs, d_clubs, "Cohen's d")

    u_ratio, p_ratio = sp_stats.mannwhitneyu(
        stats_data["Men"]["player_to_club_ratio"],
        stats_data["Women's"]["player_to_club_ratio"],
        alternative="two-sided",
    )
    d_ratio = cohens_d(
        stats_data["Men"]["player_to_club_ratio"],
        stats_data["Women's"]["player_to_club_ratio"],
    )
    print("  Mann-Whitney U test (player-to-club ratio):")
    print_stat("Mann-Whitney U", u_ratio, p_ratio, d_ratio)
    log_test("H9", "Mann-Whitney U (player-to-club ratio per article)", u_ratio, p_ratio, d_ratio, "Cohen's d")


# ═══════════════════════════════════════════════════════════════════════
# H10: Name formality — first-name vs surname-only
# ═══════════════════════════════════════════════════════════════════════

def h10_name_formality(other_ents, women_ents, out_dir):
    print("\n" + "="*70)
    print("H10: Name formality — first-name vs surname-only references")
    print("="*70)

    def classify_name(text):
        words = text.strip().split()
        if len(words) == 1:
            return "single_name"  # surname-only (e.g., "Messi")
        elif len(words) == 2:
            return "full_name"    # first + last (e.g., "Lionel Messi")
        else:
            return "multi_name"   # compound (e.g., "Pierre-Emerick Aubameyang")

    fig, ax = plt.subplots(figsize=(8, 5))

    results = {}
    raw_counts = {}
    for label, ents in [("Men", other_ents), ("Women's", women_ents)]:
        players = ents[ents["entity_label"] == "PLAYER"].copy()
        players["name_type"] = players["entity_text"].apply(classify_name)
        counts = players["name_type"].value_counts(normalize=True) * 100
        raw = players["name_type"].value_counts()
        results[label] = counts
        raw_counts[label] = raw
        print(f"\n  {label}:")
        for nt in ["single_name", "full_name", "multi_name"]:
            print(f"    {nt}: {counts.get(nt, 0):.1f}% (n={raw.get(nt, 0)})")

    # Chi-square test on name format distribution
    name_types = ["single_name", "full_name", "multi_name"]
    contingency = pd.DataFrame({
        "Other": [raw_counts["Men"].get(t, 0) for t in name_types],
        "Women": [raw_counts["Women's"].get(t, 0) for t in name_types],
    }, index=name_types)
    chi2, p_val, dof, _ = chi2_contingency(contingency)
    cv = cramers_v(contingency)
    print("\n  Statistical test (name format distribution):")
    print_stat("Chi-square", chi2, p_val, cv, "Cramér's V")
    log_test("H10", "Chi-square (name format distribution)", chi2, p_val, cv, "Cramér's V")

    # Grouped bar
    x = np.arange(len(name_types))
    w = 0.35
    ax.bar(x - w/2, [results["Men"].get(t, 0) for t in name_types], w,
           label="Men", color=COLORS["other"], alpha=0.8)
    ax.bar(x + w/2, [results["Women's"].get(t, 0) for t in name_types], w,
           label="Women's", color=COLORS["women"], alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(["Surname Only\n(e.g. Messi)", "Full Name\n(e.g. Lionel Messi)", "Multi-word\n(e.g. Van Dijk)"])
    ax.set_ylabel("% of PLAYER mentions")
    ax.set_title(f"H10: Name Formality (χ²={chi2:.1f}, p={p_val:.2e}, V={cv:.3f})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "H10_name_formality.png")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
# H11: Role framing — youth/age vs experience/legacy near PLAYER
# ═══════════════════════════════════════════════════════════════════════

def h11_role_framing(other_ents, women_ents, out_dir):
    """Test H11: youth/age framing vs experience/legacy framing near PLAYER mentions."""
    print("\n" + "="*70)
    print("H11: Role framing — youth/age vs experience/legacy near PLAYER")
    print("="*70)

    youth_age_terms = {
        "young", "youngster", "youth", "teen", "teenage", "rookie",
        "prospect", "debut", "up-and-coming", "girl", "schoolgirl",
        "age", "years old", "emerging",
    }
    experience_terms = {
        "experienced", "veteran", "legacy", "leader", "captain",
        "senior", "world-class", "elite", "decorated", "proven",
        "legend", "legendary", "established", "maestro",
    }

    youth_pattern = compile_term_pattern(youth_age_terms)
    exp_pattern = compile_term_pattern(experience_terms)

    fig, ax = plt.subplots(figsize=(9, 5))
    pct_rows = []
    raw_rows = []
    per_article_rates = {}

    for label, ents in [("Men", other_ents), ("Women's", women_ents)]:
        players = ents[ents["entity_label"] == "PLAYER"].copy()
        contexts = players["sentence_context"].fillna("").astype(str).str.lower()
        youth_hits = contexts.apply(lambda x: contains_term(x, youth_pattern)).astype(int)
        exp_hits = contexts.apply(lambda x: contains_term(x, exp_pattern)).astype(int)
        total = len(players)

        youth_count = int(youth_hits.sum())
        exp_count = int(exp_hits.sum())
        youth_pct = youth_count / max(total, 1) * 100
        exp_pct = exp_count / max(total, 1) * 100

        print(f"\n  {label}:")
        print(f"    Youth/age descriptors near PLAYER: {youth_count} ({youth_pct:.2f}%)")
        print(f"    Experience descriptors near PLAYER: {exp_count} ({exp_pct:.2f}%)")

        pct_rows.append({"group": label, "youth_pct": youth_pct, "experience_pct": exp_pct})
        raw_rows.append({"group": label, "youth": youth_count, "experience": exp_count, "total": total})

        by_article = pd.DataFrame(
            {
                "article_id": players["article_id"].values,
                "youth": youth_hits.values,
                "exp": exp_hits.values,
            }
        ).groupby("article_id").sum()
        by_article["youth_rate"] = by_article["youth"] / by_article[["youth", "exp"]].sum(axis=1).replace(0, np.nan)
        by_article["youth_rate"] = by_article["youth_rate"].fillna(0)
        per_article_rates[label] = by_article["youth_rate"].values

    # Chi-square: group x descriptor-type
    contingency = pd.DataFrame(
        {
            "Other": [raw_rows[0]["youth"], raw_rows[0]["experience"]],
            "Women": [raw_rows[1]["youth"], raw_rows[1]["experience"]],
        },
        index=["Youth/Age", "Experience"],
    )
    chi2, p_val, _, _ = chi2_contingency(contingency)
    cv = cramers_v(contingency)
    print("\n  Statistical test (descriptor distribution Youth vs Experience by group):")
    print_stat("Chi-square", chi2, p_val, cv, "Cramér's V")
    log_test("H11", "Chi-square (Youth/Age vs Experience descriptor distribution)", chi2, p_val, cv, "Cramér's V")

    # Per-article youth-rate comparison
    u_stat, p_u = mannwhitneyu(
        per_article_rates["Men"],
        per_article_rates["Women's"],
        alternative="two-sided",
    )
    d = cohens_d(per_article_rates["Men"], per_article_rates["Women's"])
    ci_lo, ci_hi = bootstrap_mean_diff_ci(per_article_rates["Men"], per_article_rates["Women's"])
    print("  Statistical test (per-article youth-rate among youth/experience contexts):")
    print_stat("Mann-Whitney U", u_stat, p_u, d)
    print(f"    95% bootstrap CI mean diff (Men - Women): [{ci_lo:.3f}, {ci_hi:.3f}]")
    log_test(
        "H11",
        "Mann-Whitney U (per-article youth-rate)",
        u_stat,
        p_u,
        d,
        "Cohen's d",
        note=f"95% CI mean diff Men-Women: [{ci_lo:.3f}, {ci_hi:.3f}]",
    )

    pct_df = pd.DataFrame(pct_rows)
    x = np.arange(2)
    w = 0.35
    ax.bar(
        x - w / 2,
        [pct_df.loc[pct_df["group"] == "Men", "youth_pct"].values[0],
         pct_df.loc[pct_df["group"] == "Men", "experience_pct"].values[0]],
        w,
        label="Men",
        color=COLORS["other"],
        alpha=0.85,
    )
    ax.bar(
        x + w / 2,
        [pct_df.loc[pct_df["group"] == "Women's", "youth_pct"].values[0],
         pct_df.loc[pct_df["group"] == "Women's", "experience_pct"].values[0]],
        w,
        label="Women's",
        color=COLORS["women"],
        alpha=0.85,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(["Youth/Age", "Experience/Legacy"])
    ax.set_ylabel("% of PLAYER mentions")
    ax.set_title("H11: Role Framing Near PLAYER Mentions")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "H11_role_framing.png")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
# H12: Attribute mix — mentality/effort vs physicality/tactical
# ═══════════════════════════════════════════════════════════════════════

def h12_attribute_mix(other_ents, women_ents, out_dir):
    print("\n" + "="*70)
    print("H12: Attribute mix — mentality/effort vs physicality/tactical")
    print("="*70)

    mentality_terms = {
        "brave", "courage", "courageous", "determined", "determination",
        "heart", "passion", "passionate", "resilient", "resilience",
        "mental", "mentality", "character", "spirit", "willing",
        "desire", "grit", "gutsy", "battled", "fight", "fighting",
        "hunger", "hungry", "committed", "commitment", "dedicated",
        "inspired", "inspirational", "effort", "work rate", "workrate",
    }

    physical_terms = {
        "powerful", "power", "strong", "strength", "pace", "speed",
        "quick", "fast", "athletic", "physical", "physicality",
        "clinical", "clinical finish", "pressing", "press", "tackle",
        "aerial", "header", "duel", "sprint", "explosive", "agile",
        "agility", "stamina", "endurance", "muscular", "dominant",
        "imposing", "technical", "technique", "tactical", "tactic",
    }
    mentality_pattern = compile_term_pattern(mentality_terms)
    physical_pattern = compile_term_pattern(physical_terms)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax_idx, (term_set, term_label, terms) in enumerate([
        (mentality_terms, "Mentality / Effort", mentality_terms),
        (physical_terms, "Physicality / Tactical", physical_terms),
    ]):
        counts_both = {"Other": Counter(), "Women": Counter()}
        term_patterns = {t: compile_term_pattern({t}) for t in terms}
        for key, ents in [("Other", other_ents), ("Women", women_ents)]:
            players = ents[ents["entity_label"] == "PLAYER"]
            for _, row in players.iterrows():
                ctx = str(row.get("sentence_context", "")).lower()
                for term, term_pat in term_patterns.items():
                    if contains_term(ctx, term_pat):
                        counts_both[key][term] += 1

        all_terms = sorted(
            set(counts_both["Other"].keys()) | set(counts_both["Women"].keys()),
            key=lambda t: counts_both["Other"].get(t, 0) + counts_both["Women"].get(t, 0),
            reverse=True,
        )[:12]

        x = np.arange(len(all_terms))
        w = 0.35
        axes[ax_idx].bar(x - w/2, [counts_both["Other"].get(t, 0) for t in all_terms], w,
                          label="Men", color=COLORS["other"], alpha=0.8)
        axes[ax_idx].bar(x + w/2, [counts_both["Women"].get(t, 0) for t in all_terms], w,
                          label="Women's", color=COLORS["women"], alpha=0.8)
        axes[ax_idx].set_xticks(x)
        axes[ax_idx].set_xticklabels(all_terms, rotation=45, ha="right", fontsize=7)
        axes[ax_idx].set_ylabel("Frequency")
        axes[ax_idx].set_title(f"{term_label} Terms Near PLAYER")
        axes[ax_idx].legend(fontsize=8)

    # Print rates and collect for chi-square
    h12_data = {}
    for label, ents in [("Men", other_ents), ("Women's", women_ents)]:
        players = ents[ents["entity_label"] == "PLAYER"]
        total = len(players)
        ment = sum(1 for _, r in players.iterrows()
                   if contains_term(str(r.get("sentence_context", "")).lower(), mentality_pattern))
        phys = sum(1 for _, r in players.iterrows()
                   if contains_term(str(r.get("sentence_context", "")).lower(), physical_pattern))
        h12_data[label] = {"mentality": ment, "physicality": phys, "total": total}
        print(f"\n  {label}:")
        print(f"    Mentality/effort near PLAYER: {ment} ({ment/max(total,1)*100:.1f}%)")
        print(f"    Physicality/tactical near PLAYER: {phys} ({phys/max(total,1)*100:.1f}%)")

    # Chi-square test: compare mentality-vs-physicality ratio between groups
    contingency = pd.DataFrame({
        "Other": [h12_data["Men"]["mentality"], h12_data["Men"]["physicality"]],
        "Women": [h12_data["Women's"]["mentality"], h12_data["Women's"]["physicality"]],
    }, index=["Mentality", "Physicality"])
    chi2, p_val, dof, _ = chi2_contingency(contingency)
    cv = cramers_v(contingency)
    print("\n  Statistical test (mentality vs physicality distribution):")
    print_stat("Chi-square", chi2, p_val, cv, "Cramér's V")
    log_test("H12", "Chi-square (mentality vs physicality descriptor distribution)", chi2, p_val, cv, "Cramér's V")

    fig.suptitle("H12: Attribute Mix Near PLAYER Mentions", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "H12_attribute_mix.png")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
# H13: Relational framing — relational vs star framing
# ═══════════════════════════════════════════════════════════════════════

def h13_relational_framing(other_ents, women_ents, out_dir):
    print("\n" + "="*70)
    print("H13: Relational vs Star framing near PLAYER")
    print("="*70)

    relational_terms = {
        "captain", "teammate", "team-mate", "partner", "colleague",
        "coach said", "manager said", "alongside", "together",
        "partnership", "combination", "link-up", "connection",
        "support", "helped", "assist", "serve", "collective",
        "group", "squad", "dressing room", "family", "bond",
    }

    star_terms = {
        "star", "superstar", "hero", "genius", "brilliant",
        "magnificent", "world-class", "elite", "best in the world",
        "unstoppable", "unplayable", "single-handedly", "magic",
        "maestro", "virtuoso", "wonder", "sensation", "sensational",
        "incredible", "extraordinary", "remarkable", "stunning",
        "spectacular", "outstanding", "phenomenal", "icon",
    }
    relational_pattern = compile_term_pattern(relational_terms)
    star_pattern = compile_term_pattern(star_terms)

    fig, ax = plt.subplots(figsize=(8, 5))
    results = {}
    raw_data = {}
    for label, ents in [("Men", other_ents), ("Women's", women_ents)]:
        players = ents[ents["entity_label"] == "PLAYER"]
        total = len(players)
        rel_count = sum(1 for _, r in players.iterrows()
                        if contains_term(str(r.get("sentence_context", "")).lower(), relational_pattern))
        star_count = sum(1 for _, r in players.iterrows()
                         if contains_term(str(r.get("sentence_context", "")).lower(), star_pattern))

        results[label] = {
            "relational_pct": rel_count / max(total, 1) * 100,
            "star_pct": star_count / max(total, 1) * 100,
        }
        raw_data[label] = {"relational": rel_count, "star": star_count, "total": total}
        print(f"\n  {label}:")
        print(f"    Relational framing: {rel_count} ({results[label]['relational_pct']:.1f}%)")
        print(f"    Star/hero framing:  {star_count} ({results[label]['star_pct']:.1f}%)")

    # Proportion z-test on relational rate
    z, p = proportion_ztest(
        raw_data["Women's"]["relational"], raw_data["Women's"]["total"],
        raw_data["Men"]["relational"], raw_data["Men"]["total"],
    )
    p1 = raw_data["Women's"]["relational"] / max(raw_data["Women's"]["total"], 1)
    p2 = raw_data["Men"]["relational"] / max(raw_data["Men"]["total"], 1)
    h_eff = abs(2 * np.arcsin(np.sqrt(p1)) - 2 * np.arcsin(np.sqrt(p2)))
    print("\n  Statistical test (relational framing rate):")
    print_stat("Proportion z-test", z, p, h_eff, "Cohen's h")
    log_test("H13", "Proportion z-test (relational framing rate)", z, p, h_eff, "Cohen's h")

    categories = ["Relational\n(captain, teammate...)", "Star / Hero\n(genius, superstar...)"]
    x = np.arange(len(categories))
    w = 0.35
    ax.bar(x - w/2, [results["Men"]["relational_pct"], results["Men"]["star_pct"]],
           w, label="Men", color=COLORS["other"], alpha=0.8)
    ax.bar(x + w/2, [results["Women's"]["relational_pct"], results["Women's"]["star_pct"]],
           w, label="Women's", color=COLORS["women"], alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel("% of PLAYER mentions with framing")
    ax.set_title(f"H13: Relational vs Star Framing (z={z:.2f}, p={p:.2e})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "H13_relational_vs_star.png")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
# H14: Meta-discourse — growth/visibility terms near COMPETITION/CLUB
# ═══════════════════════════════════════════════════════════════════════

def h14_meta_discourse(other_ents, women_ents, out_dir):
    print("\n" + "="*70)
    print("H14: Meta-discourse — growth/visibility near COMPETITION/CLUB")
    print("="*70)

    growth_terms = {
        "growth", "grow", "growing", "momentum", "professionalisation",
        "professionalism", "professional", "visibility", "visible",
        "investment", "invest", "expansion", "expand", "development",
        "develop", "progress", "evolve", "evolution", "milestone",
        "historic", "history", "record", "attendance", "viewership",
        "broadcast", "tv deal", "commercial", "sponsor", "funding",
        "equal pay", "prize money", "revenue", "popularity", "profile",
        "recognition", "awareness", "mainstream", "barrier",
    }
    growth_pattern = compile_term_pattern(growth_terms)

    fig, ax = plt.subplots(figsize=(10, 5))
    results = {}
    raw_data = {}
    for label, ents in [("Men", other_ents), ("Women's", women_ents)]:
        comp_club = ents[ents["entity_label"].isin(["COMPETITION", "CLUB"])]
        total = len(comp_club)
        growth_count = sum(1 for _, r in comp_club.iterrows()
                           if contains_term(str(r.get("sentence_context", "")).lower(), growth_pattern))
        results[label] = growth_count / max(total, 1) * 100
        raw_data[label] = {"growth": growth_count, "total": total}
        print(f"\n  {label}:")
        print(f"    COMP/CLUB mentions with growth terms: {growth_count} ({results[label]:.1f}%)")

    # Proportion z-test on growth rate
    z, p = proportion_ztest(
        raw_data["Women's"]["growth"], raw_data["Women's"]["total"],
        raw_data["Men"]["growth"], raw_data["Men"]["total"],
    )
    p1 = raw_data["Women's"]["growth"] / max(raw_data["Women's"]["total"], 1)
    p2 = raw_data["Men"]["growth"] / max(raw_data["Men"]["total"], 1)
    h_eff = abs(2 * np.arcsin(np.sqrt(p1)) - 2 * np.arcsin(np.sqrt(p2)))
    print("\n  Statistical test (growth term rate):")
    print_stat("Proportion z-test", z, p, h_eff, "Cohen's h")
    log_test("H14", "Proportion z-test (growth term rate near COMPETITION/CLUB)", z, p, h_eff, "Cohen's h")

    # frequency per term
    counts = {"Other": Counter(), "Women": Counter()}
    term_patterns = {t: compile_term_pattern({t}) for t in growth_terms}
    for key, ents in [("Other", other_ents), ("Women", women_ents)]:
        comp_club = ents[ents["entity_label"].isin(["COMPETITION", "CLUB"])]
        for _, row in comp_club.iterrows():
            ctx = str(row.get("sentence_context", "")).lower()
            for term in growth_terms:
                if contains_term(ctx, term_patterns[term]):
                    counts[key][term] += 1

    top_terms = sorted(
        set(counts["Other"].keys()) | set(counts["Women"].keys()),
        key=lambda t: counts["Other"].get(t, 0) + counts["Women"].get(t, 0),
        reverse=True,
    )[:15]

    x = np.arange(len(top_terms))
    w = 0.35
    ax.bar(x - w/2, [counts["Other"].get(t, 0) for t in top_terms], w,
           label="Men", color=COLORS["other"], alpha=0.8)
    ax.bar(x + w/2, [counts["Women"].get(t, 0) for t in top_terms], w,
           label="Women's", color=COLORS["women"], alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(top_terms, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Frequency near COMPETITION/CLUB")
    ax.set_title("H14: Meta-Discussion — Growth / Visibility Terms")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "H14_meta_discourse.png")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
# H15: Credit assignment — individual vs collective in wins
# ═══════════════════════════════════════════════════════════════════════

def h15_credit_assignment(other_ents, women_ents, out_dir):
    print("\n" + "="*70)
    print("H15: Credit assignment — PLAYER share vs CLUB share per article")
    print("="*70)

    win_terms = {"win", "won", "victory", "triumph", "beat", "defeated",
                 "clinch", "clinched", "secure", "secured", "cruise",
                 "cruised", "dominant", "thrash", "demolish", "romp"}
    win_pattern = compile_term_pattern(win_terms)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    all_player_shares = {}  # for cross-gender test
    for ax_idx, (label, ents, col) in enumerate([
        ("Men", other_ents, COLORS["other"]),
        ("Women's", women_ents, COLORS["women"]),
    ]):
        # Identify "win" articles
        win_articles = set()
        for _, row in ents.iterrows():
            ctx = str(row.get("sentence_context", "")).lower()
            title = str(row.get("article_title", "")).lower()
            if contains_term(ctx, win_pattern) or contains_term(title, win_pattern):
                win_articles.add(row["article_id"])

        win_ents = ents[ents["article_id"].isin(win_articles)]
        other_ents_subset = ents[~ents["article_id"].isin(win_articles)]

        # PLAYER share
        for subset, sublabel in [(win_ents, "Win articles"), (other_ents_subset, "Non-win articles")]:
            per_art = subset.groupby("article_id").apply(
                lambda g: pd.Series({
                    "player_share": len(g[g["entity_label"] == "PLAYER"]) / max(len(g), 1),
                    "club_share": len(g[g["entity_label"] == "CLUB"]) / max(len(g), 1),
                })
            )
            if len(per_art) > 0:
                print(f"  {label} — {sublabel}: PLAYER share={per_art['player_share'].mean()*100:.1f}%, "
                      f"CLUB share={per_art['club_share'].mean()*100:.1f}%")

        # Box plot: PLAYER share in win vs non-win
        win_player_shares = win_ents.groupby("article_id").apply(
            lambda g: len(g[g["entity_label"] == "PLAYER"]) / max(len(g), 1)
        )
        nonwin_player_shares = other_ents_subset.groupby("article_id").apply(
            lambda g: len(g[g["entity_label"] == "PLAYER"]) / max(len(g), 1)
        )
        all_player_shares[label] = {"win": win_player_shares.values, "nonwin": nonwin_player_shares.values}

        # Within-category test: win vs non-win
        if len(win_player_shares) > 1 and len(nonwin_player_shares) > 1:
            u, p = mannwhitneyu(win_player_shares, nonwin_player_shares, alternative="two-sided")
            d = cohens_d(win_player_shares.values, nonwin_player_shares.values)
            ci_lo, ci_hi = bootstrap_mean_diff_ci(win_player_shares.values, nonwin_player_shares.values)
            print(f"  {label} — win vs non-win test:")
            print_stat("Mann-Whitney U", u, p, abs(d))
            print(f"    95% bootstrap CI mean diff (win - non-win): [{ci_lo:.3f}, {ci_hi:.3f}]")
            log_test(
                "H15",
                f"Mann-Whitney U PLAYER share win-vs-nonwin ({label})",
                u,
                p,
                abs(d),
                "Cohen's d",
                note=f"95% CI mean diff win-nonwin: [{ci_lo:.3f}, {ci_hi:.3f}]",
            )

        bp = axes[ax_idx].boxplot(
            [win_player_shares.values, nonwin_player_shares.values],
            labels=["Win Articles", "Other Articles"],
            patch_artist=True,
        )
        bp["boxes"][0].set_facecolor(col)
        bp["boxes"][1].set_facecolor(COLORS["grey"])
        for box in bp["boxes"]:
            box.set_alpha(0.6)
        axes[ax_idx].set_ylabel("PLAYER share of entities")
        axes[ax_idx].set_title(f"{label}")

    # Cross-gender test: overall PLAYER share in all articles
    all_other = np.concatenate([all_player_shares["Men"]["win"],
                                all_player_shares["Men"]["nonwin"]])
    all_women = np.concatenate([all_player_shares["Women's"]["win"],
                                all_player_shares["Women's"]["nonwin"]])
    u, p = mannwhitneyu(all_other, all_women, alternative="two-sided")
    d = cohens_d(all_other, all_women)
    print("\n  Cross-gender test (overall PLAYER share):")
    print_stat("Mann-Whitney U", u, p, abs(d))
    log_test("H15", "Mann-Whitney U (overall PLAYER share Men vs Women)", u, p, abs(d), "Cohen's d")

    fig.suptitle("H15: Credit Assignment — PLAYER Share in Win vs Non-Win Articles", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "H15_credit_assignment.png")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
# H16: Source framing bias — source outlet differences
# ═══════════════════════════════════════════════════════════════════════

def h16_source_framing_bias(other_ents, women_ents, other_arts, women_arts, out_dir, min_articles=30, top_n=12):
    """
    Test whether framing metrics vary by source outlet.
    Metrics are computed at article-level and aggregated per source:
      - player_share
      - club_share
      - relational_rate_near_player
      - star_rate_near_player
      - meta_rate_near_comp_club
    """
    print("\n" + "="*70)
    print("H16: Source framing bias — variation across outlets")
    print("="*70)

    relational_terms = {
        "captain", "teammate", "team-mate", "partner", "colleague",
        "coach said", "manager said", "alongside", "together",
        "partnership", "combination", "link-up", "connection",
        "support", "helped", "assist", "serve", "collective",
        "group", "squad", "dressing room", "family", "bond",
    }
    star_terms = {
        "star", "superstar", "hero", "genius", "brilliant",
        "magnificent", "world-class", "elite", "best in the world",
        "unstoppable", "unplayable", "single-handedly", "magic",
        "maestro", "virtuoso", "wonder", "sensation", "sensational",
        "incredible", "extraordinary", "remarkable", "stunning",
        "spectacular", "outstanding", "phenomenal", "icon",
    }
    growth_terms = {
        "growth", "grow", "growing", "momentum", "professionalisation",
        "professionalism", "professional", "visibility", "visible",
        "investment", "invest", "expansion", "expand", "development",
        "develop", "progress", "evolve", "evolution", "milestone",
        "historic", "history", "record", "attendance", "viewership",
        "broadcast", "tv deal", "commercial", "sponsor", "funding",
        "equal pay", "prize money", "revenue", "popularity", "profile",
        "recognition", "awareness", "mainstream", "barrier",
    }
    relational_pattern = compile_term_pattern(relational_terms)
    star_pattern = compile_term_pattern(star_terms)
    growth_pattern = compile_term_pattern(growth_terms)

    def build_article_metrics(ents, arts, group_label):
        source_map = arts[["article_id", "source"]].copy()
        source_map["source"] = source_map["source"].fillna("Unknown").astype(str).str.strip()
        source_map.loc[source_map["source"] == "", "source"] = "Unknown"

        total = ents.groupby("article_id").size().rename("total_entities")
        player_mentions = ents[ents["entity_label"] == "PLAYER"].groupby("article_id").size().rename("player_mentions")
        club_mentions = ents[ents["entity_label"] == "CLUB"].groupby("article_id").size().rename("club_mentions")

        art_df = pd.concat([total, player_mentions, club_mentions], axis=1).fillna(0)
        art_df["player_share"] = art_df["player_mentions"] / art_df["total_entities"].replace(0, np.nan)
        art_df["club_share"] = art_df["club_mentions"] / art_df["total_entities"].replace(0, np.nan)
        art_df["player_share"] = art_df["player_share"].fillna(0.0)
        art_df["club_share"] = art_df["club_share"].fillna(0.0)

        players = ents[ents["entity_label"] == "PLAYER"][["article_id", "sentence_context"]].copy()
        if len(players) > 0:
            pctx = players["sentence_context"].fillna("").astype(str).str.lower()
            players["relational_hit"] = pctx.apply(lambda x: contains_term(x, relational_pattern)).astype(float)
            players["star_hit"] = pctx.apply(lambda x: contains_term(x, star_pattern)).astype(float)
            rel_rate = players.groupby("article_id")["relational_hit"].mean().rename("relational_rate")
            star_rate = players.groupby("article_id")["star_hit"].mean().rename("star_rate")
            art_df = art_df.join(rel_rate, how="left").join(star_rate, how="left")
        else:
            art_df["relational_rate"] = 0.0
            art_df["star_rate"] = 0.0

        comp_club = ents[ents["entity_label"].isin(["COMPETITION", "CLUB"])][["article_id", "sentence_context"]].copy()
        if len(comp_club) > 0:
            cctx = comp_club["sentence_context"].fillna("").astype(str).str.lower()
            comp_club["meta_hit"] = cctx.apply(lambda x: contains_term(x, growth_pattern)).astype(float)
            meta_rate = comp_club.groupby("article_id")["meta_hit"].mean().rename("meta_rate")
            art_df = art_df.join(meta_rate, how="left")
        else:
            art_df["meta_rate"] = 0.0

        art_df = art_df.fillna(0).reset_index()
        art_df = art_df.merge(source_map, on="article_id", how="left")
        art_df["source"] = art_df["source"].fillna("Unknown").astype(str).str.strip()
        art_df.loc[art_df["source"] == "", "source"] = "Unknown"
        art_df["group"] = group_label
        return art_df

    men_article = build_article_metrics(other_ents, other_arts, "Men")
    women_article = build_article_metrics(women_ents, women_arts, "Women's")
    article_level = pd.concat([men_article, women_article], ignore_index=True)

    metrics = ["player_share", "club_share", "relational_rate", "star_rate", "meta_rate"]
    metric_labels = ["PLAYER share", "CLUB share", "Relational near PLAYER", "Star near PLAYER", "Meta near COMP/CLUB"]

    # Keep sources with minimum article support within each group
    valid_parts = []
    for grp in ["Men", "Women's"]:
        g = article_level[article_level["group"] == grp].copy()
        source_counts = g["source"].value_counts()
        valid_sources = source_counts[source_counts >= min_articles].index.tolist()
        if not valid_sources:
            valid_sources = source_counts.head(top_n).index.tolist()
        g = g[g["source"].isin(valid_sources)].copy()
        valid_parts.append(g)

    filtered = pd.concat(valid_parts, ignore_index=True)

    source_summary = (
        filtered.groupby(["group", "source"], as_index=False)
        .agg(
            n_articles=("article_id", "nunique"),
            player_share=("player_share", "mean"),
            club_share=("club_share", "mean"),
            relational_rate=("relational_rate", "mean"),
            star_rate=("star_rate", "mean"),
            meta_rate=("meta_rate", "mean"),
        )
    )
    source_summary.to_csv(out_dir / "H16_source_bias_metrics_by_source.csv", index=False)
    print(f"  Saved: {out_dir / 'H16_source_bias_metrics_by_source.csv'}")

    # Source variability tests (within each group)
    for grp in ["Men", "Women's"]:
        g = filtered[filtered["group"] == grp]
        print(f"\n  {grp} — source coverage:")
        for src, n in g["source"].value_counts().head(12).items():
            print(f"    {src}: {n} articles")

        for metric in metrics:
            groups = [sub[metric].values for _, sub in g.groupby("source") if len(sub) >= min_articles]
            if len(groups) >= 3:
                h_stat, p_val = sp_stats.kruskal(*groups)
                n = sum(len(x) for x in groups)
                k = len(groups)
                eps2 = max((h_stat - k + 1) / max(n - k, 1), 0)
                print(f"    Kruskal-Wallis {metric}: H={h_stat:.3f}, p={p_val:.6f}, epsilon²={eps2:.3f}")
                log_test(
                    "H16",
                    f"Kruskal-Wallis {metric} across sources ({grp})",
                    h_stat,
                    p_val,
                    eps2,
                    "Epsilon^2",
                )

    # Heatmap-like matrix per group
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    heat_vmax = max(5, float((source_summary[metrics] * 100).values.max())) if len(source_summary) else 5
    im = None
    for ax, grp in zip(axes, ["Men", "Women's"]):
        s = (
            source_summary[source_summary["group"] == grp]
            .sort_values("n_articles", ascending=False)
            .head(top_n)
            .copy()
        )
        if len(s) == 0:
            ax.set_axis_off()
            continue

        mat = (s[metrics] * 100.0).to_numpy()
        im = ax.imshow(mat, aspect="auto", cmap="YlGnBu", vmin=0, vmax=heat_vmax)
        ax.set_yticks(np.arange(len(s)))
        ax.set_yticklabels(s["source"].tolist(), fontsize=8)
        ax.set_xticks(np.arange(len(metric_labels)))
        ax.set_xticklabels(metric_labels, rotation=40, ha="right")
        ax.set_title(f"{grp}: Source Framing Metrics (top {len(s)} sources)")

        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax.text(j, i, f"{mat[i, j]:.1f}", ha="center", va="center", fontsize=7, color="black")

    if im is not None:
        cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8)
        cbar.set_label("Rate (%)")
    fig.suptitle("H16: Source Framing Bias by Outlet", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "H16_source_bias_heatmap.png")
    plt.close(fig)

    # Variance plot across sources
    var_rows = []
    for grp in ["Men", "Women's"]:
        s = source_summary[source_summary["group"] == grp]
        if len(s) < 2:
            continue
        for m in metrics:
            var_rows.append(
                {
                    "group": grp,
                    "metric": m,
                    "std_across_sources": float(s[m].std(ddof=1)),
                }
            )
    var_df = pd.DataFrame(var_rows)
    if len(var_df) > 0:
        fig, ax = plt.subplots(figsize=(11, 5))
        metric_order = metrics
        x = np.arange(len(metric_order))
        w = 0.35
        men_vals = [var_df[(var_df["group"] == "Men") & (var_df["metric"] == m)]["std_across_sources"].mean() for m in metric_order]
        women_vals = [var_df[(var_df["group"] == "Women's") & (var_df["metric"] == m)]["std_across_sources"].mean() for m in metric_order]
        ax.bar(x - w / 2, men_vals, w, label="Men", color=COLORS["other"], alpha=0.85)
        ax.bar(x + w / 2, women_vals, w, label="Women's", color=COLORS["women"], alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(metric_labels, rotation=25, ha="right")
        ax.set_ylabel("Std. dev. across sources")
        ax.set_title("H16: Source-to-Source Variability in Framing Metrics")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "H16_source_bias_variability.png")
        plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
# D1: Diagnostic — Entity-label diversity entropy per article
# ═══════════════════════════════════════════════════════════════════════

def d1_entity_diversity_entropy(other_ents, women_ents, out_dir):
    """Additional diagnostic: entropy of entity-label mix per article."""
    print("\n" + "="*70)
    print("D1: Diagnostic — entity-label diversity entropy")
    print("="*70)

    def shannon_entropy(counts):
        total = sum(counts.values())
        if total == 0:
            return 0.0
        p = np.asarray(list(counts.values()), dtype=float) / total
        p = p[p > 0]
        return float(-(p * np.log2(p)).sum())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    data = {}
    for ax, (label, ents, col) in zip(
        axes,
        [("Men", other_ents, COLORS["other"]), ("Women's", women_ents, COLORS["women"])],
    ):
        entropy_vals = []
        for _, g in ents.groupby("article_id"):
            entropy_vals.append(shannon_entropy(g["entity_label"].value_counts().to_dict()))
        entropy_vals = np.asarray(entropy_vals, dtype=float)
        data[label] = entropy_vals
        ax.hist(entropy_vals, bins=40, color=col, alpha=0.7, edgecolor="white")
        ax.axvline(entropy_vals.mean(), color="black", ls="--", lw=1, label=f"mean={entropy_vals.mean():.3f}")
        ax.set_title(label)
        ax.set_xlabel("Entropy (bits)")
        ax.set_ylabel("Articles")
        ax.legend()
        print(f"  {label}: mean={entropy_vals.mean():.3f}, median={np.median(entropy_vals):.3f}, n={len(entropy_vals)}")

    u_stat, p_val = mannwhitneyu(data["Men"], data["Women's"], alternative="two-sided")
    d = cohens_d(data["Men"], data["Women's"])
    ci_lo, ci_hi = bootstrap_mean_diff_ci(data["Men"], data["Women's"])
    print("  Statistical test (entropy per article):")
    print_stat("Mann-Whitney U", u_stat, p_val, d)
    print(f"    95% bootstrap CI mean diff (Men - Women): [{ci_lo:.3f}, {ci_hi:.3f}]")
    log_test(
        "D1",
        "Mann-Whitney U (entity entropy per article)",
        u_stat,
        p_val,
        d,
        "Cohen's d",
        note=f"95% CI mean diff Men-Women: [{ci_lo:.3f}, {ci_hi:.3f}]",
    )

    fig.suptitle("D1: Entity-label Diversity Entropy per Article", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "D1_entity_diversity_entropy.png")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
# SUMMARY DASHBOARD
# ═══════════════════════════════════════════════════════════════════════

def summary_dashboard(other_ents, women_ents, out_dir):
    """Create a combined entity distribution comparison."""
    print("\n" + "="*70)
    print("SUMMARY: Entity Distribution Comparison")
    print("="*70)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax_idx, (label, ents, col) in enumerate([
        ("Men", other_ents, COLORS["other"]),
        ("Women's", women_ents, COLORS["women"]),
    ]):
        dist = ents["entity_label"].value_counts()
        dist_pct = dist / dist.sum() * 100
        bars = axes[ax_idx].barh(range(len(dist_pct)), dist_pct.values, color=col, alpha=0.8)
        axes[ax_idx].set_yticks(range(len(dist_pct)))
        axes[ax_idx].set_yticklabels(dist_pct.index)
        axes[ax_idx].invert_yaxis()
        axes[ax_idx].set_xlabel("% of entities")
        axes[ax_idx].set_title(f"{label} ({len(ents)} entities)")

        for bar, pct in zip(bars, dist_pct.values):
            axes[ax_idx].text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                             f"{pct:.1f}%", va="center", fontsize=8)

    fig.suptitle("Entity Label Distribution: Men vs Women's Coverage", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "summary_entity_distribution.png")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Hypothesis analysis on NER results")
    parser.add_argument("--ner-dir", type=str, default=str(DEFAULT_NER_DIR))
    parser.add_argument("--art-dir", type=str, default=str(DEFAULT_ART_DIR))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUT))
    args = parser.parse_args()

    ner_dir = Path(args.ner_dir)
    art_dir = Path(args.art_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    TEST_LOG.clear()

    print("Loading data...")
    other_ents, women_ents, other_arts, women_arts = load_data(ner_dir, art_dir)
    print(f"  Men entities: {len(other_ents)}, Women entities: {len(women_ents)}")
    print(f"  Men articles: {len(other_arts)}, Women articles: {len(women_arts)}")
    print("  Note: publish_time has heavy missingness in this corpus; H3/H4 are context-based proxies (not true weekly/window time-series tests).")

    # Summary
    summary_dashboard(other_ents, women_ents, out_dir)

    # A. Core football hypotheses
    h1_prominence(other_ents, women_ents, out_dir)
    h2_club_centrality(other_ents, women_ents, out_dir)
    h3_manager_focus(other_ents, women_ents, out_dir)
    h4_transfer_windows(other_ents, women_ents, out_dir)
    h5_injury_narrative(other_ents, women_ents, out_dir)

    # B. Men vs Women portrayal
    h9_individual_vs_team(other_ents, women_ents, out_dir)
    h10_name_formality(other_ents, women_ents, out_dir)
    h11_role_framing(other_ents, women_ents, out_dir)
    h12_attribute_mix(other_ents, women_ents, out_dir)
    h13_relational_framing(other_ents, women_ents, out_dir)
    h14_meta_discourse(other_ents, women_ents, out_dir)
    h15_credit_assignment(other_ents, women_ents, out_dir)
    h16_source_framing_bias(other_ents, women_ents, other_arts, women_arts, out_dir)
    d1_entity_diversity_entropy(other_ents, women_ents, out_dir)

    # Statistical summary table
    export_test_summary(out_dir)
    print(f"\nSaved statistical summary table to: {out_dir / 'hypothesis_tests_summary.csv'}")

    print(f"\n{'='*70}")
    print(f"ALL HYPOTHESES ANALYZED. Plots saved to: {out_dir}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
