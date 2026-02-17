#!/usr/bin/env python3
"""
9_analysis.py – Analyze NER outputs to test football domain hypotheses.

Produces visualizations and statistical summaries for:
  H1: Player prominence (heavy-tail distribution)
  H2: Club co-occurrence networks
  H9: Individual vs team emphasis (men vs women)
  H10: Name formality analysis
  H11-H12: Descriptor context analysis around PLAYER entities
  H14: Meta-discourse in women's coverage
  H15: Credit assignment patterns

Usage:
    python 9_analysis.py                    # Full analysis
    python 9_analysis.py --model lg         # Use specific model output
"""

import argparse
import re
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from config import PROCESSED_DIR, logger

NER_OUTPUT = Path("data/ner_outputs")
PLOTS_DIR = Path("data/plots")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Set plot style
sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["figure.dpi"] = 150


# ═══════════════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_ner_data(model_key: str = "lg"):
    """Load NER entity outputs for both genders."""
    model_dir = NER_OUTPUT / model_key
    men_ents = pd.read_pickle(model_dir / "men_entities.pkl")
    women_ents = pd.read_pickle(model_dir / "women_entities.pkl")
    men_freq = pd.read_pickle(model_dir / "men_entity_freq.pkl")
    women_freq = pd.read_pickle(model_dir / "women_entity_freq.pkl")
    men_articles = pd.read_csv(PROCESSED_DIR / "men_articles.csv")
    women_articles = pd.read_csv(PROCESSED_DIR / "women_articles.csv")
    return men_ents, women_ents, men_freq, women_freq, men_articles, women_articles


# ═══════════════════════════════════════════════════════════════════════════════
# H1: Player Prominence — Heavy-tail distribution
# ═══════════════════════════════════════════════════════════════════════════════

def h1_player_prominence(men_ents, women_ents):
    """Test if player mentions follow heavy-tail distribution."""
    logger.info("\n=== H1: Player Prominence (Heavy-tail Distribution) ===")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax, (gender, ents) in zip(axes, [("Men", men_ents), ("Women", women_ents)]):
        players = ents[ents["entity_label"] == "PLAYER"]
        player_counts = players.groupby("entity_text").size().sort_values(ascending=False)

        if len(player_counts) == 0:
            continue

        # Top 10 share
        total = player_counts.sum()
        top10_share = player_counts.head(10).sum() / total * 100
        top20_share = player_counts.head(20).sum() / total * 100

        logger.info(f"  {gender}: {len(player_counts)} unique players, "
                    f"top 10 = {top10_share:.1f}%, top 20 = {top20_share:.1f}%")

        # Plot rank vs frequency (log-log)
        ranks = np.arange(1, len(player_counts) + 1)
        ax.loglog(ranks, player_counts.values, "o-", markersize=2, alpha=0.7)
        ax.set_xlabel("Rank (log scale)")
        ax.set_ylabel("Mention Frequency (log scale)")
        ax.set_title(f"{gender}'s Football — Player Mention Distribution\n"
                     f"Top 10: {top10_share:.1f}%, Top 20: {top20_share:.1f}%")

        # Annotate top 5
        for i, (name, count) in enumerate(player_counts.head(5).items()):
            ax.annotate(name, (i + 1, count), fontsize=8, rotation=15,
                       ha="left", va="bottom")

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "h1_player_prominence.png", bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved: {PLOTS_DIR}/h1_player_prominence.png")


# ═══════════════════════════════════════════════════════════════════════════════
# H2: Club Co-occurrence Networks
# ═══════════════════════════════════════════════════════════════════════════════

def h2_club_cooccurrence(men_ents, women_ents):
    """Build club co-occurrence networks per article."""
    logger.info("\n=== H2: Club Co-occurrence Networks ===")

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    for ax, (gender, ents) in zip(axes, [("Men", men_ents), ("Women", women_ents)]):
        clubs = ents[ents["entity_label"] == "CLUB"]
        clubs_per_article = clubs.groupby("article_id")["entity_text"].apply(
            lambda x: list(set(x))
        )

        # Build co-occurrence matrix for top 30 clubs
        club_freq = clubs["entity_text"].value_counts().head(30)
        top_clubs = set(club_freq.index)

        cooc = Counter()
        for _, club_list in clubs_per_article.items():
            relevant = [c for c in club_list if c in top_clubs]
            for i, c1 in enumerate(relevant):
                for c2 in relevant[i + 1:]:
                    pair = tuple(sorted([c1, c2]))
                    cooc[pair] += 1

        if not cooc:
            continue

        # Create heatmap
        club_names = list(club_freq.index[:15])  # Top 15 for readability
        matrix = pd.DataFrame(0, index=club_names, columns=club_names)
        for (c1, c2), count in cooc.items():
            if c1 in club_names and c2 in club_names:
                matrix.loc[c1, c2] = count
                matrix.loc[c2, c1] = count

        sns.heatmap(matrix, ax=ax, cmap="YlOrRd", annot=True, fmt="d",
                   cbar_kws={"shrink": 0.8}, linewidths=0.5)
        ax.set_title(f"{gender}'s — Top Club Co-occurrences")
        ax.tick_params(axis="x", rotation=45)
        ax.tick_params(axis="y", rotation=0)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "h2_club_cooccurrence.png", bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved: {PLOTS_DIR}/h2_club_cooccurrence.png")


# ═══════════════════════════════════════════════════════════════════════════════
# H9: Individual vs Team Emphasis (Men vs Women)
# ═══════════════════════════════════════════════════════════════════════════════

def h9_individual_vs_team(men_ents, women_ents, men_articles, women_articles):
    """Compare distinct PLAYER and CLUB entities per article."""
    logger.info("\n=== H9: Individual vs Team Emphasis ===")

    results = {}
    for gender, ents, articles in [("Men", men_ents, men_articles),
                                     ("Women", women_ents, women_articles)]:
        players_per_art = ents[ents["entity_label"] == "PLAYER"].groupby("article_id")["entity_text"].nunique()
        clubs_per_art = ents[ents["entity_label"] == "CLUB"].groupby("article_id")["entity_text"].nunique()

        # Reindex to all articles (fill 0 for articles with no entities)
        all_ids = set(articles["article_id"])
        players_per_art = players_per_art.reindex(all_ids, fill_value=0)
        clubs_per_art = clubs_per_art.reindex(all_ids, fill_value=0)

        results[gender] = {
            "avg_players": players_per_art.mean(),
            "avg_clubs": clubs_per_art.mean(),
            "player_to_club_ratio": players_per_art.mean() / clubs_per_art.mean() if clubs_per_art.mean() > 0 else 0,
            "players_per_art": players_per_art,
            "clubs_per_art": clubs_per_art,
        }
        logger.info(f"  {gender}: avg {results[gender]['avg_players']:.1f} players, "
                    f"{results[gender]['avg_clubs']:.1f} clubs per article, "
                    f"ratio: {results[gender]['player_to_club_ratio']:.2f}")

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Bar chart
    labels = ["Avg Players/Article", "Avg Clubs/Article"]
    men_vals = [results["Men"]["avg_players"], results["Men"]["avg_clubs"]]
    women_vals = [results["Women"]["avg_players"], results["Women"]["avg_clubs"]]

    x = np.arange(len(labels))
    width = 0.35
    axes[0].bar(x - width/2, men_vals, width, label="Men's", color="#3498db")
    axes[0].bar(x + width/2, women_vals, width, label="Women's", color="#e74c3c")
    axes[0].set_ylabel("Average Count")
    axes[0].set_title("H9: Entity Emphasis per Article")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].legend()

    # Distribution plot
    for gender, color in [("Men", "#3498db"), ("Women", "#e74c3c")]:
        vals = results[gender]["players_per_art"]
        axes[1].hist(vals, bins=30, alpha=0.5, label=f"{gender}'s", color=color, density=True)
    axes[1].set_xlabel("Distinct Players per Article")
    axes[1].set_ylabel("Density")
    axes[1].set_title("H9: Distribution of Player Mentions per Article")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "h9_individual_vs_team.png", bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved: {PLOTS_DIR}/h9_individual_vs_team.png")


# ═══════════════════════════════════════════════════════════════════════════════
# H10: Name Formality (first name vs surname)
# ═══════════════════════════════════════════════════════════════════════════════

def h10_name_formality(men_ents, women_ents):
    """Analyze whether women's articles use first names more often."""
    logger.info("\n=== H10: Name Formality Analysis ===")

    results = {}
    for gender, ents in [("Men", men_ents), ("Women", women_ents)]:
        players = ents[ents["entity_label"] == "PLAYER"]["entity_text"]

        single_word = 0
        multi_word = 0
        for name in players:
            parts = name.strip().split()
            if len(parts) == 1:
                single_word += 1  # Likely surname only
            else:
                multi_word += 1   # Full name

        total = single_word + multi_word
        results[gender] = {
            "single_word_pct": single_word / total * 100 if total > 0 else 0,
            "multi_word_pct": multi_word / total * 100 if total > 0 else 0,
            "total": total,
        }
        logger.info(f"  {gender}: {results[gender]['single_word_pct']:.1f}% single-word (surname), "
                    f"{results[gender]['multi_word_pct']:.1f}% multi-word (full name)")

    fig, ax = plt.subplots(figsize=(10, 6))
    labels = ["Single-word\n(Surname)", "Multi-word\n(Full Name)"]
    men_vals = [results["Men"]["single_word_pct"], results["Men"]["multi_word_pct"]]
    women_vals = [results["Women"]["single_word_pct"], results["Women"]["multi_word_pct"]]

    x = np.arange(len(labels))
    width = 0.35
    ax.bar(x - width/2, men_vals, width, label="Men's", color="#3498db")
    ax.bar(x + width/2, women_vals, width, label="Women's", color="#e74c3c")
    ax.set_ylabel("Percentage (%)")
    ax.set_title("H10: Name Formality — Single-word vs Full Name References")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "h10_name_formality.png", bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved: {PLOTS_DIR}/h10_name_formality.png")


# ═══════════════════════════════════════════════════════════════════════════════
# H11-H12: Descriptor Context Analysis
# ═══════════════════════════════════════════════════════════════════════════════

def h11_h12_descriptor_context(men_ents, women_ents):
    """Analyze descriptors near PLAYER mentions (±8 tokens context window)."""
    logger.info("\n=== H11-H12: Descriptor Context Analysis ===")

    # Descriptor lexicons
    mentality_effort = {"brave", "determined", "passionate", "resilient", "committed",
                        "dedicated", "inspiring", "courageous", "tenacious", "fierce",
                        "brilliant", "incredible", "amazing", "emotional", "proud"}
    physicality_tactical = {"powerful", "strong", "fast", "clinical", "pressing",
                            "tactical", "physical", "pace", "strength", "dominant",
                            "composed", "technical", "creative", "dribbling", "aerial"}
    youth_age = {"young", "youth", "teenager", "youngster", "prodigy", "age", "girl",
                 "promising", "emerging", "prospect", "debut", "breakthrough"}
    experience_legacy = {"experienced", "veteran", "legend", "legendary", "career",
                         "legacy", "senior", "captain", "world-class", "elite",
                         "established", "proven", "decorated", "maestro"}

    results = {}
    for gender, ents in [("Men", men_ents), ("Women", women_ents)]:
        players = ents[ents["entity_label"] == "PLAYER"]
        contexts = players["sentence_context"].fillna("").str.lower()

        counts = {
            "mentality/effort": sum(contexts.str.contains("|".join(mentality_effort), na=False)),
            "physicality/tactical": sum(contexts.str.contains("|".join(physicality_tactical), na=False)),
            "youth/age": sum(contexts.str.contains("|".join(youth_age), na=False)),
            "experience/legacy": sum(contexts.str.contains("|".join(experience_legacy), na=False)),
        }

        total = len(contexts)
        pcts = {k: v / total * 100 if total > 0 else 0 for k, v in counts.items()}
        results[gender] = pcts
        logger.info(f"  {gender}: " + ", ".join(f"{k}: {v:.2f}%" for k, v in pcts.items()))

    # Grouped bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    categories = list(results["Men"].keys())
    men_vals = [results["Men"][c] for c in categories]
    women_vals = [results["Women"][c] for c in categories]

    x = np.arange(len(categories))
    width = 0.35
    bars1 = ax.bar(x - width/2, men_vals, width, label="Men's", color="#3498db")
    bars2 = ax.bar(x + width/2, women_vals, width, label="Women's", color="#e74c3c")

    ax.set_ylabel("% of PLAYER mentions with descriptor")
    ax.set_title("H11-H12: Descriptor Types Near PLAYER Mentions")
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=15)
    ax.legend()

    # Add value labels
    for bar in bars1:
        ax.annotate(f"{bar.get_height():.1f}%",
                   xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                   ha="center", va="bottom", fontsize=9)
    for bar in bars2:
        ax.annotate(f"{bar.get_height():.1f}%",
                   xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                   ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "h11_h12_descriptors.png", bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved: {PLOTS_DIR}/h11_h12_descriptors.png")


# ═══════════════════════════════════════════════════════════════════════════════
# H14: Meta-discourse (league growth/visibility)
# ═══════════════════════════════════════════════════════════════════════════════

def h14_meta_discourse(men_ents, women_ents):
    """Check if women's coverage has more meta-discourse about league growth."""
    logger.info("\n=== H14: Meta-discourse Analysis ===")

    meta_terms = {
        "growth", "visibility", "professionalisation", "professionalization",
        "investment", "attendance", "viewership", "broadcast", "rights",
        "sponsorship", "commercial", "development", "gap", "equality",
        "parity", "recognition", "milestone", "historic", "landmark",
        "pioneering", "trailblazing", "barrier", "progress", "expansion",
    }

    results = {}
    for gender, ents in [("Men", men_ents), ("Women", women_ents)]:
        # Look at contexts of COMPETITION and CLUB entities
        relevant = ents[ents["entity_label"].isin(["COMPETITION", "CLUB"])]
        contexts = relevant["sentence_context"].fillna("").str.lower()

        meta_count = sum(contexts.str.contains("|".join(meta_terms), na=False))
        total = len(contexts)
        pct = meta_count / total * 100 if total > 0 else 0

        results[gender] = {"count": meta_count, "total": total, "pct": pct}
        logger.info(f"  {gender}: {meta_count}/{total} ({pct:.2f}%) COMPETITION/CLUB "
                    f"mentions with meta-discourse terms")

    fig, ax = plt.subplots(figsize=(8, 5))
    genders = list(results.keys())
    pcts = [results[g]["pct"] for g in genders]
    colors = ["#3498db", "#e74c3c"]
    bars = ax.bar(genders, pcts, color=colors, width=0.5)
    ax.set_ylabel("% of mentions with meta-discourse terms")
    ax.set_title("H14: Meta-discourse About League Growth/Visibility")
    for bar, pct in zip(bars, pcts):
        ax.annotate(f"{pct:.2f}%", xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                   ha="center", va="bottom", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "h14_meta_discourse.png", bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved: {PLOTS_DIR}/h14_meta_discourse.png")


# ═══════════════════════════════════════════════════════════════════════════════
# H15: Credit Assignment
# ═══════════════════════════════════════════════════════════════════════════════

def h15_credit_assignment(men_ents, women_ents):
    """Analyze if men's reports attribute success to individuals more than women's."""
    logger.info("\n=== H15: Credit Assignment Patterns ===")

    results = {}
    for gender, ents in [("Men", men_ents), ("Women", women_ents)]:
        player_mentions = len(ents[ents["entity_label"] == "PLAYER"])
        club_mentions = len(ents[ents["entity_label"] == "CLUB"])
        total = player_mentions + club_mentions

        results[gender] = {
            "player_share": player_mentions / total * 100 if total > 0 else 0,
            "club_share": club_mentions / total * 100 if total > 0 else 0,
            "player_count": player_mentions,
            "club_count": club_mentions,
        }
        logger.info(f"  {gender}: PLAYER {results[gender]['player_share']:.1f}% vs "
                    f"CLUB {results[gender]['club_share']:.1f}% "
                    f"(ratio: {player_mentions/club_mentions:.2f})" if club_mentions > 0 else "")

    fig, ax = plt.subplots(figsize=(10, 6))
    genders = ["Men's", "Women's"]
    player_shares = [results["Men"]["player_share"], results["Women"]["player_share"]]
    club_shares = [results["Men"]["club_share"], results["Women"]["club_share"]]

    x = np.arange(len(genders))
    width = 0.35
    ax.bar(x - width/2, player_shares, width, label="PLAYER mentions", color="#3498db")
    ax.bar(x + width/2, club_shares, width, label="CLUB mentions", color="#2ecc71")
    ax.set_ylabel("% of Total Entity Mentions")
    ax.set_title("H15: Credit Assignment — Player vs Club Entity Share")
    ax.set_xticks(x)
    ax.set_xticklabels(genders)
    ax.legend()

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "h15_credit_assignment.png", bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved: {PLOTS_DIR}/h15_credit_assignment.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Entity Summary Tables
# ═══════════════════════════════════════════════════════════════════════════════

def entity_summary_tables(men_ents, women_ents, men_freq, women_freq):
    """Generate summary tables of top entities for each category."""
    logger.info("\n=== Entity Summary Tables ===")

    summary_data = []
    for gender, freq in [("Men", men_freq), ("Women", women_freq)]:
        for label in ["PLAYER", "CLUB", "COMPETITION", "LOCATION", "NATIONALITY"]:
            top = freq[freq["entity_label"] == label].head(20)
            for _, row in top.iterrows():
                summary_data.append({
                    "gender": gender,
                    "entity_label": label,
                    "entity_text": row["entity_text"],
                    "frequency": row["frequency"],
                })
            logger.info(f"  {gender} top 5 {label}: " +
                       ", ".join(f"{r['entity_text']}({r['frequency']})" for _, r in top.head(5).iterrows()))

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(NER_OUTPUT / "entity_summary.csv", index=False)
    summary_df.to_pickle(NER_OUTPUT / "entity_summary.pkl")
    logger.info(f"  Saved: {NER_OUTPUT}/entity_summary.csv")

    # Overall comparison
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    for ax, (gender, freq) in zip(axes, [("Men", men_freq), ("Women", women_freq)]):
        label_counts = freq.groupby("entity_label")["frequency"].sum().sort_values(ascending=True)
        label_counts = label_counts[label_counts.index.isin(
            ["PLAYER", "CLUB", "COMPETITION", "LOCATION", "NATIONALITY"]
        )]
        label_counts.plot.barh(ax=ax, color=sns.color_palette("Set2", len(label_counts)))
        ax.set_title(f"{gender}'s — Entity Type Distribution")
        ax.set_xlabel("Total Mentions")

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "entity_distribution.png", bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved: {PLOTS_DIR}/entity_distribution.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Football NER Analysis")
    parser.add_argument("--model", default="lg", help="Model key: sm, lg, trf")
    args = parser.parse_args()

    logger.info(f"Loading NER data for model: {args.model}")
    men_ents, women_ents, men_freq, women_freq, men_articles, women_articles = load_ner_data(args.model)

    logger.info(f"Men: {len(men_ents)} entities, Women: {len(women_ents)} entities")

    # Run all analyses
    entity_summary_tables(men_ents, women_ents, men_freq, women_freq)
    h1_player_prominence(men_ents, women_ents)
    h2_club_cooccurrence(men_ents, women_ents)
    h9_individual_vs_team(men_ents, women_ents, men_articles, women_articles)
    h10_name_formality(men_ents, women_ents)
    h11_h12_descriptor_context(men_ents, women_ents)
    h14_meta_discourse(men_ents, women_ents)
    h15_credit_assignment(men_ents, women_ents)

    logger.info(f"\n{'='*60}")
    logger.info("ALL ANALYSES COMPLETE")
    logger.info(f"Plots saved to: {PLOTS_DIR}/")
    logger.info(f"Data saved to: {NER_OUTPUT}/")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
