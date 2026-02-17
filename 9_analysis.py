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

NER_OUTPUT_DEFAULT = Path("data/ner_outputs")
PLOTS_DIR_DEFAULT = Path("data/plots")
PROCESSED_DIR_DEFAULT = PROCESSED_DIR

# Set plot style
sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["figure.dpi"] = 150


# ═══════════════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_ner_data(data_dir: Path, ner_dir: Path, model_key: str = "lg"):
    """Load NER entity outputs for both genders."""
    model_dir = ner_dir / model_key
    
    logger.info(f"Loading NER data from {model_dir}")
    men_ents = pd.read_pickle(model_dir / "men_entities.pkl")
    women_ents = pd.read_pickle(model_dir / "women_entities.pkl")
    men_freq = pd.read_pickle(model_dir / "men_entity_freq.pkl")
    women_freq = pd.read_pickle(model_dir / "women_entity_freq.pkl")
    
    logger.info(f"Loading articles from {data_dir}")
    men_articles = pd.read_csv(data_dir / "men_articles.csv")
    women_articles = pd.read_csv(data_dir / "women_articles.csv")
    
    return men_ents, women_ents, men_freq, women_freq, men_articles, women_articles


# ═══════════════════════════════════════════════════════════════════════════════
# H1: Player Prominence — Heavy-tail distribution
# ═══════════════════════════════════════════════════════════════════════════════

def h1_player_prominence(men_ents, women_ents, plots_dir):
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
    plt.savefig(plots_dir / "h1_player_prominence.png", bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved: {plots_dir}/h1_player_prominence.png")


# ═══════════════════════════════════════════════════════════════════════════════
# H2: Club Co-occurrence Networks
# ═══════════════════════════════════════════════════════════════════════════════

def h2_club_cooccurrence(men_ents, women_ents, plots_dir):
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
    plt.savefig(plots_dir / "h2_club_cooccurrence.png", bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved: {plots_dir}/h2_club_cooccurrence.png")


# ═══════════════════════════════════════════════════════════════════════════════
# H9: Individual vs Team Emphasis (Men vs Women)
# ═══════════════════════════════════════════════════════════════════════════════

def h9_individual_vs_team(men_ents, women_ents, men_articles, women_articles, plots_dir):
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
    plt.savefig(plots_dir / "h9_individual_vs_team.png", bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved: {plots_dir}/h9_individual_vs_team.png")


# ═══════════════════════════════════════════════════════════════════════════════
# H10: Name Formality (first name vs surname)
# ═══════════════════════════════════════════════════════════════════════════════

def h10_name_formality(men_ents, women_ents, plots_dir):
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
    plt.savefig(plots_dir / "h10_name_formality.png", bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved: {plots_dir}/h10_name_formality.png")


# ═══════════════════════════════════════════════════════════════════════════════
# H11-H12: Descriptor Context Analysis
# ═══════════════════════════════════════════════════════════════════════════════

def h11_h12_descriptor_context(men_ents, women_ents, plots_dir):
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
    plt.savefig(plots_dir / "h11_h12_descriptors.png", bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved: {plots_dir}/h11_h12_descriptors.png")


# ═══════════════════════════════════════════════════════════════════════════════
# H14: Meta-discourse (league growth/visibility)
# ═══════════════════════════════════════════════════════════════════════════════

def h14_meta_discourse(men_ents, women_ents, plots_dir):
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
    plt.savefig(plots_dir / "h14_meta_discourse.png", bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved: {plots_dir}/h14_meta_discourse.png")


# ═══════════════════════════════════════════════════════════════════════════════
# H15: Credit Assignment
# ═══════════════════════════════════════════════════════════════════════════════

def h15_credit_assignment(men_ents, women_ents, plots_dir):
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
    plt.savefig(plots_dir / "h15_credit_assignment.png", bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved: {plots_dir}/h15_credit_assignment.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Entity Summary Tables
# ═══════════════════════════════════════════════════════════════════════════════

def entity_summary_tables(men_ents, women_ents, men_freq, women_freq, plots_dir):
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
    # Note: Saving this is trickier if we want it in plots_dir or ner_dir. 
    # But usually this goes to NER output. However, the function takes plots_dir. 
    # Let's save it to plots_dir for simplicity as 'analysis_summary.csv'
    summary_df.to_csv(plots_dir / "entity_summary.csv", index=False)
    # summary_df.to_pickle(NER_OUTPUT / "entity_summary.pkl") # Skip pickle to avoid global ref
    logger.info(f"  Saved: {plots_dir}/entity_summary.csv")

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
    plt.savefig(plots_dir / "entity_distribution.png", bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved: {plots_dir}/entity_distribution.png")


# ═══════════════════════════════════════════════════════════════════════════════
# NER Diagnostics: Coverage + Quality-Oriented Visuals
# ═══════════════════════════════════════════════════════════════════════════════

def ner_coverage_diagnostics(men_ents, women_ents, men_articles, women_articles, plots_dir):
    """Visualize article-level NER coverage and save a compact summary table."""
    logger.info("\n=== NER Coverage Diagnostics ===")

    rows = []
    for gender, ents, articles in [
        ("Men", men_ents, men_articles),
        ("Women", women_ents, women_articles),
    ]:
        total_articles = len(articles)
        with_entities = ents["article_id"].nunique()
        without_entities = max(total_articles - with_entities, 0)
        avg_entities_all = len(ents) / total_articles if total_articles > 0 else 0
        avg_entities_processed = len(ents) / with_entities if with_entities > 0 else 0

        rows.append(
            {
                "gender": gender,
                "total_articles": total_articles,
                "articles_with_entities": with_entities,
                "articles_without_entities": without_entities,
                "coverage_pct": with_entities / total_articles * 100 if total_articles > 0 else 0,
                "avg_entities_per_article_all": avg_entities_all,
                "avg_entities_per_article_with_entities": avg_entities_processed,
            }
        )
        logger.info(
            f"  {gender}: {with_entities}/{total_articles} articles with entities "
            f"({rows[-1]['coverage_pct']:.1f}%), avg entities/article={avg_entities_all:.1f}"
        )

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(plots_dir / "ner_coverage_summary.csv", index=False)
    logger.info(f"  Saved: {plots_dir}/ner_coverage_summary.csv")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    x = np.arange(len(summary_df))
    with_vals = summary_df["articles_with_entities"].values
    without_vals = summary_df["articles_without_entities"].values
    labels = summary_df["gender"].tolist()

    axes[0].bar(x, with_vals, label="With entities", color="#2e86de")
    axes[0].bar(x, without_vals, bottom=with_vals, label="Without entities", color="#d0d7de")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylabel("Article count")
    axes[0].set_title("NER Coverage by Article")
    axes[0].legend()

    width = 0.35
    axes[1].bar(
        x - width / 2,
        summary_df["avg_entities_per_article_all"].values,
        width,
        label="All articles",
        color="#16a085",
    )
    axes[1].bar(
        x + width / 2,
        summary_df["avg_entities_per_article_with_entities"].values,
        width,
        label="Articles with entities",
        color="#f39c12",
    )
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylabel("Average entity mentions")
    axes[1].set_title("Entity Density by Coverage Mode")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(plots_dir / "ner_coverage_diagnostics.png", bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved: {plots_dir}/ner_coverage_diagnostics.png")


def ner_entities_per_article_distribution(men_ents, women_ents, men_articles, women_articles, plots_dir):
    """Show per-article entity density and major-label load."""
    logger.info("\n=== NER Per-Article Distribution ===")

    def build_counts(ents: pd.DataFrame, articles: pd.DataFrame, gender: str) -> pd.DataFrame:
        article_ids = articles["article_id"].astype(str).tolist()
        total = ents.groupby("article_id").size().reindex(article_ids, fill_value=0)
        players = (
            ents[ents["entity_label"] == "PLAYER"]
            .groupby("article_id")
            .size()
            .reindex(article_ids, fill_value=0)
        )
        clubs = (
            ents[ents["entity_label"] == "CLUB"]
            .groupby("article_id")
            .size()
            .reindex(article_ids, fill_value=0)
        )
        competitions = (
            ents[ents["entity_label"] == "COMPETITION"]
            .groupby("article_id")
            .size()
            .reindex(article_ids, fill_value=0)
        )
        return pd.DataFrame(
            {
                "article_id": article_ids,
                "gender": gender,
                "total_entities": total.values,
                "players": players.values,
                "clubs": clubs.values,
                "competitions": competitions.values,
            }
        )

    men_counts = build_counts(men_ents, men_articles, "Men")
    women_counts = build_counts(women_ents, women_articles, "Women")
    counts_df = pd.concat([men_counts, women_counts], ignore_index=True)
    counts_df.to_csv(plots_dir / "ner_entities_per_article_summary.csv", index=False)
    logger.info(f"  Saved: {plots_dir}/ner_entities_per_article_summary.csv")

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    for gender, color in [("Men", "#2e86de"), ("Women", "#e74c3c")]:
        vals = counts_df[counts_df["gender"] == gender]["total_entities"]
        axes[0].hist(vals, bins=35, alpha=0.45, label=gender, color=color, density=True)
    axes[0].set_xlabel("Entity mentions per article")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Distribution of Total Entity Mentions per Article")
    axes[0].legend()

    melted = counts_df.melt(
        id_vars=["article_id", "gender"],
        value_vars=["players", "clubs", "competitions"],
        var_name="category",
        value_name="mentions",
    )
    sns.boxplot(
        data=melted,
        x="category",
        y="mentions",
        hue="gender",
        ax=axes[1],
        showfliers=False,
    )
    axes[1].set_title("Per-Article Mentions by Category")
    axes[1].set_ylabel("Mentions per article")
    axes[1].set_xlabel("")
    axes[1].legend(title="")

    plt.tight_layout()
    plt.savefig(plots_dir / "ner_entities_per_article_distribution.png", bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved: {plots_dir}/ner_entities_per_article_distribution.png")


def ner_label_mapping_heatmap(men_ents, women_ents, plots_dir):
    """Heatmap: how raw spaCy labels map into your football-domain labels."""
    logger.info("\n=== NER Label Mapping Heatmap ===")

    fig, axes = plt.subplots(1, 2, figsize=(17, 8))

    for ax, (gender, ents) in zip(axes, [("Men", men_ents), ("Women", women_ents)]):
        top_spacy = ents["spacy_label"].value_counts().head(10).index
        ctab = pd.crosstab(ents["spacy_label"], ents["entity_label"], normalize="index") * 100
        ctab = ctab.loc[top_spacy]
        ctab = ctab[ctab.sum(axis=0).sort_values(ascending=False).index]

        sns.heatmap(
            ctab,
            ax=ax,
            cmap="Blues",
            annot=True,
            fmt=".1f",
            cbar_kws={"shrink": 0.8},
            linewidths=0.3,
        )
        ax.set_title(f"{gender}'s: spaCy label -> domain label (%)")
        ax.set_xlabel("Mapped entity_label")
        ax.set_ylabel("spaCy label")

    plt.tight_layout()
    plt.savefig(plots_dir / "ner_spacy_to_domain_heatmap.png", bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved: {plots_dir}/ner_spacy_to_domain_heatmap.png")


def ner_span_length_diagnostics(men_ents, women_ents, plots_dir):
    """Analyze entity span length to surface noisy extractions."""
    logger.info("\n=== NER Span-Length Diagnostics ===")

    keep_labels = {"PLAYER", "CLUB", "COMPETITION", "LOCATION", "ORG_OTHER"}

    def prep(df: pd.DataFrame, gender: str) -> pd.DataFrame:
        d = df.copy()
        d["gender"] = gender
        d["char_length"] = (d["end_char"] - d["start_char"]).clip(lower=1)
        d["token_length"] = d["entity_text"].fillna("").astype(str).str.split().str.len().clip(lower=1)
        d = d[d["entity_label"].isin(keep_labels)]
        d = d[d["token_length"] <= 12]  # trim extreme outliers for readability
        return d

    combined = pd.concat([prep(men_ents, "Men"), prep(women_ents, "Women")], ignore_index=True)
    summary = (
        combined.groupby(["gender", "entity_label"])
        .agg(
            n=("entity_text", "size"),
            mean_token_length=("token_length", "mean"),
            median_token_length=("token_length", "median"),
            p95_token_length=("token_length", lambda x: np.percentile(x, 95)),
        )
        .reset_index()
        .sort_values(["gender", "entity_label"])
    )
    summary.to_csv(plots_dir / "ner_span_length_summary.csv", index=False)
    logger.info(f"  Saved: {plots_dir}/ner_span_length_summary.csv")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    sns.boxplot(
        data=combined,
        x="entity_label",
        y="token_length",
        hue="gender",
        ax=axes[0],
        showfliers=False,
    )
    axes[0].set_title("Entity Token-Length by Label")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Token length")
    axes[0].tick_params(axis="x", rotation=30)
    axes[0].legend(title="")

    sns.violinplot(
        data=combined,
        x="gender",
        y="char_length",
        hue="gender",
        ax=axes[1],
        cut=0,
        inner="quartile",
    )
    axes[1].set_title("Character Span Length Distribution")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Character span length")
    if axes[1].legend_:
        axes[1].legend_.remove()

    plt.tight_layout()
    plt.savefig(plots_dir / "ner_span_length_diagnostics.png", bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved: {plots_dir}/ner_span_length_diagnostics.png")


def ner_topk_concentration_curves(men_freq, women_freq, plots_dir):
    """Cumulative share captured by top-K entities for key categories."""
    logger.info("\n=== NER Top-K Concentration Curves ===")

    focus_labels = ["PLAYER", "CLUB", "COMPETITION"]
    max_k = 50
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

    for ax, (gender, freq_df) in zip(axes, [("Men", men_freq), ("Women", women_freq)]):
        for label in focus_labels:
            subset = freq_df[freq_df["entity_label"] == label].sort_values("frequency", ascending=False)
            if subset.empty:
                continue
            vals = subset["frequency"].values
            cum = np.cumsum(vals)
            cum_share = cum / cum[-1] * 100
            k = min(max_k, len(cum_share))
            ax.plot(np.arange(1, k + 1), cum_share[:k], label=label)

        ax.set_title(f"{gender}'s: cumulative share by top-K entities")
        ax.set_xlabel("Top-K entities")
        ax.set_ylabel("Cumulative mention share (%)")
        ax.set_ylim(0, 100)
        ax.grid(alpha=0.3)
        ax.legend()

    plt.tight_layout()
    plt.savefig(plots_dir / "ner_topk_concentration_curves.png", bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved: {plots_dir}/ner_topk_concentration_curves.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Football NER Analysis")
    parser.add_argument("--model", default="lg", help="Model key: sm, lg, trf")
    parser.add_argument("--input-dir", default=None, help="Directory containing men_articles.csv / women_articles.csv")
    parser.add_argument("--ner-dir", default=None, help="Directory containing NER outputs (model subfolders)")
    parser.add_argument("--output-dir", default=None, help="Directory to save plots/analysis")
    args = parser.parse_args()

    # Resolve paths
    data_dir = Path(args.input_dir) if args.input_dir else PROCESSED_DIR_DEFAULT
    ner_dir = Path(args.ner_dir) if args.ner_dir else NER_OUTPUT_DEFAULT
    plots_dir = Path(args.output_dir) if args.output_dir else PLOTS_DIR_DEFAULT
    
    plots_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading NER data for model: {args.model}")
    men_ents, women_ents, men_freq, women_freq, men_articles, women_articles = load_ner_data(data_dir, ner_dir, args.model)

    logger.info(f"Men: {len(men_ents)} entities, Women: {len(women_ents)} entities")

    # Run all analyses
    entity_summary_tables(men_ents, women_ents, men_freq, women_freq, plots_dir)
    h1_player_prominence(men_ents, women_ents, plots_dir)
    h2_club_cooccurrence(men_ents, women_ents, plots_dir)
    h9_individual_vs_team(men_ents, women_ents, men_articles, women_articles, plots_dir)
    h10_name_formality(men_ents, women_ents, plots_dir)
    h11_h12_descriptor_context(men_ents, women_ents, plots_dir)
    h14_meta_discourse(men_ents, women_ents, plots_dir)
    h15_credit_assignment(men_ents, women_ents, plots_dir)
    ner_coverage_diagnostics(men_ents, women_ents, men_articles, women_articles, plots_dir)
    ner_entities_per_article_distribution(men_ents, women_ents, men_articles, women_articles, plots_dir)
    ner_label_mapping_heatmap(men_ents, women_ents, plots_dir)
    ner_span_length_diagnostics(men_ents, women_ents, plots_dir)
    ner_topk_concentration_curves(men_freq, women_freq, plots_dir)

    logger.info(f"\n{'='*60}")
    logger.info("ALL ANALYSES COMPLETE")
    logger.info(f"Plots saved to: {plots_dir}/")
    logger.info(f"Data saved to: {ner_dir}/")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
