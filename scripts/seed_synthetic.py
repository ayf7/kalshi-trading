#!/usr/bin/env python3
"""
Seed the database with realistic synthetic data so you can test
the full pipeline: train a model -> run a backtest -> see results.

Usage:
    python scripts/seed_synthetic.py
    python scripts/train_model.py --model logistic
    python scripts/run_backtest.py --start 2025-02-01 --end 2025-03-01 --model logistic
"""

import random
import math

from kalshi_trader.config import AppConfig
from kalshi_trader.data import db
from kalshi_trader.data.models import Market, MarketSnapshot, NewsArticle


def main():
    random.seed(42)
    config = AppConfig()
    conn = db.init_db(config.db_path)

    print("Seeding synthetic data...")

    # Create 40 settled markets (20 YES outcomes, 20 NO outcomes)
    # spread across Jan-Feb 2025, simulating NBA games
    teams = [
        ("MIL", "BOS"), ("LAL", "GSW"), ("PHX", "DEN"), ("MIA", "NYK"),
        ("DAL", "LAC"), ("PHI", "CLE"), ("MIN", "OKC"), ("SAC", "HOU"),
        ("ATL", "CHI"), ("IND", "TOR"),
    ]

    markets_created = 0
    snapshots_created = 0
    articles_created = 0

    for game_day in range(40):
        team_a, team_b = teams[game_day % len(teams)]
        day = game_day + 1
        month = 1 if day <= 31 else 2
        day_of_month = day if day <= 31 else day - 31

        ticker = f"KXNBA-{team_a}-{team_b}-2025{month:02d}{day_of_month:02d}"
        event_ticker = f"KXNBA-{team_a}-{team_b}"

        # Market opens 24h before close
        close_ts = 1735689600 + game_day * 86400  # start from Jan 1 2025
        open_ts = close_ts - 86400

        # Decide the true outcome -- slightly favor team_a
        true_prob = 0.4 + random.random() * 0.3  # between 0.4 and 0.7
        result = "yes" if random.random() < true_prob else "no"

        market = Market(
            ticker=ticker,
            event_ticker=event_ticker,
            title=f"Will the {team_a} beat the {team_b}?",
            status="settled",
            series_ticker="KXNBA",
            category="sports",
            result=result,
            open_ts=open_ts,
            close_ts=close_ts,
            settled_ts=close_ts + 3600,
        )
        db.upsert_market(conn, market)
        markets_created += 1

        # Generate ~200 snapshots per market (one every ~7 minutes over 24h)
        # Price starts near 50 and drifts toward the outcome
        price = 50.0
        for snap_i in range(200):
            ts = open_ts + int(snap_i * (86400 / 200))

            # Price random walks with drift toward outcome
            time_frac = snap_i / 200.0  # 0 to 1
            target = 85 if result == "yes" else 15
            drift = (target - price) * 0.005 * (1 + time_frac)
            noise = random.gauss(0, 1.5)
            price = max(5, min(95, price + drift + noise))

            mid = int(round(price))
            spread = random.choice([2, 4, 6, 8])
            yes_bid = max(1, mid - spread // 2)
            yes_ask = min(99, mid + spread // 2)

            volume_base = 50 + snap_i * 2
            volume_noise = random.randint(-20, 20)

            snap = MarketSnapshot(
                ticker=ticker,
                ts=ts,
                yes_bid=yes_bid,
                yes_ask=yes_ask,
                last_price=mid,
                volume=max(0, volume_base + volume_noise),
                open_interest=300 + snap_i,
                yes_bid_size=random.randint(20, 200),
                yes_ask_size=random.randint(20, 200),
            )
            db.insert_snapshot(conn, snap)
            snapshots_created += 1

        # Generate 5-15 news articles per market
        n_articles = random.randint(5, 15)
        for art_i in range(n_articles):
            art_ts = open_ts + random.randint(0, 86400)
            # Tone correlates weakly with outcome
            if result == "yes":
                tone = random.gauss(2.0, 4.0)
            else:
                tone = random.gauss(-1.0, 4.0)

            headlines = [
                f"{team_a} looking strong ahead of matchup vs {team_b}",
                f"{team_b} injury report raises concerns",
                f"Preview: {team_a} vs {team_b} game analysis",
                f"Experts weigh in on {team_a}-{team_b} odds",
                f"{team_a} coach confident about upcoming game",
                f"{team_b} star player questionable for game",
                f"Betting lines shift for {team_a} vs {team_b}",
            ]

            article = NewsArticle(
                url=f"https://example.com/sports/{ticker}-article-{art_i}",
                title=random.choice(headlines),
                seen_date=f"2025-{month:02d}-{day_of_month:02d}T{random.randint(8,23):02d}:00:00",
                domain=random.choice(["espn.com", "reuters.com", "bleacherreport.com", "nba.com"]),
                language="en",
                source_country="US",
                tone=round(tone, 1),
            )
            article_id = db.insert_news_article(conn, article)
            if article_id is not None:
                db.link_news_to_market(conn, article_id, ticker)
                articles_created += 1

    # Also create 5 currently "open" markets (for testing ingestion display)
    for i in range(5):
        team_a, team_b = teams[i]
        ticker = f"KXNBA-{team_a}-{team_b}-20250315"
        close_ts = 1742169600 + i * 86400
        market = Market(
            ticker=ticker,
            event_ticker=f"KXNBA-{team_a}-{team_b}",
            title=f"Will the {team_a} beat the {team_b}?",
            status="open",
            series_ticker="KXNBA",
            category="sports",
            open_ts=close_ts - 86400,
            close_ts=close_ts,
        )
        db.upsert_market(conn, market)
        markets_created += 1

    conn.close()

    print(f"Done!")
    print(f"  {markets_created} markets (40 settled + 5 open)")
    print(f"  {snapshots_created} price snapshots (~200 per market)")
    print(f"  {articles_created} news articles")
    print()
    print("Next steps:")
    print("  1. Train a model:    python scripts/train_model.py --model logistic -v")
    print("  2. Train XGBoost:    python scripts/train_model.py --model xgboost -v")
    print("  3. Run a backtest:   python scripts/run_backtest.py --start 2025-01-15 --end 2025-02-10 --model logistic -v")


if __name__ == "__main__":
    main()
