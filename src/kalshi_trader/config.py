from pydantic_settings import BaseSettings


class AppConfig(BaseSettings):
    # Kalshi API
    kalshi_api_key_id: str = ""
    kalshi_private_key_path: str = ""
    kalshi_base_url: str = "https://api.elections.kalshi.com/trade-api/v2"
    kalshi_demo: bool = True

    # Database
    db_path: str = "data/kalshi.db"

    # Ingestion
    snapshot_interval_seconds: int = 30
    tracked_series: list[str] = ["KXNBAGAME", "KXNBASPREAD", "KXNBATOTAL", "KXNBATEAMTOTAL"]
    gdelt_poll_interval_seconds: int = 900  # 15 minutes

    # Google BigQuery (for GDELT data without rate limits)
    gcp_project: str = ""
    gcp_credentials_path: str = ""  # path to service-account JSON key

    # Risk limits
    max_position_per_market: int = 50
    max_total_exposure_cents: int = 5000  # $50
    max_drawdown_fraction: float = 0.20
    max_open_orders: int = 20

    # Strategy
    min_edge: float = 0.05  # 5% minimum probability edge
    default_order_size: int = 5

    model_config = {"env_file": ".env", "env_prefix": ""}
