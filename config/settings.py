"""Application configuration via pydantic-settings.

All settings load from environment variables (and an optional `.env` file).
Secrets are held as ``SecretStr`` so they never leak into logs or reprs.

Safety contract enforced here:
  * ``PAPER`` defaults to ``True`` -- simulated trading is the only default path.
  * Real-money trading (``PAPER=false``) is *refused* unless the operator also
    sets ``ALLOW_LIVE_TRADING=true``. This makes live trading impossible without
    a deliberate, manual config change.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Always printed on startup.
DISCLAIMER_BANNER = r"""
============================================================================
  PaperPilot  --  Educational project. SIMULATED (paper) trading only.
  Not financial advice. Past or simulated performance does not predict
  future results. Use entirely at your own risk.
============================================================================
""".strip("\n")

# Printed ONLY when real-money trading has been deliberately enabled.
LIVE_TRADING_WARNING = r"""
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!                                                                        !!
!!   ##  LIVE REAL-MONEY TRADING IS ENABLED  ##                           !!
!!                                                                        !!
!!   This is NOT a simulation. Orders placed here can lose REAL money.    !!
!!   PaperPilot is an educational project and is NOT financial advice.    !!
!!   You have explicitly set PAPER=false and ALLOW_LIVE_TRADING=true.     !!
!!                                                                        !!
!!   Press Ctrl-C now if this was not intentional.                        !!
!!                                                                        !!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
""".strip("\n")


class Settings(BaseSettings):
    """Strongly-typed, validated application settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Alpaca credentials (never logged) ---
    alpaca_api_key: SecretStr = Field(default=SecretStr(""))
    alpaca_secret_key: SecretStr = Field(default=SecretStr(""))

    # --- Trading mode ---
    paper: bool = Field(
        default=True,
        description="True = simulated paper trading (default, safe).",
    )
    allow_live_trading: bool = Field(
        default=False,
        description="Deliberate gate that must be True for PAPER=false to be allowed.",
    )

    # --- Risk limits (fractions of equity, 0..1) ---
    max_position_pct: float = Field(
        default=0.10,
        gt=0,
        le=1,
        description="Max fraction of equity allowed in a single position.",
    )
    max_daily_loss_pct: float = Field(
        default=0.03,
        gt=0,
        le=1,
        description="Daily-loss kill switch threshold as a fraction of equity.",
    )
    max_drawdown_pct: float = Field(
        default=0.20,
        gt=0,
        le=1,
        description="Max-drawdown halt threshold: stop trading once equity falls "
        "this fraction below its peak.",
    )
    stop_loss_pct: float = Field(
        default=0.05,
        ge=0,
        lt=1,
        description="Per-trade stop-loss distance below entry (0 disables stops).",
    )

    # --- Logging ---
    log_level: str = Field(default="INFO")
    log_dir: str = Field(default="logs")

    # --- Market data ---
    default_interval: str = Field(
        default="1d",
        description="Default bar interval for data fetches ('1d' or '1h').",
    )
    data_cache_dir: str = Field(
        default="data/cache",
        description="Directory for the local parquet OHLCV cache.",
    )
    data_provider: str = Field(
        default="auto",
        description="Market data source: 'alpaca', 'yfinance', or 'auto' "
        "(alpaca when Alpaca credentials are present, else yfinance).",
    )
    alpaca_data_feed: str = Field(
        default="iex",
        description="Alpaca market data feed: 'iex' (free Basic plan) or 'sip' (paid).",
    )
    market_hours_only: bool = Field(
        default=True,
        description="Skip scheduled loop ticks while the NYSE market is closed.",
    )

    # --- Live loop & trade journal ---
    default_strategy: str = Field(
        default="sma",
        description="Strategy the live loop runs by default ('sma', 'rsi', or 'llm').",
    )
    loop_interval_minutes: int = Field(
        default=60,
        gt=0,
        description="Minutes between scheduled live-loop iterations.",
    )
    db_url: str = Field(
        default="sqlite:///data/paperpilot.db",
        description="SQLAlchemy URL for the trade journal (gitignored sqlite file).",
    )

    # --- LLM signal layer (Phase 6, optional) ---
    llm_provider: str = Field(
        default="anthropic",
        description="LLM backend for the optional signal layer "
        "('anthropic'; other providers reserved).",
    )
    llm_model: str = Field(
        default="claude-3-5-haiku-latest",
        description="Model name passed to the LLM provider (override per provider).",
    )
    llm_max_tokens: int = Field(default=512, gt=0, description="Max tokens for the LLM response.")
    llm_temperature: float = Field(
        default=0.0,
        ge=0,
        le=2,
        description="LLM sampling temperature (low = more consistent).",
    )
    llm_timeout_seconds: float = Field(
        default=30.0, gt=0, description="Per-call timeout for the LLM client (seconds)."
    )
    anthropic_api_key: SecretStr = Field(
        default=SecretStr(""),
        description="Anthropic API key for the LLM layer (env ANTHROPIC_API_KEY). "
        "Blank disables the LLM strategy (it then HOLDs).",
    )

    @field_validator("log_level")
    @classmethod
    def _normalize_log_level(cls, v: str) -> str:
        level = v.strip().upper()
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if level not in allowed:
            raise ValueError(f"LOG_LEVEL must be one of {sorted(allowed)}, got {v!r}")
        return level

    @field_validator("default_interval")
    @classmethod
    def _normalize_interval(cls, v: str) -> str:
        interval = v.strip().lower()
        allowed = {"1d", "1h"}
        if interval not in allowed:
            raise ValueError(f"DEFAULT_INTERVAL must be one of {sorted(allowed)}, got {v!r}")
        return interval

    @field_validator("data_provider")
    @classmethod
    def _normalize_data_provider(cls, v: str) -> str:
        provider = v.strip().lower()
        allowed = {"auto", "alpaca", "yfinance"}
        if provider not in allowed:
            raise ValueError(f"DATA_PROVIDER must be one of {sorted(allowed)}, got {v!r}")
        return provider

    @field_validator("alpaca_data_feed")
    @classmethod
    def _normalize_data_feed(cls, v: str) -> str:
        feed = v.strip().lower()
        allowed = {"iex", "sip"}
        if feed not in allowed:
            raise ValueError(f"ALPACA_DATA_FEED must be one of {sorted(allowed)}, got {v!r}")
        return feed

    @field_validator("default_strategy")
    @classmethod
    def _normalize_strategy(cls, v: str) -> str:
        strategy = v.strip().lower()
        allowed = {"sma", "rsi", "llm"}
        if strategy not in allowed:
            raise ValueError(f"DEFAULT_STRATEGY must be one of {sorted(allowed)}, got {v!r}")
        return strategy

    @field_validator("llm_provider")
    @classmethod
    def _normalize_llm_provider(cls, v: str) -> str:
        provider = v.strip().lower()
        allowed = {"anthropic", "openai"}
        if provider not in allowed:
            raise ValueError(f"LLM_PROVIDER must be one of {sorted(allowed)}, got {v!r}")
        return provider

    @model_validator(mode="after")
    def _enforce_live_trading_gate(self) -> Settings:
        """Refuse to construct settings that would silently enable live trading."""
        if not self.paper and not self.allow_live_trading:
            raise ValueError(
                "Live trading requested (PAPER=false) but ALLOW_LIVE_TRADING is not "
                "true. Refusing to start. To deliberately enable real-money trading, "
                "set ALLOW_LIVE_TRADING=true -- otherwise leave PAPER=true."
            )
        return self

    @property
    def is_live(self) -> bool:
        """True only when real-money trading is both requested and permitted."""
        return (not self.paper) and self.allow_live_trading

    @property
    def has_credentials(self) -> bool:
        return bool(self.alpaca_api_key.get_secret_value()) and bool(
            self.alpaca_secret_key.get_secret_value()
        )

    @property
    def has_llm_key(self) -> bool:
        """True when an API key for the configured LLM provider is present."""
        return bool(self.anthropic_api_key.get_secret_value())

    @property
    def resolved_data_provider(self) -> str:
        """The effective data source: 'auto' resolves by credential presence."""
        if self.data_provider != "auto":
            return self.data_provider
        return "alpaca" if self.has_credentials else "yfinance"

    def safe_summary(self) -> dict[str, object]:
        """A secret-free view of the config, suitable for logging."""
        return {
            "mode": "LIVE" if self.is_live else "PAPER",
            "paper": self.paper,
            "allow_live_trading": self.allow_live_trading,
            "has_credentials": self.has_credentials,
            "max_position_pct": self.max_position_pct,
            "max_daily_loss_pct": self.max_daily_loss_pct,
            "max_drawdown_pct": self.max_drawdown_pct,
            "stop_loss_pct": self.stop_loss_pct,
            "log_level": self.log_level,
            "default_interval": self.default_interval,
            "data_provider": self.resolved_data_provider,
            "alpaca_data_feed": self.alpaca_data_feed,
            "market_hours_only": self.market_hours_only,
            "default_strategy": self.default_strategy,
            "loop_interval_minutes": self.loop_interval_minutes,
            "llm_provider": self.llm_provider,
            "llm_model": self.llm_model,
            "has_llm_key": self.has_llm_key,
        }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached, validated ``Settings`` instance.

    Cached so the environment is parsed once per process. Tests can call
    ``get_settings.cache_clear()`` to force a reload.
    """
    return Settings()
