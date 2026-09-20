# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GolemQ is a quantitative trading framework for the Chinese A-share stock market. It provides market data ingestion, feature engineering, backtesting, live trading via XTQuant, and monitoring/scheduling — backed by MongoDB and RabbitMQ.

Cross-platform: supports Windows x64 and major Linux distributions.

Requires Python >= 3.12. Install in editable mode:

```bash
conda create -n GolemQ python=3.12
conda activate GolemQ
pip install -e .
```

### Windows Conda Preamble

On Windows, activate the conda environment before running commands:

```powershell
C:\ProgramData\miniconda3\shell\condabin\conda-hook.ps1 ; conda activate C:\ProgramData\miniconda3 ; cd "y:/projects/GolemQ" ; conda activate GolemQ
```

## Commands

```bash
# Run all tests
python GolemQ/test_cases/run_tests.py

# Run a specific test module
python -m unittest GolemQ.test_cases.test_messenger -v

# Run a single test method
python -m unittest GolemQ.test_cases.test_messenger.TestDingtalkConfig.test_check_config_success -v

# CLI - configuration
python -m GolemQ.cli --setup              # Initialize MongoDB, DingTalk, Server酱
python -m GolemQ.cli --mongodb-init       # MongoDB config only
python -m GolemQ.cli --dingtalk-init      # DingTalk config only
python -m GolemQ.cli --xtquant-init       # XTQuant config only

# CLI - data operations
python -m GolemQ.cli --purge-l1           # Purge MongoDB historical collections
python -m GolemQ.cli --xtquant-sync       # One-shot XTQuant positions sync to MongoDB
python -m GolemQ.cli --xtquant-sync-daemon # Daemon mode: sync during trading hours

# CLI - monitoring
python -m GolemQ.cli --heartbeat-watchdog      # View heartbeat status
python -m GolemQ.cli --stop-heartbeat-monitor  # Stop all monitoring

# CLI - subscriptions & watchlists
python -m GolemQ.cli --sub l1_tencent           # Run L1 Tencent data subscription
python -m GolemQ.cli --eneloop-add --symbols "000001,000002"
python -m GolemQ.cli --eneloop-remove --symbols "000001,000002"
python -m GolemQ.cli --eneloop-list
```

## Architecture

### Infrastructure

- **MongoDB** — primary database for all persistence (market data, operational state, configuration)
- **RabbitMQ** — cache and message queue layer

### Package Structure (top-level `GolemQ/`)

| Module | Purpose |
|--------|---------|
| `markets/` | Market abstraction layer. `base_market.py` defines the `BaseMarket` ABC. `StockCN/` is the A-share implementation (singleton via `__new__`). `StockHK/` is a stub. Markets self-register into the global `GQMARKETS` dict. |
| `gateway/xtquant/` | XTQuant trading gateway. `trader.py` wraps `XtQuantTrader`; `xtquant_tools.py` provides position export, available volume queries; `config.py` manages XTQuant credentials in `~/.GolemQ/config.ini`. |
| `services/` | Data operations layer. `features.py` saves/fetches feature metadata to/from MongoDB. `align.py` provides checkpoint logging, kline alignment, and missing-data detection. `iwencai.py` interfaces with iWenCai queries. `persistence/` (package) provides general-purpose MongoDB CRUD helpers. |
| `analysis/` | Analysis modules. `timeseries.py` provides multi-frequency resampling and timeline utilities. |
| `pipeline/` | Stock processing pipeline. `base.py` has abstract pipeline classes with joblib parallelization. Sub-modules: `mainstream_benchmark.py`, `poolcoef_benchmark.py`, `compact_benchmark.py`. |
| `agents/` | External communication. `messenger.py` handles DingTalk bot (alibabacloud_dingtalk SDK) and Server酱 push notifications. |
| `supervisor/` | Operational monitoring. `heartbeat.py` provides `HeartbeatMonitor` (MongoDB-backed, per-module heartbeat with timeout detection) and `HeartbeatModule` (per-instance mutex/checkin). `scheduler.py` runs `schedule`-based `XtquantSyncScheduler` during A-share trading hours (9:30-11:30, 13:00-15:00 Beijing time). `function_checkin.py` provides rate-limiting for alert functions. `messenger.py` wraps alert dispatch. |
| `cli/` | CLI entry point (`__main__.py` → `main()`). `tools.py` auto-discovers and registers market modules. `watchdog_manager.py` manages symbol watchlists. |
| `core/` | `settings.py` — `GQ_Setting` wraps `~/.GolemQ/config.ini` with MongoDB fallback; also ties into QUANTAXIS settings. `constants.py` — `AKA` (field aliases), `FIELD`, `MARKET_TYPE`, `STATE` constants. `mongo.py` — MongoDB client helpers. `preprocessing.py` — pandas-to-JSON converters and data masking. |

### Key Design Patterns

- **Market registry**: `GQMARKETS` (dict in `GolemQ.__init__`) holds market instances. `cli/tools.py` auto-discovers and registers them. `GQSUBSCRIBER` maps subscription keys (e.g., `l1_tencent`) to subscriber functions.
- **StockCN singleton**: `StockCN.__new__` enforces a single instance. Auto-instantiated on module import and registered into `GQMARKETS`.
- **Configuration**: `GQ_Setting` reads/writes `~/.GolemQ/config.ini`. Sections `DINGTALK`, `SERVERCHAN`, `XTQUANT` are stored in the INI file; other sections fall back to MongoDB. QUANTAXIS settings are used in parallel (`QASETTING`, `DATABASE_QA`).
- **Dual database**: GolemQ stores operational data under `DATABASE.golemq.*` collections; QUANTAXIS stores market data under `DATABASE.quantaxis.*`.
- **Heartbeat/mutex**: `HeartbeatModule.mutex()` checks for conflicting running instances in MongoDB before starting, using a timeout-based lock. `HeartbeatMonitor` runs a daemon thread that detects stale modules and fires alerts.
- **Data flow**: Free data sources (Tencent, Sina, EastMoney via `easyquotation/`) → `services/` feature extraction → MongoDB → `analysis/` / `pipeline/` consumption. XTQuant gateway provides live position/order data.

## Coding Conventions

- **Database operations**: ALL database operations (MongoDB CRUD, queries, aggregations) MUST live in the `services/` layer. No other module may access the database directly — they call `services/` functions instead.
- **Naming**: Use `snake_case` for all function, variable, and file names.
- **CLI functions**: CLI command handlers in `cli/` do ONLY parameter validation and then delegate to `pipeline/` functions. No business logic in the CLI layer.
- **File size**: Each `services/` file must not exceed 300 lines. Split larger files into focused sub-modules.

### External Dependencies

- **QUANTAXIS**: Core market data framework. Provides `QA_util_*` date utilities, market type constants, database client.
- **xtquant**: Commercial trading SDK (QMT/miniQMT). Only available on Windows with the QMT client installed.
- **MongoDB**: All data persistence. Connection string defaults to `mongodb://localhost:27017`.
- **RabbitMQ**: Cache and message queue. Used for inter-service communication and data buffering.
- **alibabacloud_dingtalk**: DingTalk robot SDK for push notifications.

### Configuration File

`~/.GolemQ/config.ini` — sections: `MONGODB` (uri), `DINGTALK` (appkey, appsecret, robot_code, user_id_list), `SERVERCHAN` (sendkey), `XTQUANT` (account, min_path).
