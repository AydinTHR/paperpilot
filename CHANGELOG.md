# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0](https://github.com/AydinTHR/paperpilot/compare/v0.2.0...v0.3.0) (2026-07-12)


### Added

* **ops:** week-2 hardening + 24/7 Docker deployment ([#30](https://github.com/AydinTHR/paperpilot/issues/30)) ([c4b2c7f](https://github.com/AydinTHR/paperpilot/commit/c4b2c7fcfb7757893e73ab75cec1d24132f9e364))

## [0.2.0](https://github.com/AydinTHR/paperpilot/compare/v0.1.0...v0.2.0) (2026-07-04)


### Added

* **backtest:** walk-forward validation and FIFO realized-P&L trades table ([#20](https://github.com/AydinTHR/paperpilot/issues/20)) ([2ca3a64](https://github.com/AydinTHR/paperpilot/commit/2ca3a64d50684bcd4c11ef10d44a6f71acda83c8))
* **data:** add Alpaca-backed market data provider and market-hours awareness ([#16](https://github.com/AydinTHR/paperpilot/issues/16)) ([6c2ce8b](https://github.com/AydinTHR/paperpilot/commit/6c2ce8be3a651b0f03c5783e6781186d3503677c))
* **execution:** broker-held protective stops and order fill reconciliation ([#17](https://github.com/AydinTHR/paperpilot/issues/17)) ([a3a7179](https://github.com/AydinTHR/paperpilot/commit/a3a7179eca8b9cd4a1a9d5a09aa050483c53ac5d))
* **experiments:** LLM-vs-classical paper-trading comparison harness and runner ([#21](https://github.com/AydinTHR/paperpilot/issues/21)) ([0c1c188](https://github.com/AydinTHR/paperpilot/commit/0c1c1881e1e947379f98e233e61f283d9c42a386))
* **llm:** OpenRouter provider for the LLM signal layer ([#23](https://github.com/AydinTHR/paperpilot/issues/23)) ([a73cd48](https://github.com/AydinTHR/paperpilot/commit/a73cd485c265b0022047f21f42841e3b87b51151))
* **monitoring:** add Telegram/Discord alerting and Streamlit dashboard ([#19](https://github.com/AydinTHR/paperpilot/issues/19)) ([17a141d](https://github.com/AydinTHR/paperpilot/commit/17a141d5306c04a6fa843ff20a825b899fd05acd))
* **risk:** persist halt state across restarts and add --reset-halt CLI ([#18](https://github.com/AydinTHR/paperpilot/issues/18)) ([807aa06](https://github.com/AydinTHR/paperpilot/commit/807aa063f4e64915e7b5beb1ce78659314266267))


### Fixed

* **tests:** .env isolation + experiment --fresh flag ([#22](https://github.com/AydinTHR/paperpilot/issues/22)) ([36d7648](https://github.com/AydinTHR/paperpilot/commit/36d764812f7b06b587fba1bc0f61eb2ad232824b))

## [Unreleased]

### Added

- Initial project scaffold.
