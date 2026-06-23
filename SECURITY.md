# Security Policy

## Reporting a vulnerability

Please do not open a public issue for security problems.

Report vulnerabilities privately to aidinthr82@gmail.com, or use GitHub's private
vulnerability reporting on this repository (Security tab, "Report a vulnerability").

Include a description, steps to reproduce, and the impact as you understand it.
You can expect an acknowledgement within a few days.

## Supported versions

This project is pre-1.0. Only the latest release receives fixes.

## Secrets and trading safety

- API keys live in `.env`, which is gitignored. Never commit real keys. Use Alpaca
  **paper** keys for everything in this repo.
- Real-money trading is gated off by default. It requires both `PAPER=false` and
  `ALLOW_LIVE_TRADING=true`, and prints a warning banner on startup. Treat any change
  that touches that gate as security-sensitive.
- If you believe a key has been exposed in the history or a log, rotate it in the Alpaca
  dashboard immediately, then report it through the channel above.
