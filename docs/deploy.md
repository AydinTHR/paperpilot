# Deploying PaperPilot 24/7

The experiment runs on any small Linux box (1 vCPU / 1 GB is plenty) via
Docker Compose. sqlite journals live on a bind-mounted `./data`, so upgrades
are `git pull && docker compose up -d --build` and state survives.

## 1. Provision

- Ubuntu 24.04, smallest tier (Hetzner/DigitalOcean/Lightsail), region near
  US-East. Add your SSH public key at creation.
- Harden the box (once):

```bash
apt update && apt -y upgrade
apt -y install docker.io docker-compose-v2 git
ufw allow OpenSSH && ufw --force enable   # nothing else is exposed
```

## 2. Install

```bash
git clone https://github.com/AydinTHR/paperpilot /opt/paperpilot
cd /opt/paperpilot
cp .env.example .env && chmod 600 .env
# Fill .env: Alpaca keys (all 3 paper accounts), OpenRouter key, Telegram,
# UNIVERSE, DEFAULT_INTERVAL, USE_TRADE_STREAM=true, MARKET_HOURS_ONLY=true.
```

Migrating from another machine? Copy its `data/experiments/*.db` into
`/opt/paperpilot/data/experiments/` BEFORE first start, so equity history and
report baselines carry over.

## 3. Run

```bash
docker compose up -d --build
docker compose logs -f experiment     # watch the first tick
```

The `experiment` service restarts automatically on crash and on reboot
(`restart: unless-stopped`). The `dashboard` service listens on loopback only:

```bash
ssh -L 8501:localhost:8501 <user>@<server-ip>   # then open http://localhost:8501
```

## 4. Weekly report cron

```bash
crontab -e
# Sundays 20:00 UTC: send the per-arm comparison to Telegram/Discord.
0 20 * * 0 cd /opt/paperpilot && /usr/bin/docker compose exec -T experiment python scripts/weekly_report.py >> logs/weekly_report.log 2>&1
```

## 5. Operate

| Task | Command (in /opt/paperpilot) |
|---|---|
| Status | `docker compose ps` |
| Logs | `docker compose logs -f --tail 100 experiment` |
| Comparison report | `docker compose exec -T experiment python scripts/run_experiment.py --report` |
| Reset a halt | `docker compose exec -T experiment python scripts/run_live.py --reset-halt` |
| Upgrade | `git pull && docker compose up -d --build` |
| Stop everything | `docker compose down` |

## 6. Backup & restore

The only state is `data/` (journals) and `.env` (secrets). Back both up:

```bash
tar czf paperpilot-backup-$(date +%F).tar.gz data .env
```

Restore = fresh install (steps 1-2), untar over `/opt/paperpilot`, `up -d`.
