# Deploying HMM Trader

## Quick Start

### 1. Create a VPS

Any provider works. Recommended:
- **Hetzner** - $4.50/mo (CX22: 2 vCPU, 4GB RAM) - best value
- **DigitalOcean** - $6/mo (Basic: 1 vCPU, 2GB RAM)
- **Linode** - $5/mo (Nanode: 1 vCPU, 2GB RAM)

Choose **Ubuntu 22.04** and the cheapest plan with at least **2GB RAM** (IB Gateway is Java and needs it).

### 2. Run the Setup Script

```bash
ssh root@your-server-ip

# Download and run
curl -sL https://raw.githubusercontent.com/dwagner003/hmm-trader/master/deploy/setup-server.sh | bash
```

Or manually:
```bash
git clone https://github.com/dwagner003/hmm-trader.git /tmp/hmm-setup
bash /tmp/hmm-setup/deploy/setup-server.sh
```

### 3. Configure IBKR Credentials

```bash
nano /opt/ibc/config.ini
```

Set:
- `IbLoginId=` your IBKR username
- `IbPassword=` your IBKR password
- `TradingMode=paper` (change to `live` when ready)

### 4. Start Services

```bash
systemctl start xvfb
systemctl start ibgateway
systemctl start hmm-trader.timer
```

### 5. Verify

```bash
# Check IB Gateway is connected (wait 30 seconds after start)
ss -tlnp | grep 4002

# Check timer is scheduled
systemctl list-timers hmm-trader.timer

# View logs
tail -f /opt/hmm-trader/data/trader.log
```

## Architecture

```
[systemd timer: 3:30 PM ET weekdays]
    |
    v
[hmm-trader.service]  -->  python3 -m src.main run --config paper
    |
    v
[ibgateway.service]   -->  IB Gateway (auto-login via IBC)
    |
    v
[xvfb.service]        -->  Virtual display (headless GUI for Java)
```

## Switching to Live Trading

1. Edit IBC config: `nano /opt/ibc/config.ini` -> `TradingMode=live`
2. Restart Gateway: `systemctl restart ibgateway`
3. Edit trader command in service file to use `--config live`
4. Restart timer: `systemctl restart hmm-trader.timer`

## Maintenance

- **Update the app:** `cd /opt/hmm-trader && git pull && source .venv/bin/activate && pip install -r requirements.txt`
- **View trade history:** `sqlite3 /opt/hmm-trader/data/hmm_trader.db "SELECT * FROM trades ORDER BY date DESC LIMIT 20;"`
- **Check regime predictions:** `sqlite3 /opt/hmm-trader/data/hmm_trader.db "SELECT * FROM predictions ORDER BY date DESC LIMIT 10;"`
- **Reset circuit breaker:** `sudo -u hmm-trader bash -c 'cd /opt/hmm-trader && source .venv/bin/activate && python3 -m src.main reset-circuit-breaker'`

## Troubleshooting

| Issue | Fix |
|-------|-----|
| Gateway not connecting | `journalctl -u ibgateway -n 50` — check credentials in config.ini |
| No trades happening | `tail /opt/hmm-trader/data/trader.log` — check for errors |
| Port 4002 not listening | `systemctl restart ibgateway` and wait 30 seconds |
| Timer not firing | `systemctl status hmm-trader.timer` — check it's enabled |
| IBC login failed | IBKR may require 2FA reset; log in manually first via web |
