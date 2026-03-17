#!/bin/bash
# Daily HMM trader cron job
# Runs at 3:30 PM ET on weekdays

cd /home/dwagner003/dev/hmm-trader
source .venv/bin/activate

# Check if IB Gateway is running
if ! ss -tlnp 2>/dev/null | grep -q ":4002"; then
    echo "$(date) ERROR: IB Gateway not running on port 4002" >> data/cron.log
    exit 1
fi

echo "$(date) Starting daily rebalance..." >> data/cron.log
IBKR_PORT=4002 python3 -m src.main run --config paper >> data/cron.log 2>&1
echo "$(date) Done." >> data/cron.log
