#!/bin/bash
# ============================================================
# HMM Trader - VPS Deployment Script
# ============================================================
# Run this on a fresh Ubuntu 22.04+ VPS (DigitalOcean, Hetzner, etc.)
# Requires: 2GB RAM, 20GB disk, root access
#
# Usage:
#   1. Create a VPS (Ubuntu 22.04, 2GB RAM, ~$6/mo)
#   2. SSH in: ssh root@your-server-ip
#   3. Upload this script and run: bash setup-server.sh
#   4. Edit /opt/hmm-trader/.env with your credentials
#   5. Edit /opt/ibc/config.ini with your IBKR credentials
#   6. Reboot and verify
# ============================================================

set -e

echo "=========================================="
echo "  HMM Trader - Server Setup"
echo "=========================================="

# --- System packages ---
echo "[1/7] Installing system packages..."
apt-get update -qq
apt-get install -y -qq \
    python3 python3-pip python3-venv \
    xvfb \
    unzip wget curl git \
    default-jre \
    cron \
    > /dev/null 2>&1
echo "  Done."

# --- Create app user ---
echo "[2/7] Creating hmm-trader user..."
id -u hmm-trader &>/dev/null || useradd -m -s /bin/bash hmm-trader
echo "  Done."

# --- Clone the app ---
echo "[3/7] Setting up HMM Trader app..."
if [ ! -d /opt/hmm-trader ]; then
    git clone https://github.com/dwagner003/hmm-trader.git /opt/hmm-trader
else
    cd /opt/hmm-trader && git pull
fi
cd /opt/hmm-trader
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -q
pip install ibapi -q
deactivate
chown -R hmm-trader:hmm-trader /opt/hmm-trader
echo "  Done."

# --- Install IB Gateway ---
echo "[4/7] Installing IB Gateway..."
if [ ! -d /opt/ibgateway ]; then
    cd /tmp
    wget -q "https://download2.interactivebrokers.com/installers/ibgateway/latest-standalone/ibgateway-latest-standalone-linux-x64.sh" -O ibgateway-install.sh
    chmod +x ibgateway-install.sh
    # Install to /opt/ibgateway
    bash ibgateway-install.sh -q -dir /opt/ibgateway
    chown -R hmm-trader:hmm-trader /opt/ibgateway
fi
echo "  Done."

# --- Install IBC (IB Controller - auto-login for Gateway) ---
echo "[5/7] Installing IBC..."
IBC_VERSION="3.18.0"
if [ ! -d /opt/ibc ]; then
    cd /tmp
    wget -q "https://github.com/IbcAlpha/IBC/releases/download/${IBC_VERSION}/IBCLinux-${IBC_VERSION}.zip" -O ibc.zip
    mkdir -p /opt/ibc
    unzip -q -o ibc.zip -d /opt/ibc
    chmod +x /opt/ibc/*.sh
    chown -R hmm-trader:hmm-trader /opt/ibc
fi

# Create IBC config
cat > /opt/ibc/config.ini << 'IBCCONFIG'
# IBC Configuration
# IMPORTANT: Edit these with your IBKR credentials

IbLoginId=YOUR_IBKR_USERNAME
IbPassword=YOUR_IBKR_PASSWORD
TradingMode=paper

# Auto-accept non-brokerage account warning
AcceptNonBrokerageAccountWarning=yes

# Accept incoming API connections
AcceptIncomingConnectionAction=accept

# Gateway settings
IbDir=/opt/ibgateway
ExistingSessionDetectedAction=primary
OverrideTwsApiPort=4002
ReadOnlyApi=no
IBCCONFIG
chown hmm-trader:hmm-trader /opt/ibc/config.ini
chmod 600 /opt/ibc/config.ini
echo "  Done."

# --- Create systemd services ---
echo "[6/7] Creating systemd services..."

# Virtual display service (needed for IB Gateway's Java GUI)
cat > /etc/systemd/system/xvfb.service << 'EOF'
[Unit]
Description=Virtual X Display (Xvfb)
After=network.target

[Service]
Type=simple
ExecStart=/usr/bin/Xvfb :1 -screen 0 1024x768x24
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

# IB Gateway service (managed by IBC)
cat > /etc/systemd/system/ibgateway.service << 'EOF'
[Unit]
Description=IB Gateway (via IBC)
After=xvfb.service
Requires=xvfb.service

[Service]
Type=simple
User=hmm-trader
Environment=DISPLAY=:1
ExecStart=/opt/ibc/gatewaystart.sh -inline /opt/ibc/config.ini
Restart=always
RestartSec=30

[Install]
WantedBy=multi-user.target
EOF

# HMM Trader timer (runs at 3:30 PM ET weekdays)
cat > /etc/systemd/system/hmm-trader.service << 'EOF'
[Unit]
Description=HMM Trader Daily Rebalance
After=ibgateway.service
Requires=ibgateway.service

[Service]
Type=oneshot
User=hmm-trader
WorkingDirectory=/opt/hmm-trader
Environment=IBKR_PORT=4002
Environment=DATA_DIR=/opt/hmm-trader/data
ExecStart=/opt/hmm-trader/.venv/bin/python3 -m src.main run --config paper
StandardOutput=append:/opt/hmm-trader/data/trader.log
StandardError=append:/opt/hmm-trader/data/trader.log
EOF

cat > /etc/systemd/system/hmm-trader.timer << 'EOF'
[Unit]
Description=HMM Trader Daily Timer

[Timer]
# 19:30 UTC = 3:30 PM EDT (summer), 2:30 PM EST (winter)
# Both are before the 3:45 PM ET MOC cutoff
OnCalendar=Mon..Fri 19:30:00 UTC
Persistent=true

[Install]
WantedBy=timers.target
EOF

# Enable services
systemctl daemon-reload
systemctl enable xvfb ibgateway hmm-trader.timer
echo "  Done."

# --- Create .env template ---
echo "[7/7] Creating configuration..."
if [ ! -f /opt/hmm-trader/.env ]; then
    cat > /opt/hmm-trader/.env << 'ENVFILE'
# IBKR Connection
IBKR_HOST=127.0.0.1
IBKR_PORT=4002
IBKR_CLIENT_ID=1

# Data directory
DATA_DIR=/opt/hmm-trader/data

# Email Alerts (optional - fill in to receive alerts)
# SMTP_HOST=smtp.gmail.com
# SMTP_PORT=587
# SMTP_USER=your-email@gmail.com
# SMTP_PASSWORD=your-app-password
# ALERT_EMAIL_TO=your-email@gmail.com
ENVFILE
    chown hmm-trader:hmm-trader /opt/hmm-trader/.env
    chmod 600 /opt/hmm-trader/.env
fi

# Create data directory
mkdir -p /opt/hmm-trader/data
chown hmm-trader:hmm-trader /opt/hmm-trader/data

echo ""
echo "=========================================="
echo "  Setup Complete!"
echo "=========================================="
echo ""
echo "  NEXT STEPS:"
echo ""
echo "  1. Edit IBKR credentials:"
echo "     nano /opt/ibc/config.ini"
echo "     -> Set IbLoginId and IbPassword"
echo ""
echo "  2. (Optional) Set up email alerts:"
echo "     nano /opt/hmm-trader/.env"
echo ""
echo "  3. Start everything:"
echo "     systemctl start xvfb"
echo "     systemctl start ibgateway"
echo "     systemctl start hmm-trader.timer"
echo ""
echo "  4. Verify IB Gateway is running:"
echo "     sleep 30 && ss -tlnp | grep 4002"
echo ""
echo "  5. Test a manual run:"
echo "     sudo -u hmm-trader bash -c 'cd /opt/hmm-trader && source .venv/bin/activate && IBKR_PORT=4002 python3 -m src.main run --config paper'"
echo ""
echo "  6. Check timer status:"
echo "     systemctl status hmm-trader.timer"
echo "     journalctl -u hmm-trader.service"
echo ""
echo "  MONITORING:"
echo "     tail -f /opt/hmm-trader/data/trader.log"
echo "     systemctl status ibgateway"
echo ""
