#!/bin/bash
# Setup script for Raspberry Pi E-Scooter Tracker

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "Installing system dependencies..."
sudo apt-get update
sudo apt-get install -y python3 python3-pip python3-venv sqlite3

echo
echo "Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

echo
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo
echo "Initializing database..."
python3 collector.py  # This will create the database and exit

echo
echo "Setting up systemd timer for every-minute collection..."

# Create systemd user directory
mkdir -p ~/.config/systemd/user

# Copy systemd files
cp escooter-tracker.service ~/.config/systemd/user/
cp escooter-tracker.timer ~/.config/systemd/user/

# Update service file with correct path
sed -i "s|/home/erike/raspberry-pi-escooter-tracker|$SCRIPT_DIR|g" ~/.config/systemd/user/escooter-tracker.service

# Reload systemd daemon
systemctl --user daemon-reload

# Enable and start timer
systemctl --user enable escooter-tracker.timer
systemctl --user start escooter-tracker.timer
