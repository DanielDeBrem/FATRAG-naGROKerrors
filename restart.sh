#!/bin/bash
# FATRAG Restart Script

APP_NAME="FATRAG"

echo "🔄 Restarting $APP_NAME..."
echo ""

# Stop the app
./stop.sh

echo ""

# Start the app
./start.sh
