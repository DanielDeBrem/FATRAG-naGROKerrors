#!/bin/bash
# FATRAG Start Script - Force port 8020

set -e

export PORT=8020
# Ensure multi-GPU Ollama workers are used by the API (round-robin in main.py)
export OLLAMA_WORKER_PORTS="11434,11435,11436,11437,11438,11439,11440,11441"
APP_NAME="FATRAG"

echo "🚀 Starting $APP_NAME on port $PORT..."

# Check if port is in use
if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null 2>&1 ; then
    echo "⚠️  Port $PORT is in use. Killing process..."
    
    # Get PID of process using the port
    PID=$(lsof -ti:$PORT)
    
    if [ -n "$PID" ]; then
        echo "   Killing PID: $PID"
        kill -9 $PID 2>/dev/null || true
        sleep 1
    fi
    
    # Double check
    if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null 2>&1 ; then
        echo "❌ Failed to free port $PORT. Please check manually:"
        echo "   lsof -i :$PORT"
        exit 1
    fi
    
    echo "✓ Port $PORT is now free"
fi

# Kill any existing FATRAG python processes (just to be safe)
pkill -f "python.*main.py" 2>/dev/null || true
sleep 1

# Activate venv if it exists
if [ -d ".venv" ]; then
    echo "📦 Activating virtual environment..."
    source .venv/bin/activate
fi

# Start the app in background with forced port
echo "🔧 Starting $APP_NAME..."
nohup python main.py > fatrag.log 2>&1 &
PID=$!

# Wait a moment and check if it started
sleep 2

# Determine LAN IP for remote access
LAN_IP=$(hostname -I | awk '{print $1}')
if [ -z "$LAN_IP" ]; then LAN_IP="127.0.0.1"; fi

if kill -0 $PID 2>/dev/null; then
    echo "✅ $APP_NAME started successfully!"
    echo "   PID: $PID"
    echo "   URL: http://0.0.0.0:$PORT (bind)"
    echo "   LAN: http://$LAN_IP:$PORT"
    echo "   Admin: http://$LAN_IP:$PORT/admin"
    echo "   Clients: http://$LAN_IP:$PORT/admin/clients.html"
    echo ""
    echo "   Logs: tail -f fatrag.log"
    echo "   Stop: ./stop.sh"
else
    echo "❌ Failed to start $APP_NAME. Check fatrag.log"
    tail -20 fatrag.log
    exit 1
fi
