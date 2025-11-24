#!/bin/bash
# FATRAG Stop Script

PORT=8020
APP_NAME="FATRAG"

echo "🛑 Stopping $APP_NAME..."

# Kill processes on port 8020
if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null 2>&1 ; then
    PID=$(lsof -ti:$PORT)
    if [ -n "$PID" ]; then
        echo "   Killing process on port $PORT (PID: $PID)"
        kill -9 $PID 2>/dev/null || true
        sleep 1
    fi
fi

# Kill any FATRAG python processes
if pgrep -f "python.*main.py" > /dev/null; then
    echo "   Killing FATRAG Python processes..."
    pkill -9 -f "python.*main.py" 2>/dev/null || true
    sleep 1
fi

# Final check
if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null 2>&1 ; then
    echo "⚠️  Port $PORT is still in use. Manual intervention needed:"
    echo "   lsof -i :$PORT"
    exit 1
else
    echo "✅ $APP_NAME stopped successfully"
    echo "   Port $PORT is now free"
fi
