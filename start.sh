#!/bin/bash
cd /home/ubuntu/network-fault-detection
source venv/bin/activate
mkdir -p logs
pkill -f gunicorn 2>/dev/null || true

echo "🚀 Starting Hybrid ML Network Fault Detection API..."
gunicorn --bind 0.0.0.0:5000 --workers 2 --timeout 120 --daemon "src.api.app:create_app()"

sleep 3
if curl -s http://localhost:5000/health > /dev/null; then
    echo "✅ Hybrid ML API started successfully at http://13.53.207.16:5000"
else
    echo "❌ API failed to start"
fi
