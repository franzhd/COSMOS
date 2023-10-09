#!/bin/bash

PORT=6969
echo "Starting Gunicorn."
echo -e "\n\nGo to http://localhost:${PORT}/docs to access the app.\n\n"
gunicorn src.routes:app -w 1 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:${PORT} --timeout 500
