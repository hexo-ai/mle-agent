#!/bin/sh
gunicorn mle_agent.main:app -w 1 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000 -t 1000000
