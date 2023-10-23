#!/bin/sh

gunicorn main:app -w 1 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:80 -t 1000
