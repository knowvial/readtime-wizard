#!/bin/bash

# Start nginx
nginx

# Start the FastAPI application
uvicorn main:app --host 0.0.0.0 --port 8000