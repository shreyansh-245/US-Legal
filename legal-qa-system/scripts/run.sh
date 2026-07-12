#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/.."

echo "Installing dependencies..."
pip install -r requirements.txt

echo "Building the hybrid index from data/documents/..."
python backend/ingest.py

echo "Starting API server on http://localhost:8000 ..."
python backend/main.py
