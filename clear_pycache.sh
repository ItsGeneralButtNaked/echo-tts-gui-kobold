#!/bin/bash
# Run this in your ecko project root before restarting the server
# It deletes all stale bytecode caches so Python re-reads the patched .py files
find . -name "*.pyc" -delete
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null
echo "All .pyc files and __pycache__ dirs deleted."
echo "Restart ecko_web.py now."
