"""
tools — pluggable tool layer for Ecko.

Tools are discrete capabilities the LLM can draw on at inference time.
Each tool is a module that exposes a consistent interface.

Current tools:
  websearch — Brave Search API integration

Adding a tool:
  1. Create tools/mytool.py
  2. Import and register it here if it needs global init
  3. Wire its route in web/routes/ if it needs an HTTP endpoint
"""
