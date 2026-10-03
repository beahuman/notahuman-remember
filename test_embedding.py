#!/usr/bin/env python3
"""Test embedding API connections."""

import httpx
import json
import os

# Test OpenAI directly
print("=== Testing OpenAI API directly ===")
try:
    resp = httpx.post(
        "https://api.openai.com/v1/embeddings",
        headers={
            "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY', '')}",
            "Content-Type": "application/json",
        },
        json={"model": "text-embedding-3-small", "input": "test"},
        timeout=30,
    )
    data = resp.json()
    if "data" in data:
        dims = len(data["data"][0]["embedding"])
        print(f"SUCCESS: text-embedding-3-small returns {dims} dimensions")
    else:
        print(f"Error: {json.dumps(data, indent=2)}")
except Exception as e:
    print(f"Connection Error: {e}")

# Test Bifrost gateway
print("\n=== Testing Bifrost Gateway ===")
try:
    resp = httpx.post(
        "http://localhost:4000/embeddings",
        headers={
            "Authorization": f"Bearer {os.getenv('NAH_MODEL_GATEWAY_KEY', '')}",
            "Content-Type": "application/json",
        },
        json={"model": "embedding-primary", "input": "test"},
        timeout=30,
    )
    data = resp.json()
    if "data" in data:
        dims = len(data["data"][0]["embedding"])
        print(f"SUCCESS: embedding-primary returns {dims} dimensions")
    else:
        print(f"Error: {json.dumps(data, indent=2)}")
except Exception as e:
    print(f"Connection Error: {e}")
