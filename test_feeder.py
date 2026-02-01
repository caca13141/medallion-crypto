#!/usr/bin/env python3
import time
import json
import requests
import numpy as np

DASHBOARD_URL = "http://localhost:3000/push"

def push_update(topic, payload):
    try:
        data = {
            "topic": topic,
            "payload": payload,
            "timestamp": int(time.time() * 1000)
        }
        resp = requests.post(DASHBOARD_URL, json=data, timeout=1.0)
        print(f"  → Pushed {topic}, Status: {resp.status_code}")
    except Exception as e:
        print(f"  ✗ Push failed: {e}")

print("=== SIMPLE FEEDER TEST ===")
print("Sending test data to dashboard server...")

for i in range(5):
    print(f"\nPulse {i+1}...")
    
    mid_price = 90000 + i * 100
    pulse = {
        "l3_book": {
            "bids": [[mid_price - j*2, 1.0] for j in range(10)],
            "asks": [[mid_price + j*2, 1.0] for j in range(10)]
        },
        "signals": {
            "tti": 5.0 + i * 0.1,
            "vpin": 0.5
        },
        "forecast": {
            "tti": 5.0 + i * 0.1,
            "confidence": 0.85
        },
        "monte_carlo": {
            "paths": [[mid_price] * 48],
            "last_price": mid_price,
            "greeks": {"delta": 0.5, "gamma": 0.1, "var_95": -2.0, "cvar_95": -3.0, "avar_95": -2.5},
            "roughness": 0.5
        },
        "intelligence_audit": {
            "current_dominance": 15.0 + i,
            "loss_curve": [0.3] * 20,
            "expert_landscape": [0.1, 0.2, 0.15, 0.05, 0.3, 0.1, 0.05, 0.05],
            "status": "CONVERGING"
        }
    }
    
    push_update("marksman_pulse", pulse)
    time.sleep(2)

print("\n=== TEST COMPLETE ===")
