# Deploying Elite 2025 War-Room to Cloud

This guide explains how to deploy the full system (Rust Backend + React Frontend + Python Feeder) to a cloud provider (AWS/GCP/DigitalOcean) using Docker.

## Prerequisites
- Docker & Docker Compose installed on the server.
- API Keys for Binance/Hyperliquid.

## 1. Setup Environment
Copy `.env.example` to `.env` and fill in your keys:
```bash
cp .env.example .env
nano .env
```
Set `LIVE_MODE=true` to enable real data.

## 2. Build & Run
Run the entire stack in detached mode:
```bash
docker-compose up -d --build
```

## 3. Verify Deployment
- **Dashboard**: `http://<YOUR_SERVER_IP>:1420`
- **Backend Logs**: `docker-compose logs -f elite-dashboard`

## 4. Security Note
- The dashboard is currently exposed on port 1420.
- **RECOMMENDED**: Set up Nginx with Basic Auth or VPN access to protect your War Room.
- Do **NOT** expose this publicly without authentication.

## 5. Troubleshooting
If the feeder crashes:
```bash
docker-compose restart elite-dashboard
```
Check logs:
```bash
docker logs -f <container_id>
```
