# Multi-stage build for Elite Dashboard

# Stage 1: Build Rust Backend
FROM rust:1.75 as backend-builder
WORKDIR /app
COPY dashboard_server .
RUN cargo build --release

# Stage 2: Build React Frontend
FROM node:20 as frontend-builder
WORKDIR /app
COPY alien_dashboard .
RUN npm install
RUN npm run build

# Stage 3: Final Image (Python + Rust + Nginx/Static)
FROM python:3.11-slim

# Install system deps
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy Backend Binary
COPY --from=backend-builder /app/target/release/dashboard_server /app/dashboard_server

# Copy Frontend Build
COPY --from=frontend-builder /app/dist /app/static

# Copy Python Feeder & Source
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install ccxt python-dotenv requests torch numpy

COPY feed_dashboard.py .
COPY src ./src
COPY .env.example .env

# Expose Ports
EXPOSE 3000 1420

# Startup Script
COPY launch_war_room.sh .
RUN chmod +x launch_war_room.sh

# Modify launch script for Docker (no npm run dev, serve static)
# Ideally we'd use Nginx, but for simplicity we'll let the rust server serve static files or just run the feeder
# For this "War Room" setup, we kept the dev server structure. 
# In a real prod env, we'd serve the React build via Nginx.
# For now, let's stick to the script but we need Node in the final image if we want to run "npm run dev"
# OR we serve the static files from the Rust server.

# Let's add Node to the final image to keep "npm run dev" working as per the user's "War Room" script
RUN apt-get update && apt-get install -y nodejs npm

CMD ["./launch_war_room.sh"]
