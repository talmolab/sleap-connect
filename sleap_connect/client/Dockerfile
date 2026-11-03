# Base image with SLEAP and GPU support
# https://hub.docker.com/repository/docker/eberrigan/sleap-cuda/general
# FROM eberrigan/sleap-cuda:latest

# Use Python base image and install uv
FROM python:3.12-slim-trixie
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set working directory
WORKDIR /app

# Install dependencies
# opencv requires opengl https://github.com/conda-forge/opencv-feedstock/issues/401
# Default python3 is 3.8 in ubuntu 20.04 https://wiki.ubuntu.com/FocalFossa/ReleaseNotes#Python3_by_default
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     apt-get clean && \
#     rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* 

# Goal: Connect to Remote Signaling Server w/ aiortc and websockets
  
# Install websocket dependencies onto image @ build time (asyncio is included in python3.8)
# put into requirement.txt later

# Install zip function
RUN apt-get update && apt-get install -y zip unzip 

# Install dependencies 
COPY pyproject.toml /app/
RUN uv sync --extra dev

# Copy the sleap_webRTC directory into the image's root dir
COPY client.py /app/
COPY client_class.py /app/

# Create shared file directory inside container 
RUN mkdir -p /app/shared_data && chmod 777 /app/shared_data

# Port exposed by the container, NOT computer (localhost), 8080 = websocket server, 3478 = TURN server
EXPOSE 8080 8001 5000 3478/udp 3478/tcp 9001/tcp 9000/tcp

# Signaling server -> worker -> client in sequenece
# Worker makes req to signaling server, signaling server sends offer to client and accepts offer
# Connection is established between client and worker
# ENTRYPOINT ["sh", "-c", "python3 /app/worker.py"]




