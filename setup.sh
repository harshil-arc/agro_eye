#!/bin/bash
# ============================================================
# Plant Detection System - Raspberry Pi Setup Script
# ============================================================

set -e

echo "=== Updating System Packages ==="
sudo apt-get update && sudo apt-get upgrade -y

echo "=== Installing System Dependencies for OpenCV (GUI), I2C, SPI, and Serial ==="
sudo apt-get install -y \
    python3-pip \
    python3-dev \
    python3-venv \
    python3-spidev \
    python3-opencv \
    libatlas-base-dev \
    libgl1 \
    libglib2.0-0 \
    libgtk-3-dev \
    libcanberra-gtk-module \
    libcanberra-gtk3-module \
    i2c-tools \
    v4l-utils \
    spi-tools

echo "=== Adding user permissions for Video, I2C, SPI, and Serial GPIO ==="
sudo usermod -a -G video,dialout,i2c,gpio,spi $USER 2>/dev/null || true

echo "=== Creating Python Virtual Environment ==="
python3 -m venv venv
source venv/bin/activate

echo "=== Installing Python Requirements ==="
pip install --upgrade pip
pip install -r requirements.txt

# Hardware specific libraries
pip install adafruit-circuitpython-dht adafruit-circuitpython-ads1x15 RPi.GPIO spidev

echo "=== Setup Completed Successfully ==="
echo "You can now run: source venv/bin/activate && python main.py"
