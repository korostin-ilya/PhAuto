FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

# Установка системных зависимостей: Hugin CLI, libvips, ExifTool и библиотек для сборки Multiblend
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv \
    hugin-tools enblend \
    libvips-dev libimage-exiftool-perl \
    libgl1 libglib2.0-0 build-essential git cmake curl ca-certificates \
    libtiff-dev libjpeg-dev libpng-dev \
    && rm -rf /var/lib/apt/lists/*

# Сборка актуального Multiblend 2.0 из официального архива автора (David Horman)
RUN mkdir -p /tmp/multiblend \
    && curl -fsSL https://horman.net/multiblend/multiblend2.0rc5.tar.gz | tar -xz -C /tmp/multiblend --strip-components=1 \
    && cd /tmp/multiblend \
    && g++ -msse4.1 -pthread -ffast-math -Ofast -o /usr/local/bin/multiblend src/multiblend.cpp -lm -lpng -ltiff -ljpeg \
    && rm -rf /tmp/multiblend

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends python3-dev pkg-config && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip3 install --no-cache-dir --break-system-packages -r requirements.txt

COPY . .
ENTRYPOINT ["python3", "scripts/run_pipeline.py"]
