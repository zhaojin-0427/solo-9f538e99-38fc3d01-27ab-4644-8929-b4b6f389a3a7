FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    DB_PATH=/data/acceptance.db \
    MALLOC_ARENA_MAX=1

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

RUN mkdir -p /data
EXPOSE 8000

# MALLOC_ARENA_MAX=1: 限制 glibc 内存分配 arena(每个预留 64MiB 虚拟地址空间),
# 使进程在 256MiB 地址空间限制(ulimit -v / RLIMIT_AS)下也能完成大批次重算
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
