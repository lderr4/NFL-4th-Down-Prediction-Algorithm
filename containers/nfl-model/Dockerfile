FROM python:3.8-slim

COPY requirements.txt /opt
RUN pip install --no-cache-dir -r /opt/requirements.txt

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

CMD tail -f /dev/null

