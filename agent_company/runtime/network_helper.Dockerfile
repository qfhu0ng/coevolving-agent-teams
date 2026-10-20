FROM python:3.12-slim

RUN apt-get update \
    && apt-get install --no-install-recommends -y iptables \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
COPY netns_firewall.py /opt/agent-company/netns_firewall.py
ENTRYPOINT ["python", "/opt/agent-company/netns_firewall.py"]
