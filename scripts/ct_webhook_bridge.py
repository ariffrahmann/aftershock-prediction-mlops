"""
  1. Menerima webhook dari Alertmanager (HTTP POST /alert)
  2. Memetakan nama alert → tipe event GitHub
  3. Memanggil GitHub REST API `repository_dispatch` yang menjalankan
     workflow .github/workflows/continuous-training.yaml
"""
from __future__ import annotations

import json
import logging
import os
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ct-bridge")

GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
GITHUB_REPO = os.environ.get("GITHUB_REPO", "")
BRIDGE_PORT = int(os.environ.get("BRIDGE_PORT", "9000"))

# Pemetaan nama alert Prometheus → tipe event repository_dispatch
ALERT_TO_EVENT = {
    "GempaWasModelRecallLow": "ct-performance-decay",
    "GempaWasModelDecay": "ct-performance-decay",
    "GempaWasHighErrorRate": "ct-performance-decay",
    "GempaWasDataDriftCritical": "ct-data-drift",
    "GempaWasPredictionDrift": "ct-data-drift",
}


def trigger_github(event_type: str, payload: dict) -> int:
    """Panggil GitHub repository_dispatch API."""
    if not GITHUB_TOKEN or not GITHUB_REPO:
        logger.warning("GITHUB_TOKEN/GITHUB_REPO belum di-set — dispatch dilewati (dry-run).")
        logger.info("DRY-RUN dispatch: event=%s payload=%s", event_type, payload)
        return 0

    url = f"https://api.github.com/repos/{GITHUB_REPO}/dispatches"
    body = json.dumps({"event_type": event_type, "client_payload": payload}).encode()
    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Authorization", f"Bearer {GITHUB_TOKEN}")
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("X-GitHub-Api-Version", "2022-11-28")
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            logger.info("✓ repository_dispatch '%s' terkirim (HTTP %s)", event_type, resp.status)
            return resp.status
    except Exception as exc:
        logger.error("✗ Gagal dispatch ke GitHub: %s", exc)
        return 500


def handle_alertmanager(data: dict) -> dict:
    """Parse payload Alertmanager, dispatch tiap alert yang firing & dikenali."""
    dispatched = []
    for alert in data.get("alerts", []):
        if alert.get("status") != "firing":
            continue
        name = alert.get("labels", {}).get("alertname", "")
        event_type = ALERT_TO_EVENT.get(name)
        if not event_type:
            logger.info("Alert '%s' tidak dipetakan ke CT — diabaikan.", name)
            continue
        payload = {
            "alertname": name,
            "severity": alert.get("labels", {}).get("severity", "unknown"),
            "summary": alert.get("annotations", {}).get("summary", ""),
            "value": alert.get("annotations", {}).get("value", ""),
            "startsAt": alert.get("startsAt", ""),
        }
        status = trigger_github(event_type, payload)
        dispatched.append({"alert": name, "event_type": event_type, "http": status})
    return {"dispatched": dispatched, "count": len(dispatched)}


class Handler(BaseHTTPRequestHandler):
    def _json(self, code: int, obj: dict):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/health", "/healthz", "/ping"):
            self._json(200, {"status": "ok", "service": "ct-webhook-bridge"})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/alert":
            self._json(404, {"error": "use POST /alert"})
            return
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            data = json.loads(raw or b"{}")
        except json.JSONDecodeError:
            self._json(400, {"error": "invalid JSON"})
            return
        logger.info("Webhook diterima: %d alert", len(data.get("alerts", [])))
        result = handle_alertmanager(data)
        self._json(200, result)

    def log_message(self, *_):  # bungkam default access-log
        pass


def main():
    logger.info("CT Webhook Bridge di port %d", BRIDGE_PORT)
    logger.info("Repo target: %s", GITHUB_REPO or "(belum di-set, mode dry-run)")
    HTTPServer(("0.0.0.0", BRIDGE_PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
