"""
Script untuk mensimulasikan beban kerja nyata ke inference API.
Menghasilkan trafik realistis dengan variasi:
  - Burst traffic (lonjakan sesaat)
  - Steady-state load
  - Skenario data drift (fitur yang bergeser distribusinya)

Melihat/Memantau hasilnya di Grafana: http://localhost:3000
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Literal

import requests

# Logging 
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("gempawas.load_sim")

# Konfigurasi
API_URL   = "http://localhost:8080/invocations"
PING_URL  = "http://localhost:8080/ping"
TIMEOUT   = 10  # detik

# Feature Generators 
def _normal_features() -> list[float]:
    """
    representasi distribusi baseline model saat training.
    """
    return [
        round(random.gauss(5.5, 0.8), 2),   # mainshock_magnitude
        round(random.gauss(30.0, 15.0), 2), # mainshock_depth
        round(random.expovariate(1 / 12.0), 2),  # jam_sejak_mainshock
        random.randint(0, 5),               # count_susulan_1jam
        random.randint(0, 15),              # count_susulan_6jam
        random.randint(0, 40),              # count_susulan_24jam
        round(random.gauss(3.5, 0.7), 2),   # max_mag_susulan_6jam
        round(random.gauss(4.0, 0.8), 2),   # max_mag_susulan_24jam
        round(random.gauss(2.5, 1.2), 2),   # omori_rate_est
        float(random.randint(0, 3)),         # zona_sesar (0-3)
    ]


def _drift_features() -> list[float]:
    """
    distribusi bergeser dari baseline.
    Mensimulasikan kondisi di mana karakteristik gempa berubah
    untuk menguji deteksi Data Drift di dashboard Grafana.
    """
    return [
        round(random.gauss(6.8, 0.5), 2),   # magnitude lebih tinggi (drift!)
        round(random.gauss(80.0, 20.0), 2), # depth lebih dalam (drift!)
        round(random.expovariate(1 / 3.0), 2),  # lebih dekat ke mainshock
        random.randint(5, 20),              # lebih banyak susulan cepat
        random.randint(15, 50),
        random.randint(40, 100),
        round(random.gauss(5.2, 0.5), 2),   # susulan lebih kuat
        round(random.gauss(5.8, 0.6), 2),
        round(random.gauss(5.0, 1.0), 2),
        float(random.randint(0, 3)),
    ]


def _random_features() -> list[float]:
    """Fitur acak murni — untuk uji robustness model."""
    return [round(random.uniform(0, 10), 2) for _ in range(10)]


# Request Sender 

@dataclass
class RequestResult:
    success:       bool
    latency_ms:    float
    status_code:   int = 0
    replica:       str = "unknown"
    prediction:    float | None = None
    error:         str = ""


def send_request(feature_fn, session: requests.Session) -> RequestResult:
    payload = {"inputs": [feature_fn()]}
    start   = time.perf_counter()
    try:
        resp = session.post(API_URL, json=payload, timeout=TIMEOUT)
        latency = (time.perf_counter() - start) * 1000
        if resp.status_code == 200:
            data = resp.json()
            pred = data.get("predictions", [None])[0]
            replica = data.get("replica", "unknown")
            return RequestResult(
                success=True, latency_ms=latency,
                status_code=200, replica=replica, prediction=pred,
            )
        return RequestResult(
            success=False, latency_ms=latency,
            status_code=resp.status_code, error=resp.text[:200],
        )
    except requests.exceptions.Timeout:
        latency = (time.perf_counter() - start) * 1000
        return RequestResult(success=False, latency_ms=latency, error="Timeout")
    except Exception as exc:
        latency = (time.perf_counter() - start) * 1000
        return RequestResult(success=False, latency_ms=latency, error=str(exc))


# Simulation Modes

def wait_for_api(retries: int = 15, delay: float = 3.0) -> bool:
    logger.info(f"Menunggu API siap di {PING_URL}...")
    for i in range(retries):
        try:
            r = requests.get(PING_URL, timeout=5)
            if r.status_code == 200:
                logger.info("✅ API siap!")
                return True
        except Exception:
            pass
        logger.info(f"   Percobaan {i+1}/{retries} gagal, coba lagi dalam {delay}s...")
        time.sleep(delay)
    return False


def run_simulation(
    mode:     Literal["steady", "burst", "drift"] = "steady",
    rps:      int   = 10,
    duration: int   = 120,
    workers:  int   = 20,
):
    feature_fn_map = {
        "steady": _normal_features,
        "burst":  _normal_features,
        "drift":  _drift_features,
    }
    feature_fn = feature_fn_map[mode]
    interval   = 1.0 / rps if rps > 0 else 0.1

    logger.info(
        f"\n{'='*60}\n"
        f"  Mode      : {mode.upper()}\n"
        f"  Target RPS: {rps}\n"
        f"  Duration  : {duration}s\n"
        f"  Workers   : {workers}\n"
        f"  API URL   : {API_URL}\n"
        f"{'='*60}"
    )

    results: list[RequestResult] = []
    start_time = time.time()
    request_id = 0

    # Burst mode: ganti interval
    if mode == "burst":
        burst_phase    = 10   # 10 detik burst
        cooldown_phase = 20   # 20 detik cooldown, lalu burst lagi
        logger.info(f"Burst pattern: {burst_phase}s burst @ {rps} RPS → {cooldown_phase}s cooldown @ 2 RPS")

    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(pool_connections=workers, pool_maxsize=workers)
    session.mount("http://", adapter)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = []
        elapsed = 0.0

        while elapsed < duration:
            elapsed = time.time() - start_time

            # Burst mode: alternate interval
            if mode == "burst":
                phase_pos = elapsed % (burst_phase + cooldown_phase)
                effective_interval = interval if phase_pos < burst_phase else 0.5
            else:
                effective_interval = interval

            future = executor.submit(send_request, feature_fn, session)
            futures.append(future)
            request_id += 1

            # Progress log setiap 10 detik
            if request_id % (rps * 10) == 0:
                done_results = [f.result() for f in futures if f.done()]
                if done_results:
                    successes    = sum(1 for r in done_results if r.success)
                    latencies    = [r.latency_ms for r in done_results if r.success]
                    replica_dist = {}
                    for r in done_results:
                        if r.success:
                            replica_dist[r.replica] = replica_dist.get(r.replica, 0) + 1
                    p50 = statistics.median(latencies) if latencies else 0
                    p99 = sorted(latencies)[int(len(latencies) * 0.99)] if latencies else 0
                    logger.info(
                        f"  [{elapsed:.0f}s] Sent={request_id} "
                        f"OK={successes}/{len(done_results)} "
                        f"P50={p50:.1f}ms P99={p99:.1f}ms "
                        f"Replicas={dict(sorted(replica_dist.items()))}"
                    )

            time.sleep(effective_interval)

        # Kumpulkan semua hasil
        logger.info("Menunggu semua request selesai...")
        for future in as_completed(futures, timeout=30):
            try:
                results.append(future.result())
            except Exception as exc:
                results.append(RequestResult(success=False, latency_ms=0, error=str(exc)))

    # Summary 
    total     = len(results)
    succeeded = sum(1 for r in results if r.success)
    failed    = total - succeeded
    latencies = sorted(r.latency_ms for r in results if r.success)
    replica_dist = {}
    for r in results:
        if r.success:
            replica_dist[r.replica] = replica_dist.get(r.replica, 0) + 1

    actual_rps = total / duration if duration > 0 else 0

    print(f"\n{'═'*60}")
    print(f"  HASIL SIMULASI — Mode: {mode.upper()}")
    print(f"{'═'*60}")
    print(f"  Total Request  : {total}")
    print(f"  Sukses         : {succeeded} ({100*succeeded/total:.1f}%)")
    print(f"  Gagal          : {failed}")
    print(f"  Actual RPS     : {actual_rps:.1f} req/s")
    if latencies:
        print(f"  Latency Min    : {min(latencies):.1f} ms")
        print(f"  Latency Median : {statistics.median(latencies):.1f} ms")
        print(f"  Latency P90    : {latencies[int(len(latencies)*0.90)]:.1f} ms")
        print(f"  Latency P99    : {latencies[int(len(latencies)*0.99)]:.1f} ms")
        print(f"  Latency Max    : {max(latencies):.1f} ms")
    print(f"\n  Distribusi Trafik per Replika:")
    total_replica = sum(replica_dist.values()) or 1
    for replica, count in sorted(replica_dist.items()):
        bar = "█" * int(count / total_replica * 30)
        print(f"    {replica:<30} {count:>5} ({100*count/total_replica:.1f}%) {bar}")
    print(f"{'═'*60}\n")

    return results


# Main 
def main():
    parser = argparse.ArgumentParser(
        description="GempaWas Load Simulation — uji monitoring stack dengan trafik realistis"
    )
    parser.add_argument(
        "--mode",
        choices=["steady", "burst", "drift", "all"],
        default="steady",
        help="Mode simulasi: steady=beban konstan, burst=lonjakan, drift=data drift, all=ketiganya",
    )
    parser.add_argument("--rps",      type=int, default=10,  help="Target request per second")
    parser.add_argument("--duration", type=int, default=120, help="Durasi simulasi (detik)")
    parser.add_argument("--workers",  type=int, default=20,  help="Jumlah thread konkuren")
    parser.add_argument("--no-wait",  action="store_true",   help="Skip pengecekan API health")
    args = parser.parse_args()

    if not args.no_wait:
        if not wait_for_api():
            logger.error("❌ API tidak merespons. Pastikan stack berjalan: docker compose up -d")
            sys.exit(1)

    if args.mode == "all":
        logger.info("Menjalankan semua mode simulasi secara berurutan...")
        run_simulation("steady", rps=args.rps,     duration=args.duration, workers=args.workers)
        time.sleep(10)
        run_simulation("burst",  rps=args.rps * 5, duration=60,            workers=args.workers)
        time.sleep(10)
        run_simulation("drift",  rps=args.rps,     duration=args.duration, workers=args.workers)
    else:
        run_simulation(args.mode, rps=args.rps, duration=args.duration, workers=args.workers)


if __name__ == "__main__":
    main()
