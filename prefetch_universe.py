"""Pre-fetch ALL USDT perpetual 1h klines to _kline_cache/ with rate limiting.
Runs at ~300 RPM — well under Binance's 2400/min limit. Safe for long runs.
"""
import json, os, sys, time, urllib.request, urllib.error

CACHE = os.path.join(os.path.dirname(__file__), "_kline_cache")
os.makedirs(CACHE, exist_ok=True)

# 1. Exchange info
print("Fetching exchange_info...", flush=True)
url = "https://fapi.binance.com/fapi/v1/exchangeInfo"
with urllib.request.urlopen(url, timeout=15) as resp:
    ei = json.loads(resp.read().decode())
with open(os.path.join(CACHE, "_exchange_info.json"), "w") as f:
    json.dump(ei, f)

symbols = sorted(
    s["symbol"] for s in ei["symbols"]
    if s.get("contractType") == "PERPETUAL" and s.get("status") == "TRADING" and s.get("quoteAsset") == "USDT"
)
print(f"Found {len(symbols)} USDT perpetuals", flush=True)

# 2. Count existing
existing = set()
for f in os.listdir(CACHE):
    if f.endswith("_1h.json") and f != "_exchange_info.json":
        existing.add(f.replace("_1h.json", ""))
print(f"Already cached: {len(existing)}", flush=True)
missing = [s for s in symbols if s not in existing]
print(f"Need to fetch: {len(missing)}", flush=True)

if not missing:
    print("All symbols already cached!")
    sys.exit(0)

# 3. Fetch with rate limiting (~300 RPM = 200ms between requests)
fetched = 0
errors = 0
t0 = time.time()
for i, sym in enumerate(missing):
    try:
        url = f"https://fapi.binance.com/fapi/v1/klines?symbol={sym}&interval=1h&limit=1500"
        req = urllib.request.Request(url, headers={"User-Agent": "sweep/1.0"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read().decode())
        if isinstance(data, list) and len(data) > 0:
            with open(os.path.join(CACHE, f"{sym}_1h.json"), "w") as f:
                json.dump(data, f)
            fetched += 1
    except Exception as e:
        errors += 1
        if errors <= 3:
            print(f"  {sym}: {e}", flush=True)

    if (i + 1) % 50 == 0:
        elapsed = time.time() - t0
        rpm = (i + 1) / elapsed * 60
        print(f"  [{i+1}/{len(missing)}] {fetched} ok, {errors} err, {rpm:.0f} RPM", flush=True)

    # Rate limit: ~300 RPM max
    time.sleep(0.20)

    # If we get banned, stop and save progress
    if errors > 20:
        print(f"Too many errors ({errors}) — likely banned. Stopping.", flush=True)
        break

elapsed = time.time() - t0
total = len(existing) + fetched
print(f"\nDONE: {total} symbols cached in {CACHE} ({elapsed:.0f}s, {fetched} new, {errors} errors)", flush=True)
