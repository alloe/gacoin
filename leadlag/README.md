# Lead/Lag Research Pilot

Research-only branch for Binance Spot + Binance USDT Futures -> Upbit KRW lead/lag measurement.

- Pilot date: 2026-09-01 (Tardis first-day-of-month free sample)
- Coins: BTC, ETH, XRP, SOL, DOGE
- No trading/account connectivity
- Uses Tardis normalized trade CSVs and local_timestamp
- Measures best 1-second lag and conditional Upbit catch-up after independent confirmed Binance impulses.
