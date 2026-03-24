# Investment Comparison App (XGBoost)

XGBoost-powered application that predicts relative outperformance between fund pairs:

- **Value vs Growth**: VVIAX vs VIGAX
- **US vs International**: VTSAX vs VTIAX

Predictions are generated across 1, 3, 6, and 12-month horizons using macroeconomic indicators from FRED and historical fund price data.

## Quick Start (Streamlit)

```bash
pip install -r requirements.txt
cp .env.example .env  # Add your FRED API key
streamlit run app.py
```

## Deploy to Streamlit Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) and connect your repo
3. Set `app.py` as the main file
4. Add `FRED_API_KEY` in the Secrets dashboard

## Architecture

- **Streamlit app** (`app.py`): Single-file UI with predictions, charts, feature importance, and metrics
- **Backend modules** (`backend/`): Data fetching, feature engineering, XGBoost training with walk-forward validation
- **FastAPI option** (`backend/main.py`): REST API if you prefer a separate frontend
- **React frontend** (`frontend/`): Alternative React/Vite UI that talks to the FastAPI backend

## Alternative: FastAPI + React

```bash
# Terminal 1 — Backend
cd backend
pip install -r requirements.txt
uvicorn main:app --reload

# Terminal 2 — Frontend
cd frontend
npm install
npm run dev
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `FRED_API_KEY` | API key from [FRED](https://fred.stlouisfed.org/docs/api/api_key.html) |

## API Endpoints (FastAPI mode)

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/predictions` | All model predictions |
| GET | `/api/historical?comparison=value_growth` | Historical performance data |
| GET | `/api/feature-importance?comparison=value_growth&horizon=6m` | Feature importance |
| GET | `/api/model-metrics` | Walk-forward backtest metrics |
| POST | `/api/retrain` | Force data refresh and retraining |
