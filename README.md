# Investment Comparison App (XGBoost)

XGBoost-powered application that predicts relative outperformance between fund pairs:

- **Value vs Growth**: VVIAX vs VIGAX
- **US vs International**: VTSAX vs VTIAX

Predictions are generated across 1, 3, 6, and 12-month horizons using macroeconomic indicators from FRED and historical fund price data.

## Architecture

- **Backend**: Python / FastAPI with XGBoost models, walk-forward validated
- **Frontend**: React / Vite with Recharts visualizations

## Setup

### Backend

```bash
cd backend
pip install -r requirements.txt
cp ../.env.example ../.env  # Add your FRED API key
uvicorn main:app --reload
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

The frontend proxies `/api` requests to the backend at `localhost:8000`.

## Environment Variables

| Variable | Description |
|----------|-------------|
| `FRED_API_KEY` | API key from [FRED](https://fred.stlouisfed.org/docs/api/api_key.html) |

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/predictions` | All model predictions |
| GET | `/api/historical?comparison=value_growth` | Historical performance data |
| GET | `/api/feature-importance?comparison=value_growth&horizon=6m` | Feature importance |
| GET | `/api/model-metrics` | Walk-forward backtest metrics |
| POST | `/api/retrain` | Force data refresh and retraining |
