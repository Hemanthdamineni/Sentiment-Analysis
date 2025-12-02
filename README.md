# Smartwatch Sentiment (Backend + Frontend)

## Structure

- `backend`: FastAPI app and modular ML code
  - `app/` – API entry and routers (`/models`, `/comparison`, `/wordclouds`, `/predict`)
  - `scripts/` – training (`pipeline/train.py`), evaluation (`pipeline/evaluate.py`), classical ML, transformers, data processing, evaluation
  - `models/sentiment/` – saved transformer, classical models, and plots
  - `dataset/smartwatch_reviews.csv` – cached filtered dataset
- `v5/frontend`: React + Tailwind UI (dev server via Vite)

## Run

- Pipeline (train + evaluate):
  - `pixi run pipeline` (from project root) or `cd v5 && pixi run pipeline`
  - Artifacts saved under `backend/models/sentiment/`

- Backend API (port `8000`):
  - `pixi run backend` (run from `v5` directory)
  - Or `python -m uvicorn backend.app.main:app --reload --port 8000`

- Frontend dev:
  - `npm --prefix v5/frontend install`
  - `npm --prefix v5/frontend run dev`
  - Dev server on `http://localhost:5173/` or `http://localhost:5174/`

## API

- `GET /models` → `{ classical_algorithms: [...] }`
- `GET /comparison` → returns `backend/models/sentiment/results/comparison.json`
- `GET /wordclouds` → `{ wordclouds: ["/static/results/wordcloud_negative.png", ...] }`
- `POST /predict` → body `{ text, model_type: 'transformer'|'classical', algorithm? }`

## Troubleshooting

- Frontend blank page:
  - Ensure React import exists and Vite React plugin is configured (`v5/frontend/vite.config.js`).
  - Use React DevTools to inspect state/props.

- Failed to fetch from frontend:
  - Start backend: `pixi run backend` in `v5`.
  - Align hosts: use `localhost` consistently for both backend and frontend.

- Qt plugin error in pipeline:
  - Plotting uses non-GUI backend (`Agg`) to avoid Qt.
  - Rerun: `pixi run pipeline`.

## Generated Outputs

- Transformer: `backend/models/sentiment/best_model.pt`, `pytorch_model.bin`
- Classical: `backend/models/sentiment/classical/*.pkl`
- Plots: `backend/models/sentiment/results/*.png`
- Comparison: `backend/models/sentiment/results/comparison.json`
