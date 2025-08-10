# 🏏 Crickformers

**Crickformers** is a hybrid deep learning system for predicting T20 cricket outcomes — one ball at a time.

Combining Transformers, Graph Neural Networks, structured ball-by-ball data, video signals, and betting market inputs, Crickformers powers next-gen cricket intelligence, from tactical insights to shadow betting strategies.

---

## 🚀 Key Features

- **Ball-by-ball prediction engine** using Transformer encoders
- **GNN-enhanced player embeddings** from a cricket knowledge graph
- **Multi-modal input fusion**: tabular, video, and betting market data
- **Prediction heads** for next-ball outcome, win probability, and odds mispricing
- **Support for live inference + post-match learning updates**

---

## 📦 Data Sources

Crickformers ingests and fuses data from:

1. **Decimal CSV** – Ball-by-ball structured match data  
2. **Partner Video Feed** – 40+ player/body signals extracted per ball  
3. **Scraped Video** – Backup/complementary CV signals when official feed isn’t available  
4. **Betfair Markets** – Odds, liquidity, and volume per ball

---

## 🧠 Model Architecture

- **Sequence Encoder** – Transformer block over recent balls
- **Context Encoder** – Static game state + video features
- **KG Attention Module** – Attention over pretrained GNN embeddings (batter, bowler type, venue)
- **Fusion Layer** – Combines all sources for final latent state
- **Prediction Heads**:
  - `next_ball_outcome` (classification)
  - `win_probability` (regression)
  - `odds_mispricing` (binary)

---

## 🛠️ Project Structure

```bash
crickformers/
├── data/               # Ingestion, alignment, and preprocessing
├── gnn/                # KG building, GNN training, embedding generation
├── model/              # Transformer model components
├── training/           # Training loops, validation, metrics
├── inference/          # Live prediction + shadow betting wrapper
├── utils/              # Feature mapping, video sync, odds processing
├── notebooks/          # Experiments and ablation analysis
└── README.md
``` 

This will start a local web server and open the dashboard in your browser. You can then interact with the UI, load the sample data, and see the model's predictions and the agent's decisions.

I will now commit the new UI and sample data to the repository. 

---

## ▶️ Run the WicketWise UI (Figma build)

The current UI is a Figma-derived static page served alongside the Flask admin API.

- Backend API: Flask on port 5001
- UI: Static server on port 8000

### Quick start

```bash
# one-time setup (no Streamlit)
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# start both backend (5001) and UI server (8000)
bash start.sh

# open the correct UI
open "http://127.0.0.1:8000/wicketwise_admin_fixed.html"  # macOS
# or visit in your browser
# http://127.0.0.1:8000/wicketwise_admin_fixed.html

# backend health check
curl http://127.0.0.1:5001/api/health
```

Notes:
- Port 5001 is API-only; `/` returns 404 by design in this branch.
- The page `wicketwise_admin_fixed.html` contains the main dashboard (fake video window, player cards, KG query) and admin flows.

### Admin panel actions
- Build Knowledge Graph: runs the scalable, chunked KG pipeline and finishes with node/edge counts.
- Train GNN: trains and saves `models/gnn_embeddings.pt`.
- Train Model: runs the Crickformer training path.
- Aligner Strategy: choose DNA/Exact/Hybrid/LLM in the Knowledge Graph tab; saved to backend and used by align workflows.

### Backend API highlights
- `GET /api/health`: health check
- `POST /api/build-knowledge-graph`: build KG (chunked pipeline)
- `GET|POST /api/kg-settings`: chunk_size, cache_dir, use_llm_schema_hint, compute_heavy_metrics, normalize_ids
- `POST /api/kg-cache/purge`: clear aggregate caches
- `GET|POST /api/aligner-settings`: get/set aligner strategy (`dna|exact|hybrid|llm`)

### Development
- Environment: copy `.env.example` → `.env` and set keys/paths locally (not committed)
- Pre-commit: `pip install pre-commit && pre-commit install && pre-commit run -a`
- Tests: `PYTHONPATH=. pytest -q` (CI runs on push/PR)
- CI: GitHub Actions workflow at `.github/workflows/ci.yml`

### What changed recently
- Repo hygiene: `.env.example`, `artifacts/`, expanded `.gitignore`, README updates
- KG: scalable chunked pipeline, vectorized aggregations, cached aggregates, degree-only metrics by default
- UI: Aligner selector in admin modal; backend endpoints for settings
- Inference scaffolding: EmbeddingService, Policy module (fractional Kelly), hierarchical model placeholder
- Tooling: pre-commit (black, ruff) and CI for pytest

### Repo hygiene
- Use `.env.example` to create your `.env` (do not commit secrets)
- Outputs go under `artifacts/` (gitignored); caches under `models/aggregates/`
- Run pre-commit locally: `pre-commit install && pre-commit run -a`
- CI runs `pytest -q` on PRs
