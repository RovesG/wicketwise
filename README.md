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

### Repo hygiene
- Use `.env.example` to create your `.env` (do not commit secrets)
- Outputs go under `artifacts/` (gitignored); caches under `models/aggregates/`
- Run pre-commit locally: `pre-commit install && pre-commit run -a`
- CI runs `pytest -q` on PRs
