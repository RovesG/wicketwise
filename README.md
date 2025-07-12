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