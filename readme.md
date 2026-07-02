# Redrob Hackathon v4 - The Commit Crew

This repository contains our submission for the Redrob Intelligent Candidate Discovery & Ranking Challenge.

## Approach Summary

We built an ultra-fast, CPU-optimized hybrid ranker designed specifically to meet the strict 5-minute, 16GB, offline-only constraints.

Our pipeline uses:

1. **Semantic & Keyword Scoring:** A custom `scikit-learn` TF-IDF Vectorizer to measure semantic overlap with the Job Description, combined with targeted Regex extraction for core skills (Pinecone, FAISS, XGBoost, etc.).
2. **Behavioral Signal Modifier:** We actively penalize inactive candidates, flag honeypots, and enforce JD constraints (e.g., heavily penalizing candidates with only IT-services/consulting backgrounds and zero product-company ranking experience).
3. **Deterministic Reasoning:** The `reasoning` column is generated via a dynamic templating system based on extracted text facts, ensuring **zero hallucinations** and 100% factual alignment with the profile.

The entire ranking step runs strictly offline on a CPU, securely scoring 100,000 candidates in mere seconds.

## Setup & Dependencies

Ensure you have Python 3.11+ installed. Install the required libraries:

```bash
pip install -r requirements.txt
```
