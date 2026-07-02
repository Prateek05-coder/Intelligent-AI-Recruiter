
import argparse
import json
import pickle
import re
import sys
import time

from sklearn.feature_extraction.text import TfidfVectorizer

JOB_DESCRIPTION = """
Senior AI Engineer - Founding Team at Redrob.
We need a shipper, not a pure researcher: someone who ships ranking and
retrieval systems fast. Deep experience with embeddings, retrieval, BM25,
dense retrieval, hybrid search, learning to rank, XGBoost, ranking models,
vector databases, Pinecone, FAISS, Weaviate, approximate nearest neighbor
search, evaluation frameworks, NDCG, MAP, offline and online evaluation.
4-5+ years of applied machine learning and AI experience at a product
company. Location Pune or Noida, or willing to relocate. Short notice
period preferred.
"""

CORE_SKILL_WEIGHTS = {
    r"\bembeddings?\b": (3.0, "embeddings"),
    r"\bretrieval\b": (3.0, "retrieval"),
    r"\bbm25\b": (3.5, "BM25"),
    r"\bdense retrieval\b": (3.5, "dense retrieval"),
    r"\bhybrid (retrieval|search)\b": (3.5, "hybrid search"),
    r"\blearning to rank\b": (3.5, "learning to rank"),
    r"\bltr\b": (3.0, "LTR"),
    r"\bxgboost\b": (3.0, "XGBoost"),
    r"\blightgbm\b": (2.0, "LightGBM"),
    r"\branking\b": (2.5, "ranking"),
    r"\bvector (db|database|databases)\b": (3.0, "vector DB"),
    r"\bfaiss\b": (3.5, "FAISS"),
    r"\bpinecone\b": (3.5, "Pinecone"),
    r"\bweaviate\b": (3.5, "Weaviate"),
    r"\bann\b|\bapproximate nearest neighbou?r\b": (2.5, "ANN search"),
    r"\bndcg\b": (3.5, "NDCG"),
    r"\bmap@?\d*\b": (2.5, "MAP"),
    r"\binformation retrieval\b": (3.0, "information retrieval"),
    r"\boffline eval(uation)?\b": (2.0, "offline evaluation"),
    r"\bonline eval(uation)?\b|\bab test(ing)?\b|\ba/b test(ing)?\b": (2.0, "online evaluation/A-B testing"),
    r"\brecommendation( system| engine)?s?\b": (2.0, "recommendation systems"),
    r"\bsearch (relevance|infrastructure|systems?)\b": (2.5, "search infrastructure"),
    r"\bnlp\b|\bnatural language processing\b": (1.5, "NLP"),
}

TRAP_TERMS = {
    "langchain_wrapper": [r"\blangchain\b", r"\bprompt engineering\b", r"\bllm wrapper\b"],
    "cv_speech_only": [
        r"\bcomputer vision\b", r"\bobject detection\b", r"\bimage classification\b",
        r"\bspeech recognition\b", r"\basr\b", r"\bocr\b",
    ],
    "pure_academic": [r"\bphd thesis\b", r"\bpublished paper\b", r"\bacademic research\b"],
}

CONSULTING_COMPANIES = {
    "tcs", "tata consultancy services", "wipro", "infosys", "accenture",
    "cognizant", "capgemini", "hcl", "hcltech", "tech mahindra", "ibm services",
    "mindtree", "lti", "l&t infotech", "genpact", "mphasis", "hexaware",
}

TARGET_LOCATIONS = {"pune", "noida"}

CONFIG = {
    "weights": {
        "tfidf_cosine": 25.0,      
        "keyword_score": 35.0,     
        "experience_fit": 15.0,    
        "behavioral": 20.0,        
        "location_notice": 5.0,    
    },
    "penalties": {
        "consulting_only": 0.5,        
        "trap_no_core_skill": 0.6,     
        "honeypot": 0.15,              
        "inactive_recruiter_response": 0.7,  
    },
    "experience": {
        "min_ideal": 4.0,
        "max_ideal": 9.0,
        "hard_floor": 1.5,   
        "soft_ceiling": 16.0,  
    },
    "keyword_score_norm": 12.0,  
    "recency_days_full_score": 30,
    "recency_days_zero_score": 180,
}


def build_candidate_text(rec: dict) -> str:
    """Concatenate the text fields we want the vectorizer to see."""
    parts = []
    profile = rec.get("profile", {}) or {}
    parts.append(profile.get("headline", "") or "")
    parts.append(profile.get("summary", "") or "")
    parts.append(profile.get("current_title", "") or "")

    for job in rec.get("career_history", []) or []:
        parts.append(job.get("title", "") or "")
        parts.append(job.get("description", "") or "")

    for sk in rec.get("skills", []) or []:
        parts.append(sk.get("name", "") or "")

    return " ".join(parts)


def load_corpus(path: str):
    texts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            texts.append(build_candidate_text(rec))
    return texts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="candidates.jsonl",
                     help="JSONL file to fit the TF-IDF vocabulary on")
    ap.add_argument("--out", default="artifacts.pkl")
    args = ap.parse_args()

    t0 = time.time()
    print(f"[precompute] loading corpus from {args.corpus} ...")
    texts = load_corpus(args.corpus)
    if not texts:
        print("[precompute] ERROR: empty corpus", file=sys.stderr)
        sys.exit(1)
    print(f"[precompute] {len(texts)} candidate documents loaded")

    min_df = 2 if len(texts) >= 50 else 1
    max_df = 0.95

    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        stop_words="english",
        min_df=min_df,
        max_df=max_df,
        sublinear_tf=True,
    )
    vectorizer.fit(texts)
    print(f"[precompute] vocabulary size: {len(vectorizer.vocabulary_)}")

    jd_vector = vectorizer.transform([JOB_DESCRIPTION])

    artifacts = {
        "vectorizer": vectorizer,
        "jd_vector": jd_vector,
        "core_skill_weights": CORE_SKILL_WEIGHTS,
        "trap_terms": TRAP_TERMS,
        "consulting_companies": CONSULTING_COMPANIES,
        "target_locations": TARGET_LOCATIONS,
        "config": CONFIG,
        "job_description": JOB_DESCRIPTION,
        "built_at": time.time(),
        "corpus_size": len(texts),
    }

    with open(args.out, "wb") as f:
        pickle.dump(artifacts, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"[precompute] wrote {args.out} in {time.time() - t0:.2f}s")


if __name__ == "__main__":
    main()
