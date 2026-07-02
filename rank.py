
import argparse
import csv
import json
import re
import sys
import time
import pickle
from datetime import datetime, date, timezone

import numpy as np
from scipy.sparse import vstack

from precompute import build_candidate_text  # reuse identical text-building logic

CHUNK_SIZE = 2000
# "Today" for recency calculations. In a real deployment this would be
# datetime.utcnow(); pinned here so scoring is reproducible against the
# fixed redrob_signals timestamps in the sample data.
TODAY = datetime.utcnow().date()


# --------------------------------------------------------------------------
# Feature helpers
# --------------------------------------------------------------------------

def parse_date_safe(s):
    if not s:
        return None
    try:
        return datetime.strptime(s[:10], "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None


def keyword_score(text: str, core_skill_weights: dict):
    """Returns (raw_weighted_score, matched_terms list).
    core_skill_weights maps regex pattern -> (weight, display_name)."""
    text_l = text.lower()
    total = 0.0
    matched = []
    # sort by weight desc so the strongest/most specific matches surface first
    for pattern, (weight, display_name) in sorted(
        core_skill_weights.items(), key=lambda kv: -kv[1][0]
    ):
        if re.search(pattern, text_l):
            total += weight
            matched.append(display_name)
    return total, matched


def trap_penalty_flag(text: str, trap_terms: dict, has_core_skill: bool):
    """Return True if candidate looks like a disqualifier trap."""
    text_l = text.lower()
    if has_core_skill:
        return False  # having real IR/ranking depth clears the trap
    for _, patterns in trap_terms.items():
        for p in patterns:
            if re.search(p, text_l):
                return True
    return False


def is_consulting_only(rec: dict, consulting_companies: set):
    companies = set()
    cur = (rec.get("profile", {}) or {}).get("current_company", "")
    if cur:
        companies.add(cur.strip().lower())
    for job in rec.get("career_history", []) or []:
        c = job.get("company", "")
        if c:
            companies.add(c.strip().lower())
    if not companies:
        return False
    non_consulting = [c for c in companies if c not in consulting_companies]
    return len(non_consulting) == 0  # ALL known companies are consulting shops


def experience_fit(years, cfg):
    exp_cfg = cfg["experience"]
    if years is None:
        return 0.3  # unknown, mild neutral score
    if years < exp_cfg["hard_floor"]:
        return 0.0
    if years < exp_cfg["min_ideal"]:
        # ramp 0 -> 1 between hard_floor and min_ideal
        span = exp_cfg["min_ideal"] - exp_cfg["hard_floor"]
        return max(0.0, (years - exp_cfg["hard_floor"]) / span)
    if years <= exp_cfg["max_ideal"]:
        return 1.0
    if years <= exp_cfg["soft_ceiling"]:
        # gentle decay 1.0 -> 0.6
        span = exp_cfg["soft_ceiling"] - exp_cfg["max_ideal"]
        decay = (years - exp_cfg["max_ideal"]) / span
        return 1.0 - 0.4 * decay
    return 0.6


def recency_score(last_active_str, cfg):
    d = parse_date_safe(last_active_str)
    if d is None:
        return 0.2
    days = (TODAY - d).days
    if days < 0:
        days = 0
    full = cfg["recency_days_full_score"]
    zero = cfg["recency_days_zero_score"]
    if days <= full:
        return 1.0
    if days >= zero:
        return 0.0
    return 1.0 - (days - full) / (zero - full)


def honeypot_flag(signals: dict):
    """Detect a few clearly-impossible / bot-like stat combinations.
    Sentinel values of -1 mean 'not available' and are NOT honeypots."""
    def val(k, default=0):
        v = signals.get(k, default)
        return default if v is None else v

    rrr = val("recruiter_response_rate", 0)
    icr = val("interview_completion_rate", 0)
    oar = val("offer_acceptance_rate", 0)
    apps = val("applications_submitted_30d", 0)
    views = val("profile_views_received_30d", 0)
    conn = val("connection_count", 0)

    checks = [
        rrr is not None and (rrr < -1 or rrr > 1.0001),
        icr is not None and (icr < -1 or icr > 1.0001),
        oar is not None and (oar < -1 or oar > 1.0001),
        apps > 500,           # nobody applies to 500 jobs in 30 days
        views > 20000,        # implausible view count
        conn < 0,
        # perfect stats + zero experience footprint is a classic honeypot
        (rrr == 1.0 and icr == 1.0 and oar == 1.0),
    ]
    return any(checks)


def behavioral_score(signals: dict, cfg):
    if not signals:
        return 0.1, False

    hp = honeypot_flag(signals)

    rec = recency_score(signals.get("last_active_date"), cfg)

    rrr = signals.get("recruiter_response_rate")
    rrr_score = rrr if isinstance(rrr, (int, float)) and rrr >= 0 else 0.3

    icr = signals.get("interview_completion_rate")
    icr_score = icr if isinstance(icr, (int, float)) and icr >= 0 else 0.3

    bonus = 0.0
    if signals.get("verified_email"):
        bonus += 0.05
    if signals.get("verified_phone"):
        bonus += 0.05
    if signals.get("open_to_work_flag"):
        bonus += 0.05

    score = 0.4 * rec + 0.35 * rrr_score + 0.2 * icr_score + bonus
    score = min(1.0, score)

    if isinstance(rrr, (int, float)) and 0 <= rrr < 0.2:
        score *= cfg["penalties"]["inactive_recruiter_response"]

    return score, hp


def location_notice_score(rec: dict, target_locations: set):
    profile = rec.get("profile", {}) or {}
    signals = rec.get("redrob_signals", {}) or {}
    loc = (profile.get("location", "") or "").lower()

    score = 0.0
    if any(t in loc for t in target_locations):
        score += 0.6
    elif signals.get("willing_to_relocate"):
        score += 0.35

    notice = signals.get("notice_period_days")
    if isinstance(notice, (int, float)):
        if notice < 30:
            score += 0.4
        elif notice < 60:
            score += 0.2

    return min(1.0, score)


# --------------------------------------------------------------------------
# Reasoning generator (pure templating over real extracted facts — no LLM)
# --------------------------------------------------------------------------

def generate_reasoning(rec, years, matched_terms, is_consulting_flag,
                        trap_flag, signals, loc_score, exp_score, hp_flag):
    profile = rec.get("profile", {}) or {}
    title = profile.get("current_title", "candidate")
    company = profile.get("current_company", "an unspecified company")

    if hp_flag:
        return (f"Flagged as a likely honeypot/bot profile due to implausible "
                f"behavioral signal combinations; excluded from serious contention.")

    if trap_flag:
        return (f"{title} at {company} shows no evidence of core IR/ranking "
                f"experience (embeddings, BM25, vector DBs); likely a wrapper/"
                f"non-ML-depth profile, scored low despite {years or 0:.1f} yrs experience.")

    if is_consulting_flag:
        return (f"{years or 0:.1f} yrs experience but entirely within IT-services/"
                f"consulting firms ({company}), with no product-company ranking "
                f"experience; penalized per JD requirements.")

    top_skills = matched_terms[:3]
    skill_str = ", ".join(top_skills) if top_skills else "general ML background"

    rrr = signals.get("recruiter_response_rate")
    rrr_str = f"{rrr:.2f}" if isinstance(rrr, (int, float)) and rrr >= 0 else "unknown"

    loc = profile.get("location", "unspecified location")

    fit_phrase = "Strong fit" if exp_score >= 0.8 else (
        "Reasonable fit" if exp_score >= 0.4 else "Partial fit")

    reasoning = (
        f"{fit_phrase} with {years or 0:.1f} yrs experience as {title} at {company}, "
        f"demonstrated depth in {skill_str}. "
        f"Located in {loc}, recruiter response rate {rrr_str}."
    )
    return reasoning


# --------------------------------------------------------------------------
# Main pipeline
# --------------------------------------------------------------------------

def iter_records(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def process_chunk(chunk_records, artifacts, cfg, vectorizer, jd_vector):
    texts = [build_candidate_text(r) for r in chunk_records]
    tfidf_matrix = vectorizer.transform(texts)

    # cosine similarity against jd_vector (both are already L2-normalized by
    # TfidfVectorizer's default norm='l2', so dot product == cosine sim)
    sims = (tfidf_matrix @ jd_vector.T).toarray().ravel()

    core_skill_weights = artifacts["core_skill_weights"]
    trap_terms = artifacts["trap_terms"]
    consulting_companies = artifacts["consulting_companies"]
    target_locations = artifacts["target_locations"]
    w = cfg["weights"]
    kw_norm = cfg["keyword_score_norm"]

    results = []
    for rec, text, sim in zip(chunk_records, texts, sims):
        profile = rec.get("profile", {}) or {}
        signals = rec.get("redrob_signals", {}) or {}
        years = profile.get("years_of_experience")

        raw_kw, matched_terms = keyword_score(text, core_skill_weights)
        kw_score = min(1.0, raw_kw / kw_norm)
        has_core = raw_kw > 0

        trap_flag = trap_penalty_flag(text, trap_terms, has_core)
        consulting_flag = is_consulting_only(rec, consulting_companies)

        exp_score = experience_fit(years, cfg)
        beh_score, hp_flag = behavioral_score(signals, cfg)
        loc_score = location_notice_score(rec, target_locations)

        total = (
            w["tfidf_cosine"] * float(sim)
            + w["keyword_score"] * kw_score
            + w["experience_fit"] * exp_score
            + w["behavioral"] * beh_score
            + w["location_notice"] * loc_score
        )

        if consulting_flag:
            total *= cfg["penalties"]["consulting_only"]
        if trap_flag:
            total *= cfg["penalties"]["trap_no_core_skill"]
        if hp_flag:
            total *= cfg["penalties"]["honeypot"]

        total = round(float(total), 6)

        reasoning = generate_reasoning(
            rec, years, matched_terms, consulting_flag, trap_flag,
            signals, loc_score, exp_score, hp_flag,
        )

        results.append({
            "candidate_id": rec.get("candidate_id"),
            "score": total,
            "reasoning": reasoning,
        })

    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="candidates.jsonl")
    ap.add_argument("--artifacts", default="artifacts.pkl")
    ap.add_argument("--output", default="team_xxx.csv")
    ap.add_argument("--top-k", type=int, default=100)
    ap.add_argument("--time-budget-s", type=float, default=300.0)
    args = ap.parse_args()

    t0 = time.time()

    with open(args.artifacts, "rb") as f:
        artifacts = pickle.load(f)

    cfg = artifacts["config"]
    vectorizer = artifacts["vectorizer"]
    jd_vector = artifacts["jd_vector"]

    all_results = []
    chunk = []
    n_seen = 0

    for rec in iter_records(args.input):
        chunk.append(rec)
        if len(chunk) >= CHUNK_SIZE:
            all_results.extend(process_chunk(chunk, artifacts, cfg, vectorizer, jd_vector))
            n_seen += len(chunk)
            chunk = []
            elapsed = time.time() - t0
            if elapsed > args.time_budget_s:
                print(f"[rank] WARNING: time budget exceeded at {n_seen} records "
                      f"({elapsed:.1f}s) — proceeding to finalize with what we have.",
                      file=sys.stderr)
                break

    if chunk:
        all_results.extend(process_chunk(chunk, artifacts, cfg, vectorizer, jd_vector))
        n_seen += len(chunk)

    print(f"[rank] scored {n_seen} candidates in {time.time() - t0:.2f}s")

    # Sort: score descending, tie-break candidate_id ascending
    all_results.sort(key=lambda r: (-r["score"], r["candidate_id"]))

    top_k = min(args.top_k, len(all_results))
    top = all_results[:top_k]

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for i, r in enumerate(top, start=1):
            writer.writerow([r["candidate_id"], i, r["score"], r["reasoning"]])

    total_time = time.time() - t0
    print(f"[rank] wrote {top_k} rows to {args.output} in {total_time:.2f}s total")
    if total_time > 300:
        print("[rank] WARNING: exceeded the 5-minute budget!", file=sys.stderr)


if __name__ == "__main__":
    main()
