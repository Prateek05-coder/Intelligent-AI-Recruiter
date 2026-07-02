"""
test_pipeline.py
-----------------
End-to-end smoke test:
  1. Writes out the 3 real sample candidates given in the brief.
  2. Pads the dataset with synthetic filler candidates (so there's actually
     >100 candidates to rank from — the real hackathon data is 100k, the
     sample only gave us 3, which can't exercise a "top 100" pipeline on
     its own).
  3. Runs precompute.py (offline) then rank.py (timed, must be <5 min).
  4. Runs validate_submission.py against the output and reports pass/fail.

Usage:
    python test_pipeline.py
"""

import json
import random
import subprocess
import sys
import time
from pathlib import Path

WORKDIR = Path(__file__).parent
DATA_PATH = WORKDIR / "test_candidates.jsonl"
ARTIFACTS_PATH = WORKDIR / "test_artifacts.pkl"
OUTPUT_PATH = WORKDIR / "test_output.csv"

N_SYNTHETIC = 147  # + 3 real = 150 total, comfortably over top-100

# --------------------------------------------------------------------------
# The 3 real sample candidates from the brief (verbatim)
# --------------------------------------------------------------------------
REAL_CANDIDATES = [
    {"candidate_id": "CAND_0000002", "profile": {"anonymized_name": "Saanvi Sethi", "headline": "Operations Manager | 12.5+ yrs experience", "summary": "Professional with 12.5+ years of experience. My professional background is in marketing manager. Lately I've been curious about AI tools.", "location": "Chennai, Tamil Nadu", "country": "India", "years_of_experience": 12.5, "current_title": "Operations Manager", "current_company": "Wipro", "current_company_size": "10001+", "current_industry": "IT Services"}, "career_history": [{"company": "Wipro", "title": "Operations Manager", "start_date": "2022-11-14", "end_date": None, "duration_months": 43, "is_current": True, "industry": "IT Services", "company_size": "10001+", "description": "Customer support team lead at a SaaS product."}], "education": [{"institution": "Local Engineering College", "degree": "B.Sc", "field_of_study": "Mathematics", "start_year": 2007, "end_year": 2011, "grade": "77%", "tier": "tier_4"}], "skills": [{"name": "Project Management", "proficiency": "intermediate", "endorsements": 14, "duration_months": 23}], "certifications": [], "languages": [{"language": "English", "proficiency": "professional"}], "redrob_signals": {"profile_completeness_score": 78.7, "signup_date": "2025-07-28", "last_active_date": "2025-11-12", "open_to_work_flag": True, "profile_views_received_30d": 7, "applications_submitted_30d": 1, "recruiter_response_rate": 0.29, "avg_response_time_hours": 171.6, "skill_assessment_scores": {}, "connection_count": 179, "endorsements_received": 3, "notice_period_days": 60, "expected_salary_range_inr_lpa": {"min": 8.8, "max": 9.0}, "preferred_work_mode": "flexible", "willing_to_relocate": False, "github_activity_score": -1, "search_appearance_30d": 107, "saved_by_recruiters_30d": 10, "interview_completion_rate": 0.62, "offer_acceptance_rate": -1, "verified_email": False, "verified_phone": False, "linkedin_connected": False}},
    {"candidate_id": "CAND_0000010", "profile": {"anonymized_name": "Aarav Kapoor", "headline": "Data Engineer | Data pipelines & analytics", "summary": "Software / data professional with 4.6 years of experience building data pipelines, backend systems, and analytics infrastructure.", "location": "London", "country": "UK", "years_of_experience": 4.6, "current_title": "Data Engineer", "current_company": "Ola", "current_company_size": "5001-10000", "current_industry": "Transportation"}, "career_history": [{"company": "Ola", "title": "Data Engineer", "start_date": "2021-11-19", "end_date": None, "duration_months": 55, "is_current": True, "industry": "Transportation", "company_size": "5001-10000", "description": "Mixed data science and analytics-engineering role. Built our experimentation framework that supports the product team's A/B tests."}], "education": [{"institution": "Generic State University", "degree": "B.E.", "field_of_study": "Mathematics", "start_year": 2007, "end_year": 2011, "grade": "85%", "tier": "tier_4"}], "skills": [{"name": "Python", "proficiency": "intermediate", "endorsements": 7, "duration_months": 14}, {"name": "BM25", "proficiency": "advanced", "endorsements": 55, "duration_months": 55}], "certifications": [], "languages": [{"language": "English", "proficiency": "native"}], "redrob_signals": {"profile_completeness_score": 81.6, "signup_date": "2026-01-09", "last_active_date": "2026-04-29", "open_to_work_flag": False, "profile_views_received_30d": 60, "applications_submitted_30d": 13, "recruiter_response_rate": 0.4, "avg_response_time_hours": 19.0, "skill_assessment_scores": {"BM25": 81.3}, "connection_count": 712, "endorsements_received": 38, "notice_period_days": 120, "expected_salary_range_inr_lpa": {"min": 13.0, "max": 32.0}, "preferred_work_mode": "hybrid", "willing_to_relocate": False, "github_activity_score": 33.7, "search_appearance_30d": 256, "saved_by_recruiters_30d": 2, "interview_completion_rate": 0.53, "offer_acceptance_rate": -1, "verified_email": True, "verified_phone": True, "linkedin_connected": False}},
    {"candidate_id": "CAND_0000031", "profile": {"anonymized_name": "Ela Singh", "headline": "Recommendation Systems Engineer | Search, Ranking & Retrieval", "summary": "Machine learning engineer with 6.0 years of experience building ML-powered features in production. Strong background in NLP, recommendation systems, and applied AI.", "location": "Hyderabad, Telangana", "country": "India", "years_of_experience": 6.0, "current_title": "Recommendation Systems Engineer", "current_company": "Swiggy", "current_company_size": "5001-10000", "current_industry": "Food Delivery"}, "career_history": [{"company": "Swiggy", "title": "Recommendation Systems Engineer", "start_date": "2025-04-02", "end_date": None, "duration_months": 14, "is_current": True, "industry": "Food Delivery", "company_size": "5001-10000", "description": "Trained and shipped multiple ranking models for our product's discovery feed using XGBoost and LightGBM. Designed features across three families: content metadata, user behavior signals, and item engagement history."}], "education": [{"institution": "SRM University", "degree": "M.Tech", "field_of_study": "Computer Engineering", "start_year": 2002, "end_year": 2006, "grade": "9.16 CGPA", "tier": "tier_2"}], "skills": [{"name": "Pinecone", "proficiency": "expert", "endorsements": 34, "duration_months": 88}, {"name": "Information Retrieval", "proficiency": "expert", "endorsements": 2, "duration_months": 84}], "certifications": [], "languages": [{"language": "English", "proficiency": "native"}], "redrob_signals": {"profile_completeness_score": 83.4, "signup_date": "2026-01-28", "last_active_date": "2026-05-24", "open_to_work_flag": True, "profile_views_received_30d": 194, "applications_submitted_30d": 2, "recruiter_response_rate": 0.91, "avg_response_time_hours": 76.1, "skill_assessment_scores": {"Pinecone": 53.6}, "connection_count": 832, "endorsements_received": 177, "notice_period_days": 60, "expected_salary_range_inr_lpa": {"min": 27.3, "max": 60.2}, "preferred_work_mode": "flexible", "willing_to_relocate": True, "github_activity_score": 32.6, "search_appearance_30d": 778, "saved_by_recruiters_30d": 13, "interview_completion_rate": 0.6, "offer_acceptance_rate": 0.38, "verified_email": False, "verified_phone": True, "linkedin_connected": False}},
]

TITLES = ["ML Engineer", "Backend Engineer", "Data Scientist", "Search Engineer",
          "Applied Scientist", "Software Engineer", "DevOps Engineer", "QA Engineer"]
COMPANIES_PRODUCT = ["Flipkart", "Zomato", "Razorpay", "Meesho", "CRED", "Google",
                      "Amazon", "Uber", "Freshworks"]
COMPANIES_CONSULTING = ["TCS", "Infosys", "Wipro", "Accenture", "Cognizant", "Capgemini"]
LOCATIONS = ["Pune, Maharashtra", "Noida, UP", "Bangalore, Karnataka", "Chennai, Tamil Nadu",
             "Remote", "Mumbai, Maharashtra"]
SKILL_POOL = ["Python", "FAISS", "BM25", "NDCG", "XGBoost", "Vector DB", "Pinecone",
              "Weaviate", "Docker", "Kubernetes", "SQL", "PyTorch", "LangChain",
              "Computer Vision", "Speech Recognition"]
SUMMARY_TEMPLATES = [
    "Engineer with {yrs} years of experience in {skill1} and {skill2}, worked on retrieval and ranking systems.",
    "{yrs} years building backend systems, dabbling in {skill1}.",
    "Data professional, {yrs} yrs experience, focused on {skill1} and hybrid search with {skill2}.",
    "Consultant with {yrs} years at services companies, recently exploring {skill1}.",
    "Computer vision specialist, {yrs} years, expert in {skill1}.",
]


def make_synthetic_candidate(i: int, rng: random.Random) -> dict:
    # FIXED: Pattern must be strictly CAND_XXXXXXX (7 digits total)
    cid = f"CAND_9{i:06d}"
    yrs = round(rng.uniform(0.5, 15), 1)
    is_consulting = rng.random() < 0.25
    company = rng.choice(COMPANIES_CONSULTING) if is_consulting else rng.choice(COMPANIES_PRODUCT)
    skill1, skill2 = rng.sample(SKILL_POOL, 2)
    summary = rng.choice(SUMMARY_TEMPLATES).format(yrs=yrs, skill1=skill1, skill2=skill2)
    title = rng.choice(TITLES)
    loc = rng.choice(LOCATIONS)

    honeypot = rng.random() < 0.03
    signals = {
        "profile_completeness_score": round(rng.uniform(40, 99), 1),
        "signup_date": "2025-09-01",
        "last_active_date": f"2026-{rng.randint(1,6):02d}-{rng.randint(1,28):02d}",
        "open_to_work_flag": rng.random() < 0.5,
        "profile_views_received_30d": rng.randint(0, 300) if not honeypot else 50000,
        "applications_submitted_30d": rng.randint(0, 20) if not honeypot else 900,
        "recruiter_response_rate": round(rng.uniform(0, 1), 2) if not honeypot else 1.0,
        "avg_response_time_hours": round(rng.uniform(1, 200), 1),
        "skill_assessment_scores": {},
        "connection_count": rng.randint(0, 1000),
        "endorsements_received": rng.randint(0, 200),
        "notice_period_days": rng.choice([15, 30, 45, 60, 90]),
        "expected_salary_range_inr_lpa": {"min": 5.0, "max": 20.0},
        "preferred_work_mode": rng.choice(["remote", "hybrid", "flexible"]),
        "willing_to_relocate": rng.random() < 0.5,
        "github_activity_score": round(rng.uniform(0, 50), 1),
        "search_appearance_30d": rng.randint(0, 500),
        "saved_by_recruiters_30d": rng.randint(0, 20),
        "interview_completion_rate": round(rng.uniform(0, 1), 2) if not honeypot else 1.0,
        "offer_acceptance_rate": round(rng.uniform(0, 1), 2) if not honeypot else 1.0,
        "verified_email": rng.random() < 0.7,
        "verified_phone": rng.random() < 0.5,
        "linkedin_connected": rng.random() < 0.4,
    }

    return {
        "candidate_id": cid,
        "profile": {
            "anonymized_name": f"Synthetic Candidate {i}",
            "headline": title,
            "summary": summary,
            "location": loc,
            "country": "India",
            "years_of_experience": yrs,
            "current_title": title,
            "current_company": company,
            "current_company_size": "1001-5000",
            "current_industry": "IT Services" if is_consulting else "Product",
        },
        "career_history": [{
            "company": company, "title": title, "start_date": "2022-01-01",
            "end_date": None, "duration_months": 24, "is_current": True,
            "industry": "IT Services" if is_consulting else "Product",
            "company_size": "1001-5000", "description": summary,
        }],
        "education": [],
        "skills": [{"name": skill1, "proficiency": "advanced", "endorsements": 5, "duration_months": 12},
                    {"name": skill2, "proficiency": "intermediate", "endorsements": 2, "duration_months": 6}],
        "certifications": [],
        "languages": [{"language": "English", "proficiency": "professional"}],
        "redrob_signals": signals,
    }


def build_test_dataset():
    rng = random.Random(42)
    records = list(REAL_CANDIDATES)
    for i in range(N_SYNTHETIC):
        records.append(make_synthetic_candidate(i, rng))
    rng.shuffle(records)
    with open(DATA_PATH, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    return len(records)


def run(cmd, label):
    print(f"\n=== {label} ===")
    print("$ " + " ".join(cmd))
    t0 = time.time()
    result = subprocess.run(cmd, cwd=WORKDIR, capture_output=True, text=True)
    elapsed = time.time() - t0
    print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    print(f"[{label}] exit_code={result.returncode} elapsed={elapsed:.2f}s")
    return result.returncode, elapsed


def main():
    print("Building synthetic test dataset (3 real + %d synthetic)..." % N_SYNTHETIC)
    n = build_test_dataset()
    print(f"Wrote {n} candidates to {DATA_PATH}")

    rc, _ = run([sys.executable, "precompute.py",
                 "--corpus", str(DATA_PATH), "--out", str(ARTIFACTS_PATH)],
                "precompute.py")
    if rc != 0:
        print("FAIL: precompute.py failed")
        sys.exit(1)

    rc, rank_elapsed = run([sys.executable, "rank.py",
                             "--input", str(DATA_PATH),
                             "--artifacts", str(ARTIFACTS_PATH),
                             "--output", str(OUTPUT_PATH),
                             "--top-k", "100"],
                            "rank.py")
    if rc != 0:
        print("FAIL: rank.py failed")
        sys.exit(1)

    if rank_elapsed > 300:
        print(f"FAIL: rank.py exceeded 5-minute budget ({rank_elapsed:.1f}s)")
        sys.exit(1)
    else:
        print(f"OK: rank.py finished in {rank_elapsed:.2f}s (< 300s budget)")

    # FIXED: The validate_submission.py script only expects one positional argument (the CSV path)
    rc, _ = run([sys.executable, "validate_submission.py", str(OUTPUT_PATH)],
                "validate_submission.py")

    if rc == 0:
        print("\n✅ PIPELINE TEST PASSED — output is a valid, on-time, top-100 submission.")
        # Show top 5 for a sanity-check glance
        with open(OUTPUT_PATH, encoding="utf-8") as f:
            lines = f.readlines()
        print("\nTop 5 rows:")
        for line in lines[:6]:
            print("  " + line.strip())
    else:
        print("\n❌ PIPELINE TEST FAILED — see validation errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
