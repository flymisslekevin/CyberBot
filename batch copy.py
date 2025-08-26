"""Run a batch of questions through rag_local.ask() and report the answers.

How to use
----------
    python batch_query.py                # uses QUESTIONS list below
    python batch_query.py questions.txt   # one question per line in a file

Set DEBUG = True at the top if you’d like *full* answers and extra detail.
Otherwise the script prints only a short preview of each answer to keep the
console tidy.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple
import rag_local  # relies on rag_local.ask(question) -> (answer, ctx_blocks)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DEBUG = False  # Flip to True to see full answer text for every question

# Default questions if no file is provided -----------------------------------
QUESTIONS: List[str] = [
    #basics
    "What services are covered under Medicare Part A?",
    "Does Medicare Part A have an out-of-pocket maximum?",
    "What is the 2025 Medicare Part B monthly premium?",
    "What preventive services are covered at no cost under Part B?",
    "How do I enroll in Medicare Part C (Medicare Advantage)?",
    "Are hearing aids covered under Medicare Advantage plans?",
    "Does Medicare cover acupuncture treatments?",
    "Are telehealth visits covered the same as in‑person visits?",
    "Does Medicare cover routine dental cleanings?",
    "Does Medicare cover annual vision exams?",

    # Medicare Part D & drug coverage
    "Are over‑the‑counter (OTC) drugs covered by Medicare Part D?",
    "What is a Part D ‘donut hole’ and how does it work?",
    "Does Medicare Part D cover insulin pens?",
    "Do all Part D plans cover the shingles vaccine?",
    "How does step therapy work in a Part D formulary?",
    "Can I change my Part D plan outside the Annual Enrollment Period?",

    # Supplemental & employer coordination
    "What is the difference between Medigap Plan G and Plan N?",
    "Can I keep employer group insurance and delay Part B without penalty?",
    "How does COBRA coverage coordinate with Medicare?",
    "Is TRICARE For Life considered creditable drug coverage?",

    # Medicaid interactions
    "What is a Medicare Savings Program (MSP) and who qualifies?",
    "Does Medicaid pay Medicare Part B premiums?",
    "Can Dual‑Eligible members enroll in a Medicare Advantage plan?",
    "How does Extra Help affect Part D out‑of‑pocket costs?",

    # Costs & billing
    "What is the 2025 Medicare Part A inpatient deductible?",
    "Does Medicare have a family out‑of‑pocket cap?",
    "How are coinsurance and copayments different?",
    "What is balance billing and when is it allowed under Medicare?",
    "Do Medicare Advantage plans have maximum out‑of‑pocket limits?",

    # Enrollment & penalties
    "What happens if I miss my Initial Enrollment Period for Part B?",
    "Is there a late‑enrollment penalty for Part D?",
    "What counts as a Special Enrollment Period trigger event?",
    "Can immigrants without a full work history buy into Medicare?",

    # Private individual & marketplace plans
    "What is the difference between an HMO and PPO health plan?",
    "Do ACA marketplace plans cover pediatric dental services?",
    "How does income affect ACA premium tax credits?",
    "What is the out‑of‑pocket maximum for a Bronze plan in 2025?",
    "Are short‑term limited‑duration health plans ACA‑compliant?",

    # Benefits & services
    "Does any plan cover long‑term custodial nursing‑home care?",
    "Are routine podiatry visits covered under typical insurance?",
    "Do health plans usually cover fertility treatments like IVF?",
    # "Are mental‑health teletherapy sessions covered differently than in‑office sessions?",
    "How often are colonoscopy screenings covered without cost‑sharing?",

    # Claims, appeals & legality
    "What is the process to appeal a denied Medicare claim?",
    "How long do I have to file an Internal Appeal under ACA rules?",
    "Does HIPAA protect all health information from employers?",
    "Can insurers still impose pre‑existing condition exclusions?",
    "What is surprise‑billing protection under the No Surprises Act?",

    # Travel & emergencies
    "Does Medicare cover medical emergencies while travelling abroad?",
    "How do I get dialysis coverage when travelling out of state?",
    "Do ACA plans cover urgent care at out‑of‑network facilities?",
    "Are air‑ambulance services covered under typical employer plans?",

    # Miscellaneous
    # "What preventive vaccines must ACA‑compliant plans cover for adults?",
    "How does an HSA pair with a High‑Deductible Health Plan?",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_questions(path: str | Path | None) -> List[str]:
    """Return a list of questions. If *path* is None, use the default list."""
    if not path:
        return QUESTIONS
    txt = Path(path).expanduser().read_text().strip().splitlines()
    return [line.strip() for line in txt if line.strip()]


def run_batch(questions: List[str]) -> Tuple[dict, List[Tuple[int, str, str]]]:
    """Run each question through rag_local.ask.

    Returns
    -------
    qa_dict : dict
        Dictionary mapping each question to its answer.
    unknowns : list[(int, str, str)]
        Tuples of (index, question, answer) where the model said it didn’t know.
    """
    answers: List[str] = []
    unknowns: List[Tuple[int, str, str]] = []
    qa_dict = {}

    for i, q in enumerate(questions, 1):
        print(f"\n[{i}/{len(questions)}] ❓ {q}")
        answer, _ctx = rag_local.ask(q)
        answers.append(answer)
        qa_dict[q] = answer

        # Simple heuristic to detect unknown/refusal answers
        lower = answer.lower()
        if "does not contain" in lower or "i don’t know" in lower or "i don't know" in lower:
            unknowns.append((i, q, answer))

        clean = answer.replace("\n", " ")
        print("   🟢", clean)

    return qa_dict, unknowns


# ---------------------------------------------------------------------------
# Main script
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # 1st CLI arg = optional questions file
    q_file = sys.argv[1] if len(sys.argv) > 1 else None
    qs = load_questions(q_file)

    qa_dict, unknowns = run_batch(qs)

    print("\n================ SUMMARY ================")
    print(f"Total questions: {len(qs)}")
    print(f"'I don't know' answers: {len(unknowns)}")

    if unknowns:
        print("\nTopics to add PDFs for:")
        for i, q, _ in unknowns:
            print(f"  {i}. {q}")
    else:
        print("\nGreat! All questions were answered from the current knowledge base.")

    # Optionally, print the full question-answer mapping as a dictionary
    import pprint; pprint.pprint(qa_dict)

    # Save results as JSON for further analysis
    import json
    with open("batch_results.json", "w", encoding="utf-8") as f:
        json.dump(qa_dict, f, ensure_ascii=False, indent=2)
