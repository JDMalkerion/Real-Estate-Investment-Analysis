"""
Verification script for Stage 2 (Exploratory Data Analysis).

Asserts:
- Exactly 21 PNG charts exist in reports/eda/figures/
- reports/eda/findings.md contains all 20 question headings (Q1 to Q20)
- Every image link in findings.md resolves to an existing file
- No file in reports/eda/ exceeds 1 MB
- Every percentage in the scorecard matches an independent recomputation from the CSV
"""

from __future__ import annotations

from pathlib import Path
import re
import sys
import pandas as pd

# Workspace root
BASE_DIR = Path(__file__).resolve().parent
CLEAN_DATA_PATH = BASE_DIR / "Data" / "housing_clean.csv"
EDA_REPORTS_DIR = BASE_DIR / "reports" / "eda"
FIGURES_DIR = EDA_REPORTS_DIR / "figures"
FINDINGS_PATH = EDA_REPORTS_DIR / "findings.md"


def pct_spread(s: pd.Series) -> float:
    """Calculate spread as (max - min) / min * 100."""
    lo, hi = s.min(), s.max()
    return (hi - lo) / lo * 100 if lo > 0 else 0.0


def verify_eda() -> None:
    print("=" * 60)
    print("VERIFYING STAGE 2 EDA OUTPUT & FINDINGS")
    print("=" * 60)

    # ── Check 1: 21 PNGs exist ────────────────────────────────────────────────
    expected_pngs = [
        "q01_price_distribution.png",
        "q02_size_distribution.png",
        "q03_price_per_sqft_by_property_type.png",
        "q04_size_vs_price.png",
        "q05_outliers_price_and_size.png",
        "q06_avg_price_per_sqft_by_state.png",
        "q07_avg_price_by_city.png",
        "q08_median_age_by_locality.png",
        "q09_bhk_distribution_across_cities.png",
        "q10_price_trend_top_localities.png",
        "q11_correlation_heatmap.png",
        "q12_nearby_schools_vs_price.png",
        "q13_nearby_hospitals_vs_price.png",
        "q14_price_by_furnished_status.png",
        "q15_price_per_sqft_by_facing.png",
        "q16_count_by_owner_type.png",
        "q17_count_by_availability_status.png",
        "q18_parking_vs_price.png",
        "q19_amenities_vs_price.png",
        "q20_transport_vs_price_and_label.png",
        "scorecard.png",
    ]
    assert FIGURES_DIR.exists(), f"Figures directory missing: {FIGURES_DIR}"
    png_files = list(FIGURES_DIR.glob("*.png"))
    assert len(png_files) == 21, f"Expected 21 PNGs, found {len(png_files)}"
    for name in expected_pngs:
        p = FIGURES_DIR / name
        assert p.exists(), f"Expected chart missing: {name}"
    print(f"✔ Check 1 Passed: Exactly 21 PNG charts exist in reports/eda/figures/.")

    # ── Check 2: findings.md contains all 20 question headings ────────────────
    assert FINDINGS_PATH.exists(), f"findings.md missing at {FINDINGS_PATH}"
    findings_text = FINDINGS_PATH.read_text(encoding="utf-8")

    for q in range(1, 21):
        # Match e.g. "#### Q1: Price Distribution" or "### Q1"
        pattern = rf"^#+\s+.*Q{q}\b"
        match = re.search(pattern, findings_text, re.MULTILINE)
        assert match is not None, f"Heading for Q{q} missing in {FINDINGS_PATH}"
    print(f"✔ Check 2 Passed: findings.md contains headings for all 20 EDA questions (Q1–Q20).")

    # ── Check 3: Every image link in findings.md resolves ─────────────────────
    img_matches = re.findall(r"!\[(.*?)\]\((.*?)\)", findings_text)
    assert len(img_matches) >= 21, f"Expected at least 21 image links, found {len(img_matches)}"
    for alt, rel_path in img_matches:
        resolved = (FINDINGS_PATH.parent / rel_path).resolve()
        assert resolved.exists(), f"Image link '{rel_path}' does not resolve to an existing file!"
        assert resolved.is_file(), f"Image target '{rel_path}' is not a file!"
    print(f"✔ Check 3 Passed: All {len(img_matches)} image links in findings.md successfully resolve to valid files.")

    # ── Check 4: No file in reports/eda/ exceeds 1 MB ──────────────────────────
    max_allowed_bytes = 1024 * 1024  # 1 MB
    checked_files = 0
    for file_path in EDA_REPORTS_DIR.rglob("*"):
        if file_path.is_file():
            size = file_path.stat().st_size
            assert size <= max_allowed_bytes, (
                f"File {file_path.relative_to(BASE_DIR)} ({size / 1024:.1f} KB) exceeds 1 MB!"
            )
            checked_files += 1
    print(f"✔ Check 4 Passed: All {checked_files} files in reports/eda/ are strictly under 1 MB.")

    # ── Check 5: Scorecard percentages match independent CSV recomputation ────
    assert CLEAN_DATA_PATH.exists(), f"Cleaned data missing: {CLEAN_DATA_PATH}"
    df = pd.read_csv(CLEAN_DATA_PATH)

    factor_configs = {
        "Q6": ("State", "Price_per_SqFt"),
        "Q7": ("City", "Price_in_Lakhs"),
        "Q3": ("Property_Type", "Price_per_SqFt"),
        "Q12": ("Nearby_Schools", "Price_per_SqFt"),
        "Q13": ("Nearby_Hospitals", "Price_per_SqFt"),
        "Q14": ("Furnished_Status", "Price_in_Lakhs"),
        "Q15": ("Facing", "Price_per_SqFt"),
        "Q18": ("Parking_Space", "Price_in_Lakhs"),
        "Q19": ("amenity_count", "Price_per_SqFt"),
        "Q20": ("Public_Transport_Accessibility", "Price_per_SqFt"),
    }

    # Recompute independently from df
    recomputed = {}
    for q_id, (group_col, metric_col) in factor_configs.items():
        g = df.groupby(group_col)[metric_col].mean()
        recomputed[q_id] = round(pct_spread(g), 1)

    # Parse scorecard table from findings.md
    # Line format: | Q6 | State | Rajasthan ₹11,727 | Karnataka ₹11,951 | 1.9% | No signal |
    table_rows = re.findall(
        r"\|\s*(Q\d+)\s*\|[^|]+\|[^|]+\|[^|]+\|\s*([\d\.]+)%",
        findings_text
    )
    assert len(table_rows) == 10, f"Expected 10 scorecard table rows, found {len(table_rows)}"

    for q_id, pct_str in table_rows:
        parsed_pct = float(pct_str)
        expected_pct = recomputed[q_id]
        assert abs(parsed_pct - expected_pct) < 1e-4, (
            f"Scorecard percentage mismatch for {q_id}: found {parsed_pct}%, "
            f"expected {expected_pct}%"
        )
    print("✔ Check 5 Passed: All 10 scorecard percentages match independent recomputations from CSV:")
    for q_id, (col, _) in factor_configs.items():
        print(f"    - {q_id} ({col}): {recomputed[q_id]:.1f}%")

    print("\nSTAGE 2 VERIFICATION COMPLETE: ALL CHECKS PASSED SUCCESSFULLY!\n")


if __name__ == "__main__":
    verify_eda()
