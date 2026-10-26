"""
Verification script for Stage 5 (Streamlit Application).
Uses streamlit.testing.v1.AppTest to verify:
- App loads cleanly without exceptions
- Form returns prediction for a sample row
- Filters change row counts correctly against the dataset
- Empty filter result does not crash and renders warning message
"""

import sys
import pandas as pd
from streamlit.testing.v1 import AppTest

from src.config import CLEAN_DATA_PATH


def verify_streamlit_app() -> None:
    """Run automated AppTest test suite for Streamlit application."""
    print("=" * 60)
    print("VERIFYING STAGE 5 STREAMLIT APP VIA APPTEST")
    print("=" * 60)

    # 1. Initialize and load app
    print("Test 1: Loading app/app.py via AppTest...")
    at = AppTest.from_file("app/app.py", default_timeout=35)
    at.run()
    assert not at.exception, f"App crashed on initial load: {at.exception}"
    print("✔ Check 1 Passed: App loaded successfully with 0 exceptions.")

    # 2. Test Prediction Form submission
    print("\nTest 2: Submitting property prediction form...")
    assert len(at.button) > 0, "Submit button not found in app!"
    at.button[0].click().run()
    assert not at.exception, f"App crashed on form submit: {at.exception}"

    # Verify metrics rendered
    metric_labels = [m.label for m in at.metric]
    assert "Current Valuation" in metric_labels, "Current Valuation metric missing!"
    assert "Forecasted 5-Year Price" in metric_labels, "Forecasted 5-Year Price metric missing!"
    print(f"✔ Check 2 Passed: Form returned predictions successfully (Metrics: {metric_labels}).")

    # 3. Test Sidebar Filtering against dataset
    print("\nTest 3: Testing dynamic sidebar filtering against CSV ground truth...")
    df = pd.read_csv(CLEAN_DATA_PATH)
    target_state = "Karnataka"
    expected_state_count = len(df[df["State"] == target_state])

    # Select state filter (multiselect[0])
    at.sidebar.multiselect[0].select(target_state).run()
    assert not at.exception, f"App crashed after filtering by state: {at.exception}"

    # Verify matching count rendered in stat card
    stat_cards_text = "".join([m.value for m in at.markdown])
    assert f"{expected_state_count:,}" in stat_cards_text, (
        f"Filtered count does not match expected {expected_state_count:,} records!"
    )
    print(f"✔ Check 3 Passed: State filter accurately filtered to {expected_state_count:,} rows matching CSV.")

    # 4. Test Empty Filter Edge Case
    print("\nTest 4: Testing zero-match filter edge case...")
    # Restrict price and BHK to mutually exclusive range
    at.sidebar.slider[0].set_value((10.0, 10.0))  # Price range ₹10.0L to ₹10.0L
    at.sidebar.multiselect[3].select(5)            # BHK 5
    at.sidebar.slider[1].set_value((500, 500)).run()
    assert not at.exception, f"App crashed on empty filter result: {at.exception}"

    # Verify empty state warning is displayed
    warning_texts = [w.value for w in at.warning]
    assert any("No properties match" in w for w in warning_texts), (
        "Empty filter warning not displayed!"
    )
    print("✔ Check 4 Passed: Empty filter handled gracefully without crashing.")

    print("\nSTAGE 5 VERIFICATION COMPLETE: ALL APPTEST CHECKS PASSED SUCCESSFULLY!\n")


if __name__ == "__main__":
    verify_streamlit_app()
