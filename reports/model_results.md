# Model Results & Experiment Tracking Summary

This document presents the empirical evaluation of all trained models for both the **Investment Classification** and **5-Year Price Regression** tasks, logged via local MLflow tracking (`./mlruns`).

---

## 1. Feature Lists Used in Production

### A. Investment Classifier Features (`CLASSIFIER_FEATURE_COLUMNS` — 19 features)
```python
[
    'State', 'City', 'Locality', 'Property_Type', 'BHK', 'Size_in_SqFt',
    'Price_in_Lakhs', 'Floor_No', 'Total_Floors', 'Age_of_Property',
    'Nearby_Schools', 'Nearby_Hospitals', 'Parking_Space', 'Security',
    'Facing', 'Owner_Type', 'Availability_Status', 'amenity_count', 'floor_ratio'
]
```

### B. 5-Year Price Regressor Features (`REGRESSOR_FEATURE_COLUMNS` — 26 features)
```python
[
    'State', 'City', 'Locality', 'Property_Type', 'BHK', 'Size_in_SqFt',
    'Price_in_Lakhs', 'Year_Built', 'Furnished_Status', 'Floor_No', 'Total_Floors',
    'Age_of_Property', 'Nearby_Schools', 'Nearby_Hospitals',
    'Public_Transport_Accessibility', 'Parking_Space', 'Security',
    'Facing', 'Owner_Type', 'Availability_Status', 'amenity_count',
    'security_feature_count', 'floor_ratio', 'school_density_score',
    'infrastructure_score', 'age_bucket'
]
```

---

## 2. Investment Classifier: Leakage & Ablation Experiments

To evaluate the exact boundaries of feature leakage and understand where predictive signal originates, the XGBoost classifier was trained across three controlled feature configurations on the 50,000 held-out test partition (80/20 stratified split, random seed 42):

### Feature Variant Comparison Table

| Model Variant | Excluded Features | Feature Count | Accuracy | F1-Score | ROC-AUC | Primary Driver / Behavior |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Current Model** | Direct formula outputs (`Price_per_SqFt`, `infrastructure_score`, `growth_rate`, `City_Tier`, `transit`) | **19** | **87.43%** | **0.7492** | **0.9423** | **Production Balance**: Learns non-linear valuation & local infrastructure signals without 100% rule inversion. |
| **Variant A** | + Dropping `Price_in_Lakhs` & `Size_in_SqFt` | **17** | **75.77%** | **0.3228** | **0.7642** | **Valuation Stripped**: Without pricing and area, the model cannot evaluate price-per-sqft attractiveness. F1 collapses by 42.6 points. |
| **Variant B** | + Dropping `Nearby_Schools`, `Nearby_Hospitals`, & `City` | **14** | **74.77%** | **0.0003** | **0.5710** | **Signal Depleted**: Stripped of both valuation and civic infrastructure/city signals, model collapses to majority-class guessing (74.76% Class 0). |

### Key Takeaways from the Ablation Study:
1. **Price and Size are Essential for Value Screening**: Removing `Price_in_Lakhs` and `Size_in_SqFt` drops the F1-score from **0.7492** down to **0.3228** and ROC-AUC from **0.9423** to **0.7642**. Because an investment evaluation fundamentally compares price to asset size, stripping these features disables the model's ability to identify undervalued properties.
2. **Civic Infrastructure Drives the Remaining Signal**: Further removing `Nearby_Schools`, `Nearby_Hospitals`, and `City` causes the F1-score to drop to near-zero (**0.0003**) and ROC-AUC to collapse to **0.5710** (near random guessing). This proves that the model relies on civic infrastructure density and urban markets to separate viable opportunities from unviable listings.

---

## 3. Production Model Performance Comparisons

### A. Investment Classifier (50,000 Test Properties)

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 81.75% | 68.65% | 52.09% | 0.5924 | 0.8714 | Baseline Linear |
| **Random Forest Classifier** | 80.63% | 85.62% | 28.73% | 0.4303 | 0.9029 | High Precision |
| **XGBoost Classifier (Best)** | **87.43%** | **75.33%** | **75.46%** | **0.7492** | **0.9423** | **Production Best** |

### B. 5-Year Price Regressor (50,000 Test Properties)

> **Important Caveat**: The 5-year regressor mostly relearns the synthetic growth formula (Baseline $R^2 = 0.9822$ vs Model $R^2 = 0.9996$).

| Model | RMSE (₹ Lakhs) | MAE (₹ Lakhs) | $R^2$ Score | Error Reduction vs Baseline |
| :--- | :---: | :---: | :---: | :---: |
| **Baseline (Price × Avg Growth)** | 27.318 | 19.334 | **0.9822** | Reference Heuristic |
| **Ridge Regression** | 13.257 | 9.190 | **0.9958** | 51.5% Reduction |
| **Random Forest Regressor** | 14.075 | 10.559 | **0.9953** | 48.5% Reduction |
| **XGBoost Regressor (Best)** | **4.246** | **3.083** | **0.9996** | **84.5% Reduction** |
