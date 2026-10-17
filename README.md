# Automated Valuation Model (AVM) for Residential Real Estate
*An End-to-End Explainable Machine Learning Pipeline for Property Price Estimation*

## 1. Executive Summary & Business Context
In PropTech and mortgage underwriting, Automated Valuation Models (AVMs) estimate residential property prices to streamline loan approvals, property tax assessments, and investment analysis. A practical real estate valuation model must satisfy two core criteria:
1. **Proportional Error Minimization**: Real estate errors scale relative to property value. A 5% error on a $1,000,000 property ($50,000) carries a larger absolute impact than on a $200,000 property ($10,000).
2. **Interpretability & Stakeholder Transparency**: Mortgage underwriters and appraisers cannot rely on black-box predictions; every valuation must provide attributable financial drivers.

This project implements a 5-model heterogeneous ensemble (**XGBoost**, **LightGBM**, **CatBoost**, **Lasso**, **Ridge**) combined with **SHAP model explainability**, achieving a Top 10% benchmark score (**0.12032 RMSLE**) on unseen market test data.

---

## 2. Real-World Domain Intuition & Methodology

### A. Target Normalization (`np.log1p`)
* **Real-World Intuition**: Real estate price distributions are heavily right-skewed by high-value outlier estates. Estimating raw dollar prices directly causes models to over-prioritize luxury properties where absolute dollar variance is highest.
* **Mathematical Rationale**: Transforming the target variable using $\ln(1 + \text{SalePrice})$ converts absolute dollar loss into relative proportional error. Minimizing Root Mean Squared Logarithmic Error (RMSLE) ensures a 5% valuation error carries identical loss weight regardless of whether the home is priced at $100,000 or $1,000,000.

### B. Structural Absence Imputation vs. Naive Imputation
* **Real-World Intuition**: In real estate dataset records, a missing value (`NA`) under amenities like `PoolQC`, `BsmtQual`, or `GarageType` does not represent lost data; it signifies the structural absence of the feature (i.e., "No Pool", "No Basement").
* **Implementation Rationale**: Standard mean or mode imputation would falsely assign baseline amenity ratings to non-existent features. Categorical missing entries were explicitly recoded as `'None'`, and numeric area metrics were set to `0`. This allows regression trees and linear algorithms to learn the true structural discount applied to un-amenitized homes.

### C. Hyper-Local Neighborhood Spatial Aggregations
* **Real-World Intuition**: Real estate values are fundamentally location-driven. A 2,000 sq ft home in an upscale neighborhood commands a drastically different baseline valuation than an identical home in an industrial district.
* **Implementation Rationale**: Properties were evaluated relative to their immediate local micro-market by calculating spatial ratios:
  $$R_{\text{size}} = \frac{\text{GrLivArea}}{\operatorname{Median}\left(\text{GrLivArea}_{\text{Neighborhood}}\right)}$$
  This isolates relative size premiums within specific neighborhoods from broader city-wide square footage trends.

### D. Compound Feature Interactions
* **Real-World Intuition**: Physical traits in real estate do not add value linearly in isolation. High build quality compounding with large square footage yields an exponential valuation premium, whereas old age combined with poor maintenance accelerates structural depreciation.
* **Implementation Rationale**: Engineered composite interaction metrics (`Qual_x_TotalSF`, `Cond_x_Age`, `Total_Qual_Index`) allow linear models and single-level tree nodes to capture multi-variable synergies without requiring deeper, overfitted decision trees.

---

## 3. Heterogeneous Ensembling & Model Diversity

Single model families exhibit structural biases:
* **Depth-Wise Trees (XGBoost)**: Grow trees level-by-level, effectively capturing dominant non-linear feature splits.
* **Leaf-Wise Trees (LightGBM)**: Grow trees by splitting the leaf with maximum loss reduction, capturing fine-grained interactions.
* **Symmetric Trees (CatBoost)**: Build balanced trees using identical split conditions across entire depth levels, acting as a strong regularizer against noise.
* **Regularized Linear Models (Lasso & Ridge)**: Extrapolate smooth linear surfaces across high-dimensional sparse spaces (236 one-hot encoded variables) where decision trees create step-function approximations.

### Out-Of-Fold (OOF) Weight Optimization
To prevent data leakage and avoid arbitrary weight guessing, out-of-fold validation predictions were collected across a 5-Fold Stratified Cross-Validation setup. Sequential Least Squares Programming (`SLSQP`) solved for the optimal ensemble weights that mathematically minimized validation RMSLE:

| Model Architecture | Model Family Type | Ensemble Weight Share |
| :--- | :--- | :--- |
| **XGBoost** | Depth-Wise Gradient Boosted Trees | 41.55% |
| **Lasso Regression** | $L_1$ Regularized Linear Model | 32.97% |
| **Ridge Regression** | $L_2$ Regularized Linear Model | 13.57% |
| **LightGBM** | Leaf-Wise Gradient Boosted Trees | 11.91% |
| **CatBoost** | Symmetric Oblivious Trees | Integrated / Evaluated |

---

## 4. Stakeholder Model Explainability (SHAP Analysis)

To ensure model transparency for underwriting compliance, global and local predictions were decomposed using game-theoretic **SHAP (SHapley Additive exPlanations)** values:

1. **Global Market Drivers**:
   * **Positive Price Drivers**: High `TotalSF` (total living + basement area), `OverallQual` (construction material rating), and neighborhood quality benchmarks consistently push predicted valuations upward.
   * **Negative Price Drivers**: `HouseAge` (property depreciation) and lack of dedicated garage space systematically lower property valuations.
2. **Individual Property Auditing**:
   * For any single valuation, SHAP waterfall plots quantify the exact dollar-equivalent shift caused by each attribute, allowing appraisers to trace predictions back to specific property characteristics.

---

## 5. Iterative Pipeline Performance Progression

| Development Stage | Validation Strategy | Public Leaderboard RMSLE | Global Rank |
| :--- | :--- | :--- | :--- |
| **Baseline Model** | Untuned XGBoost | ~0.13152 | ~1,500+ |
| **Hyperparameter Tuned XGBoost** | 5-Fold Cross-Validation | 0.11433 | — |
| **Stage 1 Ensemble** | XGBoost + Lasso + Ridge | 0.11615 | Rank 609 (0.12149) |
| **Stage 2 Ensemble** | 4-Model Blend (XGBoost + LightGBM + Lasso + Ridge) | 0.11782 | Rank 590 (0.12137) |
| **Stage 3 Ensemble** | SciPy OOF Weight Optimization | 0.11158 | Rank 538 (0.12114) |
| **Stage 4 Spatial Aggregations** | Neighborhood Ratios + Interaction Terms | 0.11166 | Rank 445 (0.12048) |
| **Final 5-Model Pipeline** | Full Heterogeneous Blend + Feature Refinement | **0.11120** | **Rank 424 (0.12032)** |
