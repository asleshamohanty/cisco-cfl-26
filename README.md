# Cisco Forecasting League - Demand Forecasting Solution

## Problem Statement
In today’s landscape, businesses face unprecedented challenges due to geopolitical uncertainty and economic volatility. Global supply chains are under immense pressure to balance fluctuating customer demand with profitability.

### Mission
Forecast quarterly demand for 30 critical Cisco products for **FY26 Q2**, using historical demand data and expert references to support better supply chain planning and customer fulfillment.

## Data Inputs
- Historical quarterly demand values from the data pack
- Reference forecasts from:
  - Demand Planners
  - Marketing Team
  - Machine Learning (Data Science) Team
- Product lifecycle metadata (for example: Sustaining, NPI-Ramp, Decline)
- SCMS segment trend signals
- Expert historical accuracy records

## Forecasting Methodology
The approach is a **hybrid ensemble + meta-learner**:

1. Evaluate multiple forecasting sources per product.
2. Build a model bank and score each model with walk-forward validation.
3. Score expert teams on historical accuracy and consistency.
4. Select product-level strategy (model-dominant, expert-dominant, or blend).
5. Blend model and expert forecasts with adaptive alpha weights.
6. Apply lifecycle and SCMS adjustments for realistic business behavior.

This creates a per-product optimized forecast instead of one global rule for all products.

## Forecasting Tools Used
Implemented tools/models in the codebase include:
- 4-quarter moving average
- 3-quarter moving average
- Seasonal same-quarter-last-year with trend adjustment
- Exponential smoothing (alpha grid search)
- Custom Cisco trend-seasonality heuristic model
- Croston method (for intermittent demand)
- Last value (random walk baseline)
- Median of last 4 quarters (robust anchor)
- Meta-learner source selection and weighted expert blending

## Validation and Accuracy Achieved
Validation method:
- Walk-forward validation on **FY26 Q1** as holdout
- Train source-selection logic on prior quarters
- Apply selected strategy to generate **FY26 Q2** production forecast

## Outputs
Primary scripts:
- `main_v3.py`: full v3 pipeline with validation and production forecast
- `final_submission.py`: final submission generation logic

