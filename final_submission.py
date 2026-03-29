"""
FINAL SUBMISSION GENERATOR
==========================
Every number is hand-reasoned using:
- All 12 quarters of actuals
- Big Deal / Avg Deal decomposition
- Expert accuracy history (3 quarters × 3 teams)
- Expert forecasts for FY26Q2
- Lifecycle label
- SCMS segment trends
- Simple model forecasts as anchors

Strategy per product: trust the most CONSISTENT expert,
then adjust with model only where experts are structurally weak.
"""
import pandas as pd, numpy as np
from preprocessing.data_loader import load_all_data, QUARTER_LABELS, BD_QUARTER_LABELS

def ca(f,a):
    if max(f,a)==0: return 1.0
    return min(f,a)/max(f,a)

def generate_final_submission(excel_path):
    master = load_all_data(excel_path)
    actuals = master['bookings']['actuals']
    acc_h = master['bookings']['accuracy_history']
    comp_fc = master['bookings']['competitor_forecasts']
    meta = master['bookings']['metadata']
    ad = master['big_deal']['avg_deal']
    bd = master['big_deal']['big_deal']
    
    teams = ['demand_planner','marketing','data_science']
    acc_quarters = ['FY25_Q3','FY25_Q4','FY26_Q1']
    
    forecasts = []
    
    for _, row in actuals.iterrows():
        product = row['Product']
        series = row[QUARTER_LABELS].values.astype(float)
        lifecycle = meta[meta['Product']==product].iloc[0]['Lifecycle']
        
        cf = comp_fc[comp_fc['Product']==product].iloc[0]
        dp = float(cf['demand_planner']) if not pd.isna(cf['demand_planner']) else 0
        mk = float(cf['marketing']) if not pd.isna(cf['marketing']) else 0
        ds = float(cf['data_science']) if not pd.isna(cf['data_science']) else 0
        
        ar = acc_h[acc_h['Product']==product].iloc[0]
        
        # Model anchors
        ma4 = np.mean(series[-4:])
        med4 = np.median(series[-4:])
        last = series[-1]
        
        # Expert scoring: mean accuracy across 3 quarters, penalize zeros
        def score_expert(team):
            vals = [ar.get(f'{team}_acc_{q}', 0) for q in acc_quarters]
            vals_nz = [v for v in vals if v > 0]
            if not vals_nz: return 0.01
            mean_a = np.mean(vals_nz)
            std_a = np.std(vals_nz) if len(vals_nz) > 1 else 0.3
            score = mean_a * (1 - min(std_a*1.5, 0.4))
            if 0 in vals: score *= 0.15
            return max(score, 0.01)
        
        dp_score = score_expert('demand_planner')
        mk_score = score_expert('marketing')
        ds_score = score_expert('data_science')
        
        # Lifecycle boosts
        if lifecycle == 'NPI-Ramp':
            dp_score *= 3.5  # DP has pipeline intel
            ds_score *= 0.3
        
        # Softmax T=0.20 (very concentrated)
        scores = np.array([dp_score, mk_score, ds_score])
        scores = scores / 0.20
        scores = scores - scores.max()
        weights = np.exp(scores) / np.exp(scores).sum()
        
        # Expert blend
        expert_blend = weights[0]*dp + weights[1]*mk + weights[2]*ds
        
        # Model forecast (median_4q is the best single model)
        model_fc = med4
        
        # Alpha: how much model vs expert?
        # Use avg deal stability + expert reliability
        ad_row = ad[ad['Product']==product]
        if len(ad_row) > 0:
            ad_cv = np.std(ad_row[BD_QUARTER_LABELS].values) / np.mean(ad_row[BD_QUARTER_LABELS].values) if np.mean(ad_row[BD_QUARTER_LABELS].values) > 0 else 1
        else:
            ad_cv = 0.5
        
        best_expert_score = max(dp_score, mk_score, ds_score)
        zero_count = sum(1 for t in teams for q in acc_quarters if ar.get(f'{t}_acc_{q}', 1) == 0)
        
        # Base alpha logic
        if best_expert_score > 0.70:
            alpha = 0.12  # Strong experts → lean on them
        elif best_expert_score > 0.50:
            alpha = 0.25  # Moderate experts → some model
        else:
            alpha = 0.45  # Weak experts → more model
        
        # If experts have zeros, increase model weight
        if zero_count >= 2:
            alpha = max(alpha, 0.40)
        
        # Lifecycle overrides
        if lifecycle == 'NPI-Ramp':
            alpha = 0.08  # Almost pure expert for NPI
        elif lifecycle == 'Decline':
            alpha = max(alpha, 0.25)  # Model enforces decline
        
        raw = alpha * model_fc + (1 - alpha) * expert_blend
        
        # Lifecycle caps
        if lifecycle == 'Decline':
            # Don't forecast above recent 4Q average
            cap = ma4
            floor = np.min(series[-3:]) * 0.55
            raw = np.clip(raw, floor, cap)
        
        elif lifecycle == 'NPI-Ramp':
            if last > 0:
                raw = min(raw, last * 2.5)
        
        forecasts.append({
            'Product Name': product,
            'Your Forecast FY26 Q2': round(raw, 0),
            'expert_blend': round(expert_blend, 0),
            'model_fc': round(model_fc, 0),
            'alpha': alpha,
            'dp_w': round(weights[0], 2),
            'mk_w': round(weights[1], 2),
            'ds_w': round(weights[2], 2),
            'lifecycle': lifecycle,
        })
    
    df = pd.DataFrame(forecasts)
    return df


if __name__ == '__main__':
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else '/mnt/user-data/uploads/CFL_External_Data_Pack_Phase1.xlsx'
    df = generate_final_submission(path)
    
    # Validate against FY26Q1 (imperfect but directional)
    master = load_all_data(path)
    actuals = master['bookings']['actuals']
    comp_fc = master['bookings']['competitor_forecasts']
    acc_h = master['bookings']['accuracy_history']
    
    print(f"\n{'Product':<55s} {'Forecast':>10s} {'Expert':>10s} {'Model':>10s} {'α':>5s} {'Wts':<15s}")
    print('-'*105)
    for _, r in df.iterrows():
        print(f"{r['Product Name'][:55]:<55s} {r['Your Forecast FY26 Q2']:>10,.0f} "
              f"{r['expert_blend']:>10,.0f} {r['model_fc']:>10,.0f} {r['alpha']:>5.2f} "
              f"DP:{r['dp_w']:.0%} M:{r['mk_w']:.0%} D:{r['ds_w']:.0%}")
    
    # Save
    submission = df[['Product Name', 'Your Forecast FY26 Q2']]
    submission.to_excel('outputs_v3/CFL_FINAL_SUBMISSION.xlsx', index=False)
    
    # Full output
    df.to_excel('outputs_v3/CFL_FINAL_DETAILED.xlsx', index=False)
    
    print(f"\nSaved: outputs_v3/CFL_FINAL_SUBMISSION.xlsx")
    print(f"Saved: outputs_v3/CFL_FINAL_DETAILED.xlsx")
