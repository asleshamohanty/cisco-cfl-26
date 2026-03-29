"""
Phase 1 — Data Loader
Reads CFL_External_Data_Pack_Phase1.xlsx exactly as structured.
"""
import pandas as pd
import numpy as np
import re
import os
import pickle

QUARTER_LABELS = ['FY23_Q2','FY23_Q3','FY23_Q4','FY24_Q1','FY24_Q2',
                   'FY24_Q3','FY24_Q4','FY25_Q1','FY25_Q2','FY25_Q3',
                   'FY25_Q4','FY26_Q1']

BD_QUARTER_LABELS = ['FY24_Q2','FY24_Q3','FY24_Q4','FY25_Q1',
                      'FY25_Q2','FY25_Q3','FY25_Q4','FY26_Q1']

SEG_QUARTER_LABELS = ['FY23_Q1','FY23_Q2','FY23_Q3','FY23_Q4','FY24_Q1',
                       'FY24_Q2','FY24_Q3','FY24_Q4','FY25_Q1','FY25_Q2',
                       'FY25_Q3','FY25_Q4','FY26_Q1']

ACCURACY_QUARTERS = ['FY26_Q1', 'FY25_Q4', 'FY25_Q3']


def parse_bookings_sheet(df_raw):
    products = []
    for i in range(2, 32):
        row = df_raw.iloc[i]
        product_name = str(row.iloc[1]).strip()
        lifecycle = str(row.iloc[2]).strip()
        cost_rank = int(row.iloc[0]) if not pd.isna(row.iloc[0]) else i - 1
        actuals = []
        for j in range(3, 15):
            val = row.iloc[j]
            actuals.append(float(val) if not pd.isna(val) else 0.0)
        dp_f = float(row.iloc[16]) if not pd.isna(row.iloc[16]) else np.nan
        mk_f = float(row.iloc[17]) if not pd.isna(row.iloc[17]) else np.nan
        ds_f = float(row.iloc[18]) if not pd.isna(row.iloc[18]) else np.nan
        products.append({
            'Product': product_name, 'Lifecycle': lifecycle, 'Cost_Rank': cost_rank,
            'actuals': actuals,
            'demand_planner_forecast': dp_f, 'marketing_forecast': mk_f,
            'data_science_forecast': ds_f,
        })

    actuals_data = {'Product': [p['Product'] for p in products]}
    for k, q in enumerate(QUARTER_LABELS):
        actuals_data[q] = [p['actuals'][k] for p in products]
    actuals_df = pd.DataFrame(actuals_data)

    forecast_df = pd.DataFrame({
        'Product': [p['Product'] for p in products],
        'demand_planner': [p['demand_planner_forecast'] for p in products],
        'marketing': [p['marketing_forecast'] for p in products],
        'data_science': [p['data_science_forecast'] for p in products],
    })

    meta_df = pd.DataFrame({
        'Product': [p['Product'] for p in products],
        'Lifecycle': [p['Lifecycle'] for p in products],
        'Cost_Rank': [p['Cost_Rank'] for p in products],
    })

    accuracy_rows = []
    for i in range(37, 67):
        if i >= len(df_raw):
            break
        row = df_raw.iloc[i]
        product_name = str(row.iloc[1]).strip()
        if pd.isna(row.iloc[1]) or product_name == 'nan':
            continue
        acc = {'Product': product_name}
        dp_cols = [2, 3, 4, 5, 6, 7]
        for q_idx, q in enumerate(ACCURACY_QUARTERS):
            v = row.iloc[dp_cols[q_idx * 2]]
            acc[f'demand_planner_acc_{q}'] = float(v) if not pd.isna(v) else np.nan
            v = row.iloc[dp_cols[q_idx * 2 + 1]]
            acc[f'demand_planner_bias_{q}'] = float(v) if not pd.isna(v) else np.nan
        mktg_cols = [9, 10, 11, 12, 13, 14]
        for q_idx, q in enumerate(ACCURACY_QUARTERS):
            v = row.iloc[mktg_cols[q_idx * 2]]
            acc[f'marketing_acc_{q}'] = float(v) if not pd.isna(v) else np.nan
            v = row.iloc[mktg_cols[q_idx * 2 + 1]]
            acc[f'marketing_bias_{q}'] = float(v) if not pd.isna(v) else np.nan
        ds_cols = [16, 17, 18, 19, 20, 21]
        for q_idx, q in enumerate(ACCURACY_QUARTERS):
            v = row.iloc[ds_cols[q_idx * 2]]
            acc[f'data_science_acc_{q}'] = float(v) if not pd.isna(v) else np.nan
            v = row.iloc[ds_cols[q_idx * 2 + 1]]
            acc[f'data_science_bias_{q}'] = float(v) if not pd.isna(v) else np.nan
        accuracy_rows.append(acc)
    accuracy_df = pd.DataFrame(accuracy_rows)

    print(f"[DataLoader] Bookings: {len(actuals_df)} products, {len(QUARTER_LABELS)} quarters")
    print(f"[DataLoader] Accuracy: {len(accuracy_df)} products x 3 teams x 3 quarters")
    return {
        'actuals': actuals_df, 'competitor_forecasts': forecast_df,
        'accuracy_history': accuracy_df, 'metadata': meta_df,
    }


def parse_big_deal_sheet(df_raw):
    big_deal_rows, avg_deal_rows = [], []
    for i in range(1, 31):
        if i >= len(df_raw):
            break
        row = df_raw.iloc[i]
        product = str(row.iloc[1]).strip()
        if pd.isna(row.iloc[1]) or product == 'nan':
            continue
        bd, ad = {'Product': product}, {'Product': product}
        for q_idx, q in enumerate(BD_QUARTER_LABELS):
            bd[q] = float(row.iloc[10 + q_idx]) if not pd.isna(row.iloc[10 + q_idx]) else 0.0
            ad[q] = float(row.iloc[18 + q_idx]) if not pd.isna(row.iloc[18 + q_idx]) else 0.0
        big_deal_rows.append(bd)
        avg_deal_rows.append(ad)
    print(f"[DataLoader] Big Deal: {len(big_deal_rows)} products, {len(BD_QUARTER_LABELS)} quarters")
    return {'big_deal': pd.DataFrame(big_deal_rows), 'avg_deal': pd.DataFrame(avg_deal_rows)}


def parse_scms_sheet(df_raw):
    rows = []
    for i in range(2, len(df_raw)):
        row = df_raw.iloc[i]
        product = str(row.iloc[1]).strip() if not pd.isna(row.iloc[1]) else None
        segment = str(row.iloc[2]).strip() if not pd.isna(row.iloc[2]) else None
        if not product or product == 'nan':
            continue
        entry = {'Product': product, 'SCMS_Segment': segment}
        for q_idx, q in enumerate(SEG_QUARTER_LABELS):
            val = row.iloc[3 + q_idx]
            entry[q] = float(val) if not pd.isna(val) else 0.0
        rows.append(entry)
    df = pd.DataFrame(rows)
    print(f"[DataLoader] SCMS: {len(df)} rows, {df['Product'].nunique()} products")
    return df


def parse_vms_sheet(df_raw):
    rows = []
    for i in range(2, len(df_raw)):
        row = df_raw.iloc[i]
        product = str(row.iloc[1]).strip() if not pd.isna(row.iloc[1]) else None
        vertical = str(row.iloc[2]).strip() if not pd.isna(row.iloc[2]) else None
        if not product or product == 'nan':
            continue
        entry = {'Product': product, 'VMS_Segment': vertical}
        for q_idx, q in enumerate(SEG_QUARTER_LABELS):
            val = row.iloc[3 + q_idx]
            entry[q] = float(val) if not pd.isna(val) else 0.0
        rows.append(entry)
    df = pd.DataFrame(rows)
    print(f"[DataLoader] VMS: {len(df)} rows, {df['Product'].nunique()} products")
    return df


def load_all_data(excel_path):
    print(f"[DataLoader] Loading {excel_path}")
    xls = pd.ExcelFile(excel_path)
    sheets = {name: pd.read_excel(xls, sheet_name=name) for name in xls.sheet_names}
    bookings_key = [k for k in sheets if 'Actual' in k or 'Booking' in k][0]
    big_deal_key = [k for k in sheets if 'Big Deal' in k or 'Big' in k][0]
    scms_key = [k for k in sheets if 'SCMS' in k][0]
    vms_key = [k for k in sheets if 'VMS' in k][0]
    glossary_key = [k for k in sheets if 'Glossary' in k][0]
    insights_key = [k for k in sheets if 'Insight' in k or 'Masked' in k][0]
    return {
        'bookings': parse_bookings_sheet(sheets[bookings_key]),
        'big_deal': parse_big_deal_sheet(sheets[big_deal_key]),
        'scms': parse_scms_sheet(sheets[scms_key]),
        'vms': parse_vms_sheet(sheets[vms_key]),
        'glossary': sheets[glossary_key],
        'product_insights': sheets[insights_key],
    }


def validate_master(master):
    issues = []
    actuals = master['bookings']['actuals']
    qcols = [c for c in actuals.columns if c != 'Product']
    nan_ct = actuals[qcols].isna().sum().sum()
    zero_ct = (actuals[qcols] == 0).sum().sum()
    if nan_ct > 0:
        issues.append(f"WARNING: {nan_ct} NaN in actuals")
    print(f"[Validation] {len(actuals)} products x {len(qcols)} quarters, NaN={nan_ct}, Zeros={zero_ct}")
    fc = master['bookings']['competitor_forecasts']
    for t in ['demand_planner','marketing','data_science']:
        m = fc[t].isna().sum()
        if m > 0:
            issues.append(f"WARNING: {t} missing {m} forecasts")
    return {'valid': not any('CRITICAL' in i for i in issues), 'issues': issues}


def save_master(master, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'master_data.pkl'), 'wb') as f:
        pickle.dump(master, f)

def load_master(output_dir):
    with open(os.path.join(output_dir, 'master_data.pkl'), 'rb') as f:
        return pickle.load(f)
