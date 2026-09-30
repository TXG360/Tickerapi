from flask import Flask, jsonify, send_file
import yfinance as yf
import pandas as pd
import google.generativeai as genai
from datetime import datetime
import pytz
import csv
import os
import logging
import time

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# ─── CONFIG & API KEYS ────────────────────────────────────────────────────────
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY_HERE")
genai.configure(api_key=GEMINI_API_KEY)

TARGET_TICKERS = ["AAPL", "TGT", "F", "XOM", "MCD", "PFE", "CROX", "META"]

LOG_FILE = "quant_value_ledger.csv"
CACHE_TTL = 86400  
_cache = {}

# ─── CSV Auto-Logger ──────────────────────────────────────────────────────────
def init_csv_logger():
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow([
                "Date", "Ticker", "Price", "Intrinsic Value", "Margin of Safety",
                "F-Score", "Z-Score", "ROIC", "Earnings Yield", "Signal", "AI Qualitative Override"
            ])

def log_scanned_stock(data):
    init_csv_logger()
    pacific = pytz.timezone('America/Los_Angeles')
    today_str = datetime.now(pacific).strftime('%Y-%m-%d')
    
    try:
        with open(LOG_FILE, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow([
                today_str, data['Ticker'], data['Price'], data['Intrinsic Value'], 
                f"{data['Margin of Safety']}%", data['F-Score'], data['Z-Score'], 
                f"{data['ROIC']}%", f"{data['Earnings Yield']}%", data['Signal'], data.get('AI_Signal', 'N/A')
            ])
    except Exception as e:
        logger.error(f"CSV Logging Error: {e}")

# ─── DATA ENGINE & CACHING ────────────────────────────────────────────────────
def cached(key, fn, ttl=CACHE_TTL):
    now = time.time()
    if key in _cache and now - _cache[key]['ts'] < ttl:
        return _cache[key]['data']
    result = fn()
    _cache[key] = {'data': result, 'ts': now}
    return result

def get_val(df, row_name, col_index, default=0):
    try: return df.loc[row_name].iloc[col_index]
    except: return default

# ─── THE MATH ENGINE (Phase 1 & 2) ────────────────────────────────────────────
def run_quant_math(ticker_symbol):
    def fetch():
        logger.info(f"Crunching financials for {ticker_symbol}...")
        ticker = yf.Ticker(ticker_symbol)
        
        try:
            bs = ticker.balance_sheet
            inc = ticker.financials
            cf = ticker.cashflow
            info = ticker.info
            
            if bs.empty or inc.empty or cf.empty:
                return {"Ticker": ticker_symbol, "Signal": "⚪️ SKIP (Missing Data - yFinance Blocked)"}

            price = info.get('currentPrice', info.get('previousClose', 0))
            market_cap = info.get('marketCap', 0)
            total_debt = info.get('totalDebt', 0)
            cash = info.get('totalCash', 0)
            ev = market_cap + total_debt - cash if market_cap else 0
            
            tot_assets_0 = get_val(bs, 'Total Assets', 0)
            tot_assets_1 = get_val(bs, 'Total Assets', 1)
            tot_liab_0 = get_val(bs, 'Total Liabilities Net Minority Interest', 0)
            curr_assets_0, curr_assets_1 = get_val(bs, 'Current Assets', 0), get_val(bs, 'Current Assets', 1)
            curr_liab_0, curr_liab_1 = get_val(bs, 'Current Liabilities', 0), get_val(bs, 'Current Liabilities', 1)
            lt_debt_0, lt_debt_1 = get_val(bs, 'Long Term Debt', 0), get_val(bs, 'Long Term Debt', 1)
            retained_earn = get_val(bs, 'Retained Earnings', 0)
            shares_0, shares_1 = get_val(bs, 'Ordinary Shares Number', 0), get_val(bs, 'Ordinary Shares Number', 1)

            net_income_0, net_income_1 = get_val(inc, 'Net Income', 0), get_val(inc, 'Net Income', 1)
            ebit_0 = get_val(inc, 'EBIT', 0)
            rev_0, rev_1 = get_val(inc, 'Total Revenue', 0), get_val(inc, 'Total Revenue', 1)
            gp_0, gp_1 = get_val(inc, 'Gross Profit', 0), get_val(inc, 'Gross Profit', 1)
            op_cf_0 = get_val(cf, 'Operating Cash Flow', 0)

            f_score = 0
            roa_0 = net_income_0 / tot_assets_0 if tot_assets_0 else 0
            roa_1 = net_income_1 / tot_assets_1 if tot_assets_1 else 0
            
            if roa_0 > 0: f_score += 1
            if op_cf_0 > 0: f_score += 1
            if roa_0 > roa_1: f_score += 1
            if op_cf_0 > net_income_0: f_score += 1
            if (lt_debt_0/tot_assets_0 if tot_assets_0 else 0) < (lt_debt_1/tot_assets_1 if tot_assets_1 else 0): f_score += 1
            if (curr_assets_0/curr_liab_0 if curr_liab_0 else 0) > (curr_assets_1/curr_liab_1 if curr_liab_1 else 0): f_score += 1
            if shares_0 <= shares_1: f_score += 1
            if (gp_0/rev_0 if rev_0 else 0) > (gp_1/rev_1 if rev_1 else 0): f_score += 1
            if (rev_0/tot_assets_0 if tot_assets_0 else 0) > (rev_1/tot_assets_1 if tot_assets_1 else 0): f_score += 1

            working_cap = curr_assets_0 - curr_liab_0
            A = working_cap / tot_assets_0 if tot_assets_0 else 0
            B = retained_earn / tot_assets_0 if tot_assets_0 else 0
            C = ebit_0 / tot_assets_0 if tot_assets_0 else 0
            D = market_cap / tot_liab_0 if tot_liab_0 else 0
            E = rev_0 / tot_assets_0 if tot_assets_0 else 0
            z_score = (1.2 * A) + (1.4 * B) + (3.3 * C) + (0.6 * D) + (1.0 * E)

            invested_capital = tot_assets_0 - curr_liab_0
            roic = (ebit_0 / invested_capital) * 100 if invested_capital else 0
            earnings_yield = (ebit_0 / ev) * 100 if ev else 0

            eps = info.get('trailingEps', 0)
            growth_rate = min(info.get('earningsGrowth', 0) * 100, 12) 
            intrinsic_value = eps * (8.5 + (2 * growth_rate)) if eps > 0 else 0
            margin_of_safety = ((intrinsic_value - price) / intrinsic_value) * 100 if intrinsic_value > 0 else -999

            signal = "🔴 MATH REJECTED"
            color = "#ff6b6b"
            
            if f_score >= 7 and z_score > 3.0 and roic > 15 and margin_of_safety >= 30:
                signal = "🟡 MATH PASSED - AWAITING AI"
                color = "#ffd700"

            return {
                "Ticker": ticker_symbol, "Price": round(price, 2), 
                "Intrinsic Value": round(intrinsic_value, 2), "Margin of Safety": round(margin_of_safety, 2),
                "F-Score": f_score, "Z-Score": round(z_score, 2), 
                "ROIC": round(roic, 2), "Earnings Yield": round(earnings_yield, 2),
                "Signal": signal, "Color": color, "AI_Signal": "Pending"
            }
        except Exception as e:
            logger.error(f"Error processing {ticker_symbol}: {e}")
            return {"Ticker": ticker_symbol, "Signal": "⚪️ SKIP (API Error)"}
            
    return cached(f'quant_{ticker_symbol}', fetch)

# ─── THE AI ENGINE (Phase 3) ──────────────────────────────────────────────────
def run_ai_qualitative_check(ticker, data_dict):
    if "MATH PASSED" not in data_dict['Signal']:
        data_dict['AI_Signal'] = "Skipped (Failed Math)"
        return data_dict

    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = f"""
        You are a ruthless distressed-debt analyst. Scan the internet and SEC filings for {ticker}.
        Ignore boilerplate market risks. Identify ONLY asymmetric threats:
        1. Customer/Supplier Concentration
        2. Legal & Regulatory Guillotines
        3. Massive upcoming debt maturities
        4. Accounting irregularities
        
        Respond ONLY with a JSON object: {{"threats_found": "description of threat or 'None'", "override_reject": true/false}}
        """
        response = model.generate_content(prompt)
        data_dict['AI_Signal'] = "🟢 CLEAR (No Asymmetric Threats)"
        data_dict['Signal'] = "🟢 STRONG BUY"
        data_dict['Color'] = "#00ff88"
    except Exception as e:
        logger.error(f"AI Check Failed for {ticker}: {e}")
        data_dict['AI_Signal'] = "⚠️ AI ERROR"
        
    return data_dict

# ─── ROUTES ───────────────────────────────────────────────────────────────────
def process_all_tickers():
    results = []
    for ticker in TARGET_TICKERS:
        math_result = run_quant_math(ticker)
        if "SKIP" in math_result['Signal']: 
            math_result['Price'] = 0
            math_result['Intrinsic Value'] = 0
            math_result['Margin of Safety'] = 0
            math_result['F-Score'] = 'N/A'
            math_result['Z-Score'] = 'N/A'
            math_result['ROIC'] = 'N/A'
            math_result['Earnings Yield'] = 'N/A'
            math_result['Color'] = '#888888'
            math_result['AI_Signal'] = 'Skipped'
            results.append(math_result)
            continue
            
        final_result = run_ai_qualitative_check(ticker, math_result)
        log_scanned_stock(final_result)
        results.append(final_result)
    return results

@app.route('/api/scan')
def api_scan():
    data = cached('daily_scan', process_all_tickers, ttl=43200) 
    return jsonify(data)

@app.route('/api/ledger')
def api_ledger():
    if os.path.exists(LOG_FILE):
        return send_file(LOG_FILE, mimetype='text/csv', as_attachment=True, download_name='quant_ledger.csv')
    return jsonify({"message": "Ledger empty."}), 200

@app.route('/')
def index():
    pacific = pytz.timezone('America/Los_Angeles')
    now_pt = datetime.now(pacific).strftime('%I:%M %p PT &middot; %b %d, %Y')
    
    data = cached('daily_scan', process_all_tickers, ttl=43200)
    
    buys = [d for d in data if "BUY" in d['Signal']]
    rejects = [d for d in data if "REJECT" in d['Signal']]
    skips = [d for d in data if "SKIP" in d['Signal']]
    
    css = """
    *{box-sizing:border-box;margin:0;padding:0}
    body{font-family:Arial,sans-serif;background:#1a1a2e;color:#eee;padding:16px;max-width:1100px;margin:auto}
    h1{color:#ffd700;font-size:1.5em;margin-bottom:4px}
    h2{font-size:1.1em;margin:16px 0 8px}
    .sub{color:#888;font-size:0.82em;margin-bottom:16px}
    .game{background:#16213e;border:1px solid #0f3460;padding:14px;margin:10px 0;border-radius:10px; border-left:4px solid;}
    .gh{display:flex;justify-content:space-between;align-items:center;margin-bottom:12px; font-size:1.2em; font-weight:bold;}
    .sgrid{display:grid;grid-template-columns:repeat(5,1fr);gap:8px}
    .sc{background:#1a2540;border-radius:4px;padding:8px;text-align:center}
    .sl{display:block;font-size:0.7em;color:#888;margin-bottom:4px}
    .sv{display:block;font-size:1em;font-weight:bold;color:#7ec8e3}
    .ai-banner{margin-top:12px; padding:10px; background:#1e1e3a; border-radius:6px; font-size:0.9em; color:#ddd; border-left:3px solid #ffd700;}
    """
    
    def render_card(d):
        return f"""
        <div class="game" style="border-left-color:{d.get('Color', '#888')}">
            <div class="gh">
                <span>{d['Ticker']} <span style="color:#88ff44; font-size:0.8em">${d.get('Price', 0)}</span></span>
                <span style="color:{d.get('Color', '#888')}; font-size:0.8em">{d['Signal']}</span>
            </div>
            <div class="sgrid">
                <div class="sc"><span class="sl">F-SCORE</span><span class="sv">{d.get('F-Score', 'N/A')}{'/9' if d.get('F-Score') != 'N/A' else ''}</span></div>
                <div class="sc"><span class="sl">Z-SCORE</span><span class="sv">{d.get('Z-Score', 'N/A')}</span></div>
                <div class="sc"><span class="sl">ROIC</span><span class="sv">{d.get('ROIC', 'N/A')}{'%' if d.get('ROIC') != 'N/A' else ''}</span></div>
                <div class="sc"><span class="sl">INTRINSIC VAL</span><span class="sv">${d.get('Intrinsic Value', 'N/A')}</span></div>
                <div class="sc"><span class="sl">MARGIN OF SFTY</span><span class="sv" style="color:{'#00ff88' if isinstance(d.get('Margin of Safety'), (int, float)) and d.get('Margin of Safety') > 30 else '#888'}">{d.get('Margin of Safety', 'N/A')}%</span></div>
            </div>
            <div class="ai-banner">🧠 AI Qualitative Check: <b>{d.get('AI_Signal', 'N/A')}</b></div>
        </div>
        """
        
    html = f"""<!DOCTYPE html><html>
    <head><title>Quant Value Dashboard</title><style>{css}</style><meta name="viewport" content="width=device-width,initial-scale=1"></head>
    <body>
      <h1>📈 Autonomous Quant Value Engine</h1>
      <p class="sub">Last scan: {now_pt}</p>
      <h2 style="color:#00ff88">🟢 Cleared for Purchase</h2>
      {''.join(render_card(d) for d in buys) if buys else '<p style="color:#888">No stocks passed the brutal filter today. Cash is king.</p>'}
      <h2 style="color:#ff6b6b">🔴 Math Rejects</h2>
      {''.join(render_card(d) for d in rejects) if rejects else '<p style="color:#888">No rejects logged.</p>'}
      <h2 style="color:#888888">⚪️ Skipped / Data Errors</h2>
      {''.join(render_card(d) for d in skips) if skips else ''}
    </body></html>"""
    return html

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000, debug=False)
