# ─── ROUTES ───────────────────────────────────────────────────────────────────
def process_all_tickers():
    results = []
    for ticker in TARGET_TICKERS:
        math_result = run_quant_math(ticker)
        if "SKIP" in math_result['Signal']: 
            # Fill in dummy data so the HTML card doesn't break
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
    """Returns the JSON payload of the stock scan."""
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
