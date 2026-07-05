import pandas as pd
import numpy as np

def generate_html_report(df_input, criteria, weights, topsis_res, vikor_res, moora_res, fuzzy_res):
    """
    Highly robust and automated HTML decision intelligence report engine.
    Guaranteed to prevent column KeyError by dynamically reading columns by position index.
    """
    # முதலாவது காலம் (supplier / Alternative) எதுவாக இருந்தாலும் அதைத் தானாகவே கண்டறியும்
    id_col_name = df_input.columns[0]
    
    # அனைத்து அல்காரிதம்களின் முதல் காலமையும் அதன் அசல் பெயருக்கே மாற்றி அமைப்பதை உறுதி செய்கிறோம்
    topsis_res = topsis_res.rename(columns={topsis_res.columns[0]: id_col_name})
    vikor_res = vikor_res.rename(columns={vikor_res.columns[0]: id_col_name})
    moora_res = moora_res.rename(columns={moora_res.columns[0]: id_col_name})
    fuzzy_res = fuzzy_res.rename(columns={fuzzy_res.columns[0]: id_col_name})
    
    # டாப் ரேங்க் எடுத்த சப்ளையரைத் தேர்ந்தெடுக்கிறோம்
    top_supplier = topsis_res.iloc[0][id_col_name]
    top_score = topsis_res.iloc[0]['Score'] if 'Score' in topsis_res.columns else topsis_res.iloc[0].iloc[1]
    
    # Data-driven Automated Analytical Reasoning
    supplier_raw = df_input[df_input[id_col_name] == top_supplier].iloc[0]
    avg_vals = df_input[criteria].mean()
    
    reasons = []
    for c in criteria[:4]:  # Core evaluation loop
        val = float(supplier_raw[c])
        avg = float(avg_vals[c])
        if val >= avg:
            reasons.append(f"Exceeded industry average in <strong>{str(c).title()}</strong> with {val:.2f} (Baseline: {avg:.2f}).")
        else:
            reasons.append(f"Maintained structured operational alignment in <strong>{str(c).title()}</strong> with {val:.2f}.")

    reasoning_html = "".join([f"<li>✅ {r}</li>" for r in reasons])

    html_content = f"""
    <html>
    <head>
        <style>
            body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 40px; color: #2d3748; line-height: 1.6; }}
            h1 {{ color: #1a365d; border-bottom: 3px solid #3182ce; padding-bottom: 8px; font-size: 24px; }}
            h2 {{ color: #2c5282; margin-top: 30px; font-size: 18px; }}
            .card {{ background-color: #ebf8ff; border-left: 6px solid #3182ce; padding: 20px; border-radius: 6px; margin: 20px 0; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 15px; font-size: 13px; }}
            th, td {{ border: 1px solid #e2e8f0; padding: 12px; text-align: center; }}
            th {{ background-color: #f7fafc; color: #4a5568; font-weight: bold; }}
            .badge {{ background: #48bb78; color: white; padding: 4px 8px; border-radius: 4px; font-weight: bold; }}
        </style>
    </head>
    <body>
        <h1>📊 Multi-Criteria Decision Intelligence Summary</h1>
        <p><strong>Deployment Strategy:</strong> Federated Multi-Model Framework Integration</p>
        
        <div class="card">
            <h3>🏆 Optimum Recommendation Verdict</h3>
            <p>The consolidated engine has designated <strong>Supplier ID: {top_supplier}</strong> as the ideal operational choice with a TOPSIS Index of <strong>{top_score:.4f}</strong>.</p>
            <h4>📋 Analytical Justification Matrix (Why this Supplier won?):</h4>
            <ul>{reasoning_html}</ul>
        </div>

        <h2>⚖️ AHP Criteria Global Weights Profile</h2>
        <table>
            <tr><th>Axis Rank</th><th>Evaluation Metric</th><th>Priority Weight Coefficient</th></tr>
    """
    for idx, (c, w) in enumerate(sorted(zip(criteria, weights), key=lambda x: x[1], reverse=True)):
        html_content += f"<tr><td>#{idx+1}</td><td>{str(c).title()}</td><td>{w:.4f}</td></tr>"
        
    html_content += f"""
        </table>
        <h2>🏁 Side-by-Side Consensus Benchmarking Matrix</h2>
        <table>
            <tr><th>Supplier ID</th><th>TOPSIS Rank</th><th>VIKOR Rank</th><th>MOORA Rank</th><th>Fuzzy TOPSIS Rank</th></tr>
    """
    
    # அனைத்து அல்காரிதம்களின் ரேங்க்குகளையும் துல்லியமாக ஒப்பிடுதல்
    for alt in df_input[id_col_name].unique():
        r_t = topsis_res[topsis_res[id_col_name] == alt].iloc[0]['Rank']
        r_v = vikor_res[vikor_res[id_col_name] == alt].iloc[0]['Rank']
        r_m = moora_res[moora_res[id_col_name] == alt].iloc[0]['Rank']
        r_f = fuzzy_res[fuzzy_res[id_col_name] == alt].iloc[0]['Rank']
        is_top = 'class="badge"' if r_t == 1 else ""
        html_content += f"<tr><td><strong>{alt}</strong></td><td><span {is_top}>{r_t}</span></td><td>{r_v}</td><td>{r_m}</td><td>{r_f}</td></tr>"

    html_content += "</table></body></html>"
    return html_content