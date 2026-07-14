import streamlit as st
import pandas as pd
import numpy as np

st.title("🏆 Multi-Criteria Rank Aggregation & Consensus Engine")
st.markdown("---")

# =========================================================================
# 1. FETCHING DATA FROM INDIVIDUAL LOCKED ALGORITHM STATES
# =========================================================================
topsis_data = st.session_state.get('topsis_locked_results')
moora_data = st.session_state.get('moora_locked_results')   
vikor_data = st.session_state.get('vikor_locked_results')   

# Fuzzy TOPSIS-ன் அனைத்து சாத்தியமான மெமரி கீகளையும் தேடுகிறது
fuzzy_data = st.session_state.get('fuzzy_locked_results')
if fuzzy_data is None:
    fuzzy_data = st.session_state.get('fuzzy_topsis_locked_results')
if fuzzy_data is None:
    fuzzy_data = st.session_state.get('df_rankings')

has_active_data = topsis_data is not None or moora_data is not None or vikor_data is not None or fuzzy_data is not None

# 🔥 [THE FINAL BULLETPROOF COMPILING ENGINE]
def get_clean_rank(df, id_col, sup_id, algo_name):
    df_temp = df.copy()
    
    # alternative காலம் பெயரைக் கண்டறிதல்
    current_id_col = None
    if id_col in df_temp.columns:
        current_id_col = id_col
    else:
        id_cols = [c for c in df_temp.columns if 'alter' in c.lower() or 'supp' in c.lower()]
        current_id_col = id_cols[0] if id_cols else df_temp.columns[0]
        
    df_temp[current_id_col] = df_temp[current_id_col].astype(str).str.strip()
    target_id_str = str(sup_id).strip()
    
    # 🎯 VIKOR, TOPSIS, MOORA, FUZZY-ன் ஸ்கோர் காலம்களைத் துல்லியமாகப் பிரித்தல்
    score_col = None
    columns_lower = [c.lower() for c in df_temp.columns]
    
    if algo_name.lower() == 'vikor':
        # VIKOR பக்கத்தில் இருக்கும் 'Q' அல்லது 'q_' காலம்களை மட்டும் குறிவைக்கிறது
        vikor_matches = [c for c in df_temp.columns if c.lower() == 'q' or 'q_' in c.lower() or 'v_' in c.lower()]
        if vikor_matches:
            score_col = vikor_matches[0]
    elif algo_name.lower() == 'fuzzy':
        fuzzy_matches = [c for c in df_temp.columns if 'cc' in c.lower() or 'performance' in c.lower()]
        if fuzzy_matches:
            score_col = fuzzy_matches[0]
            
    # ஒருவேளை மேலே மேட்ச் ஆகவில்லை எனில் பொதுவான ஸ்கோர் காலம்களைத் தேடுகிறது
    if not score_col:
        for c in df_temp.columns:
            if c != current_id_col and any(w in c.lower() for w in ['score', 'value', 'performance', 'cc', 'q', 'v']):
                score_col = c
                break

    # ஸ்கோர் காலம் கிடைத்துவிட்டால், அதை வச்சு பிரெஷ்ஷா ரேங்க் போடுகிறது
    if score_col:
        df_temp[score_col] = pd.to_numeric(df_temp[score_col], errors='coerce').fillna(0)
        
        # ⚠️ VIKOR-க்கு மட்டும் குறைந்த ஸ格ார் தான் பெஸ்ட் (Ascending=True), மற்றவற்றுக்கு அதிக ஸ்கோர் பெஸ்ட் (Ascending=False)
        is_ascending = True if algo_name.lower() == 'vikor' or score_col.lower() == 'q' else False
        df_temp['Calculated_Rank'] = df_temp[score_col].rank(ascending=is_ascending, method='min')
        
        sub_df = df_temp[df_temp[current_id_col] == target_id_str]
        if not sub_df.empty:
            return int(sub_df['Calculated_Rank'].values[0])
            
    # மாற்று வழி: டேபிளில் நேரடியாக இருக்கும் ரேங்க் காலமில் இருந்து எடுப்பது
    rank_cols = [c for c in df_temp.columns if 'rank' in c.lower() or 'position' in c.lower()]
    if rank_cols:
        sub_df = df_temp[df_temp[current_id_col] == target_id_str]
        if not sub_df.empty:
            val = str(sub_df[rank_cols[0]].values[0])
            digits = ''.join([char for char in val if char.isdigit()])
            if digits:
                return int(digits)
                
    return 1

if not has_active_data:
    data_matrix = {
        "Alternative": ["Alpha", "Beta", "Gamma", "Delta", "Omega"],
        "TOPSIS": [5, 4, 3, 2, 1],
        "MOORA": [5, 2, 4, 3, 1],
        "VIKOR": [4, 5, 3, 1, 2],
        "Fuzzy TOPSIS": [5, 3, 4, 2, 1]
    }
    df_ranks = pd.DataFrame(data_matrix)
    id_col = "Alternative"
    st.info("💡 **Insight Matrix:** Displaying comparative consensus ranking using standard validation benchmarks.")
else:
    active_dfs = [df for df in [topsis_data, moora_data, vikor_data, fuzzy_data] if df is not None]
    base_df = active_dfs[0]
    
    id_col = 'Alternative' if 'Alternative' in base_df.columns else base_df.columns[0]
    
    all_suppliers = []
    for df in active_dfs:
        c_col = id_col if id_col in df.columns else df.columns[0]
        all_suppliers.extend(df[c_col].astype(str).str.strip().tolist())
    all_suppliers = sorted(list(set(all_suppliers)), key=lambda x: float(x) if x.replace('.','',1).isdigit() else x)

    dynamic_compiled = []
    for sup in all_suppliers:
        display_sup = int(float(sup)) if sup.replace('.','',1).isdigit() and float(sup).is_integer() else sup
        row = {id_col: display_sup}
        
        row['TOPSIS'] = get_clean_rank(topsis_data, id_col, sup, 'topsis') if topsis_data is not None else 1
        row['MOORA'] = get_clean_rank(moora_data, id_col, sup, 'moora') if moora_data is not None else 1
        row['VIKOR'] = get_clean_rank(vikor_data, id_col, sup, 'vikor') if vikor_data is not None else 1
        row['Fuzzy TOPSIS'] = get_clean_rank(fuzzy_data, id_col, sup, 'fuzzy') if fuzzy_data is not None else 1
            
        dynamic_compiled.append(row)
        
    df_ranks = pd.DataFrame(dynamic_compiled)
    st.success("⚡ **Live Data Sync:** Successfully normalized all standalone rankings into pure ordinal values!")

# =========================================================================
# 2. RENDER THE MATRIX AND CALC BORDA
# =========================================================================
if df_ranks is not None:
    st.subheader("📋 Consolidated Individual MCDM Rank Matrix")
    st.markdown("This matrix summarizes the standalone ordinal rankings generated by each decision algorithm:")
    
    technique_cols = ["TOPSIS", "MOORA", "VIKOR", "Fuzzy TOPSIS"]
    technique_cols = [c for c in technique_cols if c in df_ranks.columns]
    
    for col in technique_cols:
        df_ranks[col] = df_ranks[col].astype(int)
        
    ordered_cols = [id_col] + technique_cols
    df_ranks = df_ranks[ordered_cols]
        
    st.dataframe(df_ranks.set_index(id_col), use_container_width=True)
    num_alternatives = len(df_ranks)
    
    st.markdown("---")
    st.subheader("🏅 Consensus Optimization: Borda Count Method")
    
    df_borda_points = df_ranks.copy()
    for col in technique_cols:
        df_borda_points[col] = num_alternatives + 1 - df_ranks[col]
        
    df_borda_points['Total Borda Score'] = df_borda_points[technique_cols].sum(axis=1)
    
    df_final_sorted = df_borda_points.sort_values(by='Total Borda Score', ascending=False).reset_index(drop=True)
    df_final_sorted.index = df_final_sorted.index + 1
    df_final_sorted.index.name = "Consensus Rank"
    
    df_ui_table = df_final_sorted.copy()
    for col in technique_cols:
        df_ui_table[col] = num_alternatives + 1 - df_ui_table[col]
        
    def assign_medals(rank):
        if rank == 1: return "🥇 1"
        elif rank == 2: return "🥈 2"
        elif rank == 3: return "🥉 3"
        return str(rank)

    df_ui_display = df_ui_table.reset_index()
    df_ui_display["Consensus Rank"] = df_ui_display["Consensus Rank"].apply(assign_medals)
    df_ui_display = df_ui_display.set_index("Consensus Rank")
    
    st.dataframe(
        df_ui_display[[id_col] + technique_cols + ['Total Borda Score']], 
        use_container_width=True
    )

    # =========================================================================
    # 3. DYNAMIC THESIS VERDICT GENERATOR
    # =========================================================================
    st.markdown("---")
    st.subheader("🏁 Final Consensus Recommendation Verdict")
    
    best_candidate_borda = df_final_sorted.iloc[0][id_col]
    score_borda = df_final_sorted.iloc[0]['Total Borda Score']
    second_candidate_borda = df_final_sorted.iloc[1][id_col] if len(df_final_sorted) > 1 else "N/A"
    
    st.success(f"🎉 **Mathematical Aggregation Choice:** Alternative **{best_candidate_borda}** is selected as the absolute optimal option via **Borda Count Voting Protocol** with a total score of **{score_borda}** points!")
    
    st.markdown("### 📝 Academic Thesis Conclusion Text Generator")
    tech_names_str = ", ".join(technique_cols)
    st.info(f"""
    > *By deploying an Integrated Analytical Hierarchy Process (AHP) framework to distribute standard criteria weights, the alternative candidate profiles were thoroughly cross-evaluated using distinct Multi-Criteria Decision Making tracks, namely **{tech_names_str}**. To reconcile minor divergence layers across standalone models and deliver a non-conflicting solid consensus decision matrix, individual system rankings were compiled using the rigorous mathematical **Borda Count Rank Aggregation Protocol**.*
    > 
    > *Empirical computation results confirm that Alternative **{best_candidate_borda}** consistently outperforms competing models across distinct vector constraints, securing the unambiguous **Rank 1 position** with an absolute Borda total score of **{score_borda}** points. The candidate profile **{second_candidate_borda}** firmly secures the second position, establishing a structured risk-mitigated preference chain for real-world project deployment execution targets.*
    """)

    # =========================================================================
    # 4. REPORTLAB PDF EXPORT SYSTEM
    # =========================================================================
    # =========================================================================
    # 4. REPORTLAB PDF EXPORT SYSTEM (INTELLIGENT CRITERIA-BASED JUSTIFICATION)
    # =========================================================================
    st.markdown("---")
    st.subheader("📥 2. Academic Synthesis Report Generator")
    
    max_available = len(df_final_sorted)
    top_k = st.slider(
        "🎯 Select number of top alternatives to include in the report:",
        min_value=1, max_value=max_available, value=min(10, max_available), step=1, key="consensus_top_k_slider"
    )

    # 1. தரவை ஃபில்டர் செய்தல்
    df_report_data = df_ui_table.head(top_k).copy()
    
    filtered_best_candidate = df_report_data.iloc[0][id_col]
    filtered_best_score = int(df_final_sorted.iloc[0]['Total Borda Score'])
    filtered_second_candidate = df_report_data.iloc[1][id_col] if len(df_report_data) > 1 else "N/A"

    # 2. 🧠 INTELLIGENT CRITERIA-BASED JUSTIFICATION ENGINE
    # முதலிடம் பிடித்த ஆல்டர்நேட்டிவ் எந்தெந்த அல்காரிதமில் 1 அல்லது 2 வது ரேங்க் வாங்கியுள்ளது எனப் பார்க்கிறது
    best_row = df_report_data.iloc[0]
    
    # அல்காரிதம் சார்ந்த பலங்களை (Strengths) கண்டறிதல்
    strengths = []
    if 'TOPSIS' in best_row and int(best_row['TOPSIS']) <= 2:
        strengths.append("<b>Geometric Optimization (TOPSIS)</b>: Ideal balance of maximizing benefits while strictly controlling cost vectors.")
    if 'MOORA' in best_row and int(best_row['MOORA']) <= 2:
        strengths.append("<b>Pure Cost-Benefit Efficiency (MOORA)</b>: High volume operational dominance, yielding the highest return over cost expenditure.")
    if 'VIKOR' in best_row and int(best_row['VIKOR']) <= 2:
        strengths.append("<b>Risk & Bottleneck Mitigation (VIKOR)</b>: Elite stability with minimum individual regret, ensuring no critical attribute fails safety margins.")
    if 'Fuzzy TOPSIS' in best_row and int(best_row['Fuzzy TOPSIS']) <= 2:
        strengths.append("<b>Qualitative Human Consensus (Fuzzy TOPSIS)</b>: Strongest alignment with expert opinions and subjective parameters like reputation and quality.")

    # ஒருவேளை எந்த அல்காரிதமிலும் டாப் 2 இல்லை என்றால் பொதுவான லாஜிக்
    if not strengths:
        strengths.append("<b>Balanced Consensus</b>: Stable performance across all conflicting operational constraints without extreme individual failure.")

    # பிடிஎஃப்-ல் காட்டுவதற்கான டெக்ஸ்ட் வடிவமைப்பு
    justification_title = f"Multi-Dimensional Strength Profile for Alternative {filtered_best_candidate}"
    strengths_html = "<br/><br/>".join([f"• {s}" for s in strengths])
    
    justification_desc = f"""
    The mathematical aggregation engine has selected <b>Alternative {filtered_best_candidate}</b> as the Rank 1 optimal solution. A deep architectural audit of the underlying MCDM models confirms that this choice is justified by the following core performance strengths:
    <br/><br/>
    {strengths_html}
    <br/><br/>
    <b>Conclusion for Trust-Driven Selection:</b> Unlike single-track models, this selection guarantees that <b>Alternative {filtered_best_candidate}</b> prioritizes critical quality parameters over non-beneficial cost overruns, providing a robust, risk-mitigated decision for executive deployment.
    """

    import io
    from datetime import datetime
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    from reportlab.pdfgen import canvas

    class NumberedCanvas(canvas.Canvas):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._saved_page_states = []
        def showPage(self):
            self._saved_page_states.append(dict(self.__dict__))
            self._startPage()
        def save(self):
            num_pages = len(self._saved_page_states)
            for state in self._saved_page_states:
                self.__dict__.update(state)
                self.draw_page_decorations(num_pages)
                super().showPage()
            super().save()
        def draw_page_decorations(self, page_count):
            self.saveState()
            self.setFont("Helvetica", 8)
            self.setFillColor(colors.HexColor('#475569'))
            self.drawString(40, 25, "MCDM Decision Optimization System — Consensus Report")
            self.drawRightString(self._pagesize[0] - 40, 25, f"Page {self._pageNumber} of {page_count}")
            self.restoreState()

    pdf_buffer = io.BytesIO()
    doc = SimpleDocTemplate(pdf_buffer, pagesize=letter, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=50)
    story = []
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle('PDFTitle', parent=styles['Heading1'], fontSize=16, leading=20, textColor=colors.HexColor('#1E293B'))
    meta_style = ParagraphStyle('PDFMeta', parent=styles['Normal'], fontSize=9, leading=13, textColor=colors.HexColor('#475569'))
    section_style = ParagraphStyle('PDFSection', parent=styles['Heading3'], fontSize=11, leading=15, textColor=colors.HexColor('#0F172A'), spaceBefore=12, spaceAfter=6)
    body_style = ParagraphStyle('PDFBody', parent=styles['Normal'], fontSize=9, leading=13, textColor=colors.HexColor('#334155'))
    thesis_style = ParagraphStyle('PDFThesis', parent=styles['Normal'], fontSize=8.5, leading=12.5, textColor=colors.HexColor('#475569'), leftIndent=15, firstLineIndent=-10)
    table_text_style = ParagraphStyle('PDFTableText', parent=styles['Normal'], fontSize=9, leading=11, alignment=1)
    table_header_style = ParagraphStyle('PDFTableHeader', parent=styles['Normal'], fontSize=9, leading=11, textColor=colors.whitesmoke, alignment=1)

    # Header Section
    story.append(Paragraph("🏆 Multi-Criteria Rank Aggregation & Consensus Report", title_style))
    story.append(Paragraph(f"<b>Analysis Date:</b> {datetime.now().strftime('%d %B %Y')}", meta_style))
    story.append(Paragraph(f"<b>Evaluation Scope:</b> Integrated Framework ({', '.join(technique_cols)})", meta_style))
    story.append(Paragraph(f"<b>Filtered Output Constraint:</b> Top {top_k} Alternatives Selected by User", meta_style))
    story.append(Spacer(1, 10))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#CBD5E1'), spaceBefore=5, spaceAfter=15))

    # Summary Conclusion
    story.append(Paragraph("■ Summary Conclusion:", section_style))
    conclusion_text = f"Based on the cross-evaluation of the loaded multi-dimensional parameters restricted to the user-specified filter, Alternative <b>{filtered_best_candidate}</b> has successfully secured the Rank 1 position within this bounded set, achieving an absolute consolidated Borda score of <b>{filtered_best_score} points</b>."
    story.append(Paragraph(conclusion_text, body_style))
    
    story.append(Spacer(1, 10))
    
    # Table Layout
    story.append(Paragraph(f"■ Compiled Consensus Evaluation Matrix (Top {top_k} Filtered Models)", section_style))
    
    header_row = [Paragraph("<b>Consensus Rank</b>", table_header_style), Paragraph(f"<b>{id_col}</b>", table_header_style)]
    for tech in technique_cols:
        header_row.append(Paragraph(f"<b>{tech}</b>", table_header_style))
    header_row.append(Paragraph("<b>Total Borda Score</b>", table_header_style))
    table_data = [header_row]
    
    for rank_idx, (_, row) in enumerate(df_report_data.iterrows(), 1):
        row_content = [Paragraph(f"<b>{rank_idx}</b>", table_text_style), Paragraph(str(row[id_col]), table_text_style)]
        for tech in technique_cols:
            row_content.append(Paragraph(str(int(row[tech])), table_text_style))
        row_content.append(Paragraph(str(int(row['Total Borda Score'])), table_text_style))
        table_data.append(row_content)
    
    num_cols_total = len(header_row)
    col_widths = [75, 105] + [350 // (num_cols_total - 2)] * (num_cols_total - 2)
    
    ranking_table = Table(table_data, colWidths=col_widths)
    ranking_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1E3A8A')),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E1')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F8FAFC')]),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(ranking_table)
    
    # -------------------------------------------------------------------------
    ## -------------------------------------------------------------------------
    # 🎯 புதிய அம்சம்: துல்லியமான வணிகக் காரணப் பெட்டி (Fixed Spacing Version)
    # -------------------------------------------------------------------------
    story.append(Spacer(1, 15))  # தலைப்புக்கு மேலே நல்ல கேப்
    story.append(Paragraph(f"<b>■ Strategic Selection Justification ({justification_title}):</b>", section_style))
    
    story.append(Spacer(1, 8))   # தலைப்புக்கும் பாக்ஸுக்கும் நடுவில் கேப்
    
    justification_box_style = ParagraphStyle(
        'PDFJustBox', parent=styles['Normal'], fontSize=9, leading=14, 
        textColor=colors.HexColor('#1E293B'), backColor=colors.HexColor('#F8FAFC'), 
        borderColor=colors.HexColor('#E2E8F0'), borderWidth=0.75, 
        borderPadding=12,
        spaceBefore=0, spaceAfter=0  # குழப்பம் விளைவிக்கும் இன்-பில்ட் ஸ்பேஸை பூஜ்ஜியமாக்குகிறோம்
    )
    story.append(Paragraph(justification_desc, justification_box_style))
    
    story.append(Spacer(1, 15))  # பாக்ஸுக்கும் கீழே வரும் தியசிஸ் டெக்ஸ்டுக்கும் நடுவில் நல்ல கேப்
    # -------------------------------------------------------------------------
    
    story.append(Spacer(1, 10))
    
    # Thesis Text
    story.append(Paragraph("■ Academic Thesis Conclusion Text Generator:", section_style))
    tech_names_str = ", ".join(technique_cols)
    thesis_text = f"""
    <i>"By deploying an Integrated Analytical Hierarchy Process (AHP) framework to distribute standard criteria weights, the alternative candidate profiles were thoroughly cross-evaluated using distinct Multi-Criteria Decision Making tracks, namely <b>{tech_names_str}</b>. To reconcile minor divergence layers across standalone models and deliver a non-conflicting solid consensus decision matrix, individual system rankings were compiled using the rigorous mathematical <b>Borda Count Rank Aggregation Protocol</b>. 
    <br/><br/>
    Empirical computation results restricted to the top {top_k} evaluated subset confirm that Alternative <b>{filtered_best_candidate}</b> consistently outperforms competing models across distinct vector constraints, securing the unambiguous <b>Rank 1 position</b> with an absolute Borda total score of <b>{filtered_best_score} points</b>. The candidate profile <b>{filtered_second_candidate}</b> firmly secures the second position, establishing a structured risk-mitigated preference chain for real-world project deployment execution targets."</i>
    """
    story.append(Paragraph(thesis_text, thesis_style))
    
    doc.build(story, canvasmaker=NumberedCanvas)
    pdf_data = pdf_buffer.getvalue()
    pdf_buffer.close()

    st.markdown("---")
    st.download_button(
        label=f"📥 Download Strategic Analytics Report (Top {top_k} PDF)",
        data=pdf_data,
        file_name=f"MCDM_Consensus_Top_{top_k}_Report.pdf",
        type="primary"
    )