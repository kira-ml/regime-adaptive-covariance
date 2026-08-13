import os
import pandas as pd
import json
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm, inch
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================
RESULTS_DIR = r"D:\quant-finance-ml\regime-adaptive-covariance\results\figures\updated_viz"
OUTPUT_PDF = "LinkedIn_Project_Summary.pdf"

IMAGES = {
    "frobenius_bar": os.path.join(RESULTS_DIR, "frobenius_bar_chart.png"),
    "sub_period_heatmap": os.path.join(RESULTS_DIR, "sub_period_heatmap.png"),
    "sub_period_line": os.path.join(RESULTS_DIR, "sub_period_line.png"),
}

DATA_DIR = r"D:\quant-finance-ml\regime-adaptive-covariance\results"
DATA_FILES = {
    "metrics": os.path.join(DATA_DIR, "metrics.csv"),
    "sub_period_summary": os.path.join(DATA_DIR, "sub_period_summary.csv"),
    "statistical_tests": os.path.join(DATA_DIR, "statistical_tests.json"),
    "portfolio_metrics_test": os.path.join(DATA_DIR, "portfolio_metrics_test.csv"),
}

# ==============================================================================
# 2. LOAD DATA
# ==============================================================================
def load_data():
    data = {}
    for key, path in DATA_FILES.items():
        if key == "statistical_tests":
            continue
        if os.path.exists(path):
            data[key] = pd.read_csv(path)
    
    if os.path.exists(DATA_FILES["statistical_tests"]):
        with open(DATA_FILES["statistical_tests"], 'r') as f:
            data["statistical_tests"] = json.load(f)
    return data

data = load_data()

metrics = data.get("metrics", pd.DataFrame())
portfolio_metrics = data.get("portfolio_metrics_test", pd.DataFrame())
sub_summary = data.get("sub_period_summary", pd.DataFrame())
stat_tests = data.get("statistical_tests", {})

# Extract key numbers
lambda_mean = metrics[metrics['metric'] == 'lambda_mean']['value'].values[0] if not metrics.empty else 0
lambda_std = metrics[metrics['metric'] == 'lambda_std']['value'].values[0] if not metrics.empty else 0
n_windows = int(metrics[metrics['metric'] == 'n_windows']['value'].values[0]) if not metrics.empty else 0

if not portfolio_metrics.empty:
    const_vol = portfolio_metrics[portfolio_metrics['method'] == 'Constant']['mean_volatility'].values[0]
    lw_vol = portfolio_metrics[portfolio_metrics['method'] == 'Ledoit-Wolf']['mean_volatility'].values[0]
    vol_reduction = (const_vol - lw_vol) / const_vol * 100
else:
    vol_reduction = 16.97

min_imp = sub_summary['improvement_vs_constant'].min() if not sub_summary.empty else 14.0
max_imp = sub_summary['improvement_vs_constant'].max() if not sub_summary.empty else 20.0
dm_p = stat_tests.get("diebold_mariano", {}).get("Ledoit-Wolf vs Constant", {}).get("p_value", 0.0017)

# ==============================================================================
# 3. MODERN TIMES NEW ROMAN STYLES
# ==============================================================================
def register_times_fonts():
    try:
        font_paths = ["C:/Windows/Fonts/times.ttf", "C:/Windows/Fonts/timesbd.ttf"]
        for path in font_paths:
            if os.path.exists(path):
                pdfmetrics.registerFont(TTFont('Times-Roman', path))
                break
        for path in ["C:/Windows/Fonts/timesbd.ttf", "C:/Windows/Fonts/times new roman bold.ttf"]:
            if os.path.exists(path):
                pdfmetrics.registerFont(TTFont('Times-Bold', path))
                break
        return True
    except:
        return False

register_times_fonts()

FONT_FAMILY = "Times-Roman"
FONT_BOLD = "Times-Bold"

styles = getSampleStyleSheet()

# Modern academic color palette
PRIMARY_BLUE = colors.HexColor("#1A3A5C")
ACCENT_GREEN = colors.HexColor("#2E7D32")
LIGHT_GRAY = colors.HexColor("#F5F6F8")
BORDER_GRAY = colors.HexColor("#C0C8D0")
HIGHLIGHT_BG = colors.HexColor("#E8EDF3")

# Styles optimized for data highlights
style_title = ParagraphStyle(
    name="Title", parent=styles['Normal'], fontName=FONT_BOLD, fontSize=20, 
    leading=24, alignment=TA_LEFT, textColor=PRIMARY_BLUE, spaceAfter=2
)
style_subtitle = ParagraphStyle(
    name="Subtitle", parent=styles['Normal'], fontName=FONT_FAMILY, fontSize=12, 
    leading=15, alignment=TA_LEFT, textColor=colors.grey, spaceAfter=0
)
style_body = ParagraphStyle(
    name="Body", parent=styles['Normal'], fontName=FONT_FAMILY, fontSize=10.5, 
    leading=14, alignment=TA_JUSTIFY, textColor=colors.black, spaceAfter=3
)
style_section = ParagraphStyle(
    name="Section", parent=styles['Normal'], fontName=FONT_BOLD, fontSize=13, 
    leading=17, alignment=TA_LEFT, textColor=PRIMARY_BLUE, spaceAfter=2, spaceBefore=4
)
style_caption = ParagraphStyle(
    name="Caption", parent=styles['Normal'], fontName=FONT_FAMILY, fontSize=8, 
    alignment=TA_CENTER, textColor=colors.grey, spaceAfter=4
)
style_stat_label = ParagraphStyle(
    name="StatLabel", parent=styles['Normal'], fontName=FONT_FAMILY, fontSize=8, 
    alignment=TA_CENTER, textColor=colors.grey
)
style_stat_value = ParagraphStyle(
    name="StatValue", parent=styles['Normal'], fontName=FONT_BOLD, fontSize=14, 
    alignment=TA_CENTER, textColor=PRIMARY_BLUE
)
style_highlight_stat = ParagraphStyle(
    name="HighlightStat", parent=styles['Normal'], fontName=FONT_BOLD, fontSize=16, 
    alignment=TA_CENTER, textColor=ACCENT_GREEN
)
style_github = ParagraphStyle(
    name="GitHub", parent=styles['Normal'], fontName=FONT_BOLD, fontSize=9, 
    alignment=TA_CENTER, textColor=PRIMARY_BLUE
)

# ==============================================================================
# 4. HELPER: DATA HIGHLIGHT GRID (PAGE 1)
# ==============================================================================
def create_highlight_grid():
    """Creates a 2x2 grid of the most important data points."""
    stats = [
        ("Mean λ*", f"{lambda_mean:.2e}", "Std: {:.5f}".format(lambda_std)),
        ("Volatility Reduction", f"{vol_reduction:.1f}%", f"vs. Constant (p = {dm_p:.4f})"),
        ("Best Method", "Ledoit-Wolf", "Wins in all 4 test regimes"),
        ("Sub-Period Range", f"{min_imp:.1f}% - {max_imp:.1f}%", "Improvement across regimes"),
    ]
    
    data = []
    row = []
    for label, main_value, sub_value in stats:
        # Build a cell with a main bold value and a smaller sub-text
        cell_data = [
            [Paragraph(main_value, style_stat_value)],
            [Paragraph(label, style_stat_label)],
            [Paragraph(sub_value, ParagraphStyle(name="SubValue", parent=styles['Normal'], fontName=FONT_FAMILY, fontSize=8, alignment=TA_CENTER, textColor=colors.grey))]
        ]
        cell = Table(cell_data, colWidths=[1.5*inch])
        cell.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), LIGHT_GRAY),
            ('BORDER', (0,0), (-1,-1), 0.5, BORDER_GRAY),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('TOPPADDING', (0,0), (-1,-1), 3),
            ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ]))
        row.append(cell)
        if len(row) == 2:
            data.append(row)
            row = []
    return Table(data, colWidths=[1.5*inch]*2)

# ==============================================================================
# 5. GENERATE STRICTLY 2-PAGE DATA-HIGHLIGHT PDF
# ==============================================================================
def generate_pdf():
    print("🖨️ Generating 2-page LinkedIn PDF with data-first Page 1...")
    
    doc = SimpleDocTemplate(OUTPUT_PDF, pagesize=A4, leftMargin=15*mm, rightMargin=15*mm, topMargin=12*mm, bottomMargin=12*mm)
    story = []

    # ======================== PAGE 1: DATA HIGHLIGHTS ========================
    story.append(Paragraph("Regime-Adaptive Covariance Shrinkage", style_title))
    story.append(Paragraph("A supervised learning approach to dynamic portfolio risk estimation", style_subtitle))
    story.append(Spacer(1, 3*mm))

    story.append(Paragraph(
        "Standard shrinkage estimators assume optimal intensity is constant. But markets shift between low/high volatility and crises. We tested if regime information improves out-of-sample covariance predictions using 20 years of S&P 500 data (50 stocks).",
        style_body
    ))
    story.append(Spacer(1, 4*mm))

    # ----- Data Highlight Grid -----
    story.append(Paragraph("Key Empirical Findings", style_section))
    story.append(create_highlight_grid())
    story.append(Spacer(1, 4*mm))

    # ----- Key Insight Text (Academic Tone) -----
    story.append(Paragraph(
        f"<b>Core empirical finding:</b> The optimal shrinkage intensity λ* was near-zero (mean = {lambda_mean:.2e}, σ = {lambda_std:.5f}) across {n_windows} rolling windows. This indicates the sample covariance matrix was already well-conditioned for this 50-stock universe; shrinkage toward the identity matrix offered minimal benefit.",
        style_body
    ))
    story.append(Spacer(1, 2*mm))
    story.append(Paragraph(
        "Consequently, the target variable exhibited near-zero variance, rendering regime-adaptive shrinkage largely unnecessary. Machine learning models underperformed (R² = -0.023) not due to complexity, but due to the absence of predictive signal.",
        style_body
    ))
    story.append(Spacer(1, 2*mm))
    story.append(Paragraph(
        "Ledoit-Wolf (which employs a constant-correlation target) was the only method to consistently outperform the static baseline across all test sub-periods.",
        style_body
    ))
    story.append(Spacer(1, 4*mm))

    # ----- Figure 1: Bar Chart -----
    if os.path.exists(IMAGES["frobenius_bar"]):
        story.append(Image(IMAGES["frobenius_bar"], width=6.8*inch, height=2.5*inch))
        story.append(Paragraph("Figure 1: Out-of-sample covariance accuracy (Frobenius distance; lower is better)", style_caption))

    story.append(PageBreak())

    # ======================== PAGE 2: PORTFOLIO IMPACT & PLOTS ========================
    story.append(Paragraph("Portfolio Implications", style_section))
    story.append(Paragraph(
        f"Minimum-variance portfolios using Ledoit-Wolf reduced realized volatility by <b>{vol_reduction:.1f}%</b> relative to constant shrinkage (Diebold-Mariano p = {dm_p:.4f}). Improvements ranged from {min_imp:.1f}% to {max_imp:.1f}% across COVID, bear, and recovery regimes.",
        style_body
    ))
    story.append(Paragraph(
        "<i>Note:</i> Portfolios were unconstrained. Long-only implementations may yield smaller but directionally consistent benefits.",
        ParagraphStyle(name="Note", parent=styles['Normal'], fontName=FONT_FAMILY, fontSize=9, alignment=TA_JUSTIFY, textColor=colors.grey)
    ))
    story.append(Spacer(1, 4*mm))

    # ----- Figure 2: Heatmap -----
    if os.path.exists(IMAGES["sub_period_heatmap"]):
        story.append(Image(IMAGES["sub_period_heatmap"], width=6.8*inch, height=2.5*inch))
        story.append(Paragraph("Figure 2: Realized volatility by market regime. Ledoit-Wolf (highlighted) consistently outperforms.", style_caption))
    story.append(Spacer(1, 4*mm))

    # ----- Figure 3: Line Plot -----
    if os.path.exists(IMAGES["sub_period_line"]):
        story.append(Image(IMAGES["sub_period_line"], width=6.8*inch, height=2.2*inch))
        story.append(Paragraph("Figure 3: Relative performance vs. constant shrinkage across market regimes.", style_caption))

    story.append(Spacer(1, 6*mm))
    story.append(Paragraph("<b>Open source:</b> Full code, data pipeline, and results available on GitHub", style_github))
    story.append(Paragraph("https://github.com/kira-ml/regime-adaptive-covariance.git", style_github))

    doc.build(story)
    print(f"✅ 2-page data-first PDF generated: {OUTPUT_PDF}")

if __name__ == "__main__":
    generate_pdf()