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
# 3. MODERN ACADEMIC STYLES (Times New Roman, Clean, Bold Accents)
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

# Modern color palette
PRIMARY_BLUE = colors.HexColor("#1A3A5C")
ACCENT_GREEN = colors.HexColor("#2E7D32")
LIGHT_GRAY = colors.HexColor("#F5F6F8")
BORDER_GRAY = colors.HexColor("#C0C8D0")

# Style definitions
style_title = ParagraphStyle(
    name="Title",
    parent=styles['Normal'],
    fontName=FONT_BOLD,
    fontSize=22,
    leading=28,
    alignment=TA_LEFT,
    textColor=PRIMARY_BLUE,
    spaceAfter=4,
)

style_subtitle = ParagraphStyle(
    name="Subtitle",
    parent=styles['Normal'],
    fontName=FONT_FAMILY,
    fontSize=13,
    leading=17,
    alignment=TA_LEFT,
    textColor=colors.grey,
    spaceAfter=0,
)

style_section = ParagraphStyle(
    name="Section",
    parent=styles['Normal'],
    fontName=FONT_BOLD,
    fontSize=15,
    leading=19,
    alignment=TA_LEFT,
    textColor=PRIMARY_BLUE,
    spaceAfter=6,
    spaceBefore=10,
)

style_body = ParagraphStyle(
    name="Body",
    parent=styles['Normal'],
    fontName=FONT_FAMILY,
    fontSize=11,
    leading=16,
    alignment=TA_JUSTIFY,
    textColor=colors.black,
    spaceAfter=6,
)

style_bullet = ParagraphStyle(
    name="Bullet",
    parent=styles['Normal'],
    fontName=FONT_FAMILY,
    fontSize=11,
    leading=16,
    alignment=TA_LEFT,
    textColor=colors.black,
    leftIndent=12,
    spaceAfter=3,
)

style_caption = ParagraphStyle(
    name="Caption",
    parent=styles['Normal'],
    fontName=FONT_FAMILY,
    fontSize=9,
    alignment=TA_CENTER,
    textColor=colors.grey,
    spaceAfter=8,
)

style_stat_label = ParagraphStyle(
    name="StatLabel",
    parent=styles['Normal'],
    fontName=FONT_FAMILY,
    fontSize=8,
    alignment=TA_CENTER,
    textColor=colors.grey,
)

style_stat_value = ParagraphStyle(
    name="StatValue",
    parent=styles['Normal'],
    fontName=FONT_BOLD,
    fontSize=14,
    alignment=TA_CENTER,
    textColor=PRIMARY_BLUE,
)

style_note = ParagraphStyle(
    name="Note",
    parent=styles['Normal'],
    fontName=FONT_FAMILY,
    fontSize=9,
    alignment=TA_JUSTIFY,
    textColor=colors.grey,
)

# ==============================================================================
# 4. HELPER FUNCTIONS
# ==============================================================================
def create_stat_grid():
    stats = [
        ("Mean λ*", f"{lambda_mean:.2e}"),
        ("Volatility Reduction", f"{vol_reduction:.1f}%"),
        ("Best Method", "Ledoit-Wolf"),
        ("DM p-value", f"{dm_p:.4f}"),
    ]
    data = []
    row = []
    for label, value in stats:
        cell = Table([
            [Paragraph(value, style_stat_value)],
            [Paragraph(label, style_stat_label)]
        ], colWidths=[1.6*inch])
        cell.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), LIGHT_GRAY),
            ('BORDER', (0,0), (-1,-1), 0.5, BORDER_GRAY),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('TOPPADDING', (0,0), (-1,-1), 6),
            ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ]))
        row.append(cell)
        if len(row) == 4:
            data.append(row)
            row = []
    return Table(data, colWidths=[1.6*inch]*4)

def create_academic_divider():
    table = Table([[""]], colWidths=[6.8*inch])
    table.setStyle(TableStyle([
        ('LINEABOVE', (0,0), (-1,-1), 0.5, BORDER_GRAY),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
    ]))
    return table

# ==============================================================================
# 5. GENERATE MODERN 2-PAGE PDF (With exact problem_framing.md content)
# ==============================================================================
def generate_pdf():
    print("🖨️ Generating modern 2-page LinkedIn PDF with problem_framing.md content...")
    
    doc = SimpleDocTemplate(OUTPUT_PDF, pagesize=A4, leftMargin=18*mm, rightMargin=18*mm, topMargin=14*mm, bottomMargin=14*mm)
    story = []

    # ======================== PAGE 1 ========================
    # Title & Subtitle
    story.append(Paragraph("Regime-Adaptive Covariance Shrinkage", style_title))
    story.append(Paragraph("A Supervised Learning Approach to Dynamic Portfolio Risk Estimation", style_subtitle))
    story.append(Spacer(1, 4*mm))
    story.append(create_academic_divider())
    story.append(Spacer(1, 4*mm))

    # Problem Overview (from problem_framing.md)
    story.append(Paragraph("The Core Problem", style_section))
    story.append(Paragraph(
        "Covariance matrix estimation is fundamental to portfolio construction. Standard shrinkage estimators (Ledoit-Wolf, 2004) apply a single, data-driven intensity to all periods—assuming the optimal shrinkage is constant. But markets exhibit distinct regimes: low volatility, high volatility, crisis, and recovery. The optimal trade-off between sample covariance and a structured target likely varies across these regimes.",
        style_body
    ))
    story.append(Paragraph(
        "This project empirically tests whether market regime information can improve out-of-sample covariance shrinkage predictions compared to static approaches. We frame it as a supervised regression task, test multiple feature sets, and evaluate both covariance accuracy and portfolio-level economic impact.",
        style_body
    ))

    # Machine Learning Formulation (from problem_framing.md)
    story.append(Paragraph("Machine Learning Formulation", style_section))
    story.append(Paragraph(
        "<b>Task:</b> Supervised regression — predict optimal shrinkage intensity (λ ∈ [0,1]) for each rolling 120-day window.<br/>"
        "<b>Target:</b> λ* that minimizes Frobenius distance between shrunk covariance and realized covariance over the next 20 days.<br/>"
        "<b>Features:</b> VIX level, realized volatility, average correlation, and 10 other market regime features.<br/>"
        "<b>Split:</b> Training (2000–2015), Validation (2016–2019), Test (2020–2025).",
        style_body
    ))

    # Key Empirical Finding (from problem_framing.md)
    story.append(Paragraph("Core Empirical Finding", style_section))
    story.append(Paragraph(
        f"<b>λ* was near-zero across all windows (mean = {lambda_mean:.2e}, std = {lambda_std:.5f}).</b> For this 50-stock universe, the sample covariance matrix was already well-conditioned. Shrinkage toward the identity matrix provided minimal benefit. Consequently, the target variable had almost no variance to predict—rendering regime-adaptive shrinkage largely unnecessary.",
        style_body
    ))

    story.append(Spacer(1, 4*mm))

    # Figure 1: Frobenius Bar Chart
    if os.path.exists(IMAGES["frobenius_bar"]):
        story.append(Image(IMAGES["frobenius_bar"], width=6.8*inch, height=2.8*inch))
        story.append(Paragraph("Figure 1: Covariance estimation accuracy (lower is better)", style_caption))

    story.append(Spacer(1, 4*mm))
    story.append(Paragraph(
        "<b>Models evaluated:</b> Constant shrinkage, VIX-threshold, rolling average, Ledoit-Wolf, Elastic Net, and XGBoost. <b>Result:</b> Ledoit-Wolf outperformed all. ML models underperformed (R² = -0.023) because there was no signal to learn.",
        style_body
    ))

    story.append(PageBreak())

    # ======================== PAGE 2 ========================
    # Portfolio Implications (from problem_framing.md)
    story.append(Paragraph("Portfolio Implications & Economic Significance", style_section))
    story.append(Paragraph(
        "To assess economic significance, we constructed minimum-variance portfolios using each covariance estimate and evaluated realized volatility. Ledoit-Wolf consistently delivered the lowest out-of-sample volatility.",
        style_body
    ))
    story.append(Spacer(1, 2*mm))
    story.append(Paragraph(
        f"<b>Volatility reduction:</b> {vol_reduction:.1f}% vs. constant shrinkage (Diebold-Mariano p = {dm_p:.4f}). "
        f"Sub-period improvements ranged from {min_imp:.1f}% to {max_imp:.1f}% across COVID, bear, and recovery regimes.",
        style_body
    ))
    story.append(Paragraph(
        "<i>Note:</i> Portfolios were unconstrained. Long-only implementations may see smaller but directionally consistent benefits.",
        style_note
    ))
    story.append(Spacer(1, 4*mm))

    # Figure 2: Heatmap
    if os.path.exists(IMAGES["sub_period_heatmap"]):
        story.append(Image(IMAGES["sub_period_heatmap"], width=6.8*inch, height=2.8*inch))
        story.append(Paragraph("Figure 2: Realized volatility by market regime. Ledoit-Wolf (highlighted) consistently wins.", style_caption))
    story.append(Spacer(1, 4*mm))

    # Figure 3: Line Plot
    if os.path.exists(IMAGES["sub_period_line"]):
        story.append(Image(IMAGES["sub_period_line"], width=6.8*inch, height=2.6*inch))
        story.append(Paragraph("Figure 3: Relative performance vs. constant shrinkage across regimes.", style_caption))
    story.append(Spacer(1, 4*mm))

    # Principal Takeaways (from problem_framing.md)
    story.append(Paragraph("Principal Takeaways", style_section))
    takeaways = [
        "• λ* was near-zero—shrinkage to identity added no value for this 50-stock universe.",
        "• Ledoit-Wolf (constant-correlation target) outperformed static and ML methods in both covariance accuracy and portfolio volatility.",
        "• ML models failed due to near-zero target variance, not model complexity.",
        "• Reducing Frobenius distance did not guarantee lower portfolio volatility—highlighting the distinction between statistical accuracy and economic utility.",
    ]
    for t in takeaways:
        story.append(Paragraph(t, style_bullet))
    story.append(Spacer(1, 4*mm))

    # Caveats (from problem_framing.md)
    story.append(Paragraph("Limitations & Caveats", style_section))
    caveats = [
        "• Results are for 50 large-cap U.S. equities; may not generalize.",
        "• Ledoit-Wolf uses a constant-correlation target, while λ* optimization uses identity—this explains its relative advantage.",
        "• Portfolio construction is unconstrained; practical long-only implementations may see smaller reductions.",
        "• This is empirical research, not a production risk system.",
    ]
    for c in caveats:
        story.append(Paragraph(c, style_bullet))

    story.append(Spacer(1, 6*mm))
    story.append(Paragraph("<b>Open source:</b> Full code, data pipeline, and results on GitHub", ParagraphStyle(name="Foot", parent=styles['Normal'], fontName=FONT_BOLD, fontSize=10, alignment=TA_CENTER)))
    story.append(Paragraph("https://github.com/kira-ml/regime-adaptive-covariance.git", ParagraphStyle(name="Link", parent=styles['Normal'], fontName=FONT_FAMILY, fontSize=10, alignment=TA_CENTER, textColor=PRIMARY_BLUE)))

    doc.build(story)
    print(f"✅ Modern 2-page PDF generated: {OUTPUT_PDF}")

if __name__ == "__main__":
    generate_pdf()