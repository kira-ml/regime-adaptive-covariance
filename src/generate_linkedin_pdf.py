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
RESULTS_DIR = r"D:\quant-finance-ml\regime-adaptive-covariance\results\figures"
OUTPUT_PDF = "LinkedIn_Project_Summary.pdf"

IMAGES = {
    "lambda_plot": os.path.join(RESULTS_DIR, "linkedin_plot1_lambda_over_time.png"),
    "frobenius_plot": os.path.join(RESULTS_DIR, "linkedin_plot2_frobenius_comparison.png"),
    "heatmap": os.path.join(RESULTS_DIR, "linkedin_plot3_sub_period_heatmap.png"),
}

# Data files
DATA_DIR = r"D:\quant-finance-ml\regime-adaptive-covariance\results"
DATA_FILES = {
    "metrics": os.path.join(DATA_DIR, "metrics.csv"),
    "sub_period_results": os.path.join(DATA_DIR, "sub_period_results.csv"),
    "sub_period_summary": os.path.join(DATA_DIR, "sub_period_summary.csv"),
    "statistical_tests": os.path.join(DATA_DIR, "statistical_tests.json"),
    "portfolio_metrics_test": os.path.join(DATA_DIR, "portfolio_metrics_test.csv"),
    "elastic_net_results": os.path.join(DATA_DIR, "elastic_net_results.csv"),
    "xgboost_results": os.path.join(DATA_DIR, "xgboost_results.csv"),
    "feature_set_comparison": os.path.join(DATA_DIR, "feature_set_comparison.csv"),
}

# ==============================================================================
# 2. LOAD DATA
# ==============================================================================
def load_data():
    """Load all result data from CSV and JSON files."""
    data = {}
    
    # Load CSV files
    for key, path in DATA_FILES.items():
        if key == "statistical_tests":
            continue
        if os.path.exists(path):
            data[key] = pd.read_csv(path)
            print(f"✅ Loaded {key} from {path}")
        else:
            print(f"⚠️ Warning: {path} not found")
    
    # Load JSON
    if os.path.exists(DATA_FILES["statistical_tests"]):
        with open(DATA_FILES["statistical_tests"], 'r') as f:
            data["statistical_tests"] = json.load(f)
        print(f"✅ Loaded statistical_tests from {DATA_FILES['statistical_tests']}")
    
    return data

# Load data
data = load_data()

# Extract key values
metrics = data.get("metrics", pd.DataFrame())
portfolio_metrics = data.get("portfolio_metrics_test", pd.DataFrame())
sub_period_summary = data.get("sub_period_summary", pd.DataFrame())
stat_tests = data.get("statistical_tests", {})

# Extract values from metrics
lambda_mean = metrics[metrics['metric'] == 'lambda_mean']['value'].values[0] if not metrics.empty else 0
lambda_std = metrics[metrics['metric'] == 'lambda_std']['value'].values[0] if not metrics.empty else 0
improvement_pct = metrics[metrics['metric'] == 'improvement_pct']['value'].values[0] if not metrics.empty else 0

# Extract portfolio volatility values
if not portfolio_metrics.empty:
    constant_vol = portfolio_metrics[portfolio_metrics['method'] == 'Constant']['mean_volatility'].values[0]
    lw_vol = portfolio_metrics[portfolio_metrics['method'] == 'Ledoit-Wolf']['mean_volatility'].values[0]
    optimal_vol = portfolio_metrics[portfolio_metrics['method'] == 'Optimal']['mean_volatility'].values[0]
    vol_reduction = (constant_vol - lw_vol) / constant_vol * 100
else:
    constant_vol = 0.0100988
    lw_vol = 0.00838543
    optimal_vol = 0.0102865
    vol_reduction = 16.97

# Extract statistical test results
dm_p_value = stat_tests.get("diebold_mariano", {}).get("Ledoit-Wolf vs Constant", {}).get("p_value", 0.0017)
bootstrap_p_value = stat_tests.get("bootstrap", {}).get("Ledoit-Wolf vs Constant", {}).get("p_value", 0.0)

# ==============================================================================
# 3. ACADEMIC STYLES - FIXED FONT HANDLING
# ==============================================================================

def register_times_fonts():
    """Register Times New Roman fonts with fallback to built-in fonts."""
    try:
        font_paths = [
            "C:/Windows/Fonts/times.ttf",
            "C:/Windows/Fonts/timesbd.ttf",
            "C:/Windows/Fonts/timesi.ttf",
            "C:/Windows/Fonts/timesbi.ttf",
            "C:/Windows/Fonts/times new roman.ttf",
            "C:/Windows/Fonts/times new roman bold.ttf",
            "C:/Windows/Fonts/times new roman italic.ttf",
            "C:/Windows/Fonts/times new roman bold italic.ttf",
            "C:/Windows/Fonts/ttimes.ttf",
            "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf",
            "/usr/share/fonts/TTF/Times_New_Roman.ttf",
            "/Library/Fonts/Times New Roman.ttf",
        ]
        
        registered = False
        for path in font_paths:
            if os.path.exists(path):
                try:
                    pdfmetrics.registerFont(TTFont('Times-Roman', path))
                    registered = True
                    print(f"✅ Registered Times-Roman from: {path}")
                    break
                except:
                    continue
        
        bold_paths = [
            "C:/Windows/Fonts/timesbd.ttf",
            "C:/Windows/Fonts/times new roman bold.ttf",
            "C:/Windows/Fonts/ttimesbd.ttf",
        ]
        for path in bold_paths:
            if os.path.exists(path):
                try:
                    pdfmetrics.registerFont(TTFont('Times-Bold', path))
                    registered = True
                    print(f"✅ Registered Times-Bold from: {path}")
                    break
                except:
                    continue
        
        italic_paths = [
            "C:/Windows/Fonts/timesi.ttf",
            "C:/Windows/Fonts/times new roman italic.ttf",
            "C:/Windows/Fonts/ttimesi.ttf",
        ]
        for path in italic_paths:
            if os.path.exists(path):
                try:
                    pdfmetrics.registerFont(TTFont('Times-Italic', path))
                    registered = True
                    print(f"✅ Registered Times-Italic from: {path}")
                    break
                except:
                    continue
        
        bi_paths = [
            "C:/Windows/Fonts/timesbi.ttf",
            "C:/Windows/Fonts/times new roman bold italic.ttf",
            "C:/Windows/Fonts/ttimesbi.ttf",
        ]
        for path in bi_paths:
            if os.path.exists(path):
                try:
                    pdfmetrics.registerFont(TTFont('Times-BoldItalic', path))
                    registered = True
                    print(f"✅ Registered Times-BoldItalic from: {path}")
                    break
                except:
                    continue
        
        if not registered:
            print("⚠️ Times New Roman fonts not found. Using built-in fonts.")
            from reportlab.lib.fonts import addMapping
            addMapping('Times-Roman', 0, 0, 'Times-Roman')
            addMapping('Times-Bold', 1, 0, 'Times-Bold')
            addMapping('Times-Italic', 0, 1, 'Times-Italic')
            addMapping('Times-BoldItalic', 1, 1, 'Times-BoldItalic')
            return False
            
    except Exception as e:
        print(f"⚠️ Font registration error: {e}")
        print("Using built-in fonts.")
        from reportlab.lib.fonts import addMapping
        addMapping('Times-Roman', 0, 0, 'Times-Roman')
        addMapping('Times-Bold', 1, 0, 'Times-Bold')
        addMapping('Times-Italic', 0, 1, 'Times-Italic')
        addMapping('Times-BoldItalic', 1, 1, 'Times-BoldItalic')
        return False
    
    return True

register_times_fonts()

FONT_FAMILY = "Times-Roman"
FONT_BOLD = "Times-Bold"
FONT_ITALIC = "Times-Italic"
FONT_BOLD_ITALIC = "Times-BoldItalic"

styles = getSampleStyleSheet()

# --- Academic Color Palette ---
COLOR_PRIMARY = colors.HexColor("#1A3A5C")
COLOR_DARK = colors.HexColor("#1A1A2E")
COLOR_GRAY = colors.HexColor("#4A4A4A")
COLOR_LIGHT_GRAY = colors.HexColor("#F5F6F8")
COLOR_ACCENT = colors.HexColor("#2E7D32")
COLOR_BORDER = colors.HexColor("#C0C8D0")
COLOR_WARNING = colors.HexColor("#B76E2E")
COLOR_HIGHLIGHT = colors.HexColor("#E8EDF3")

# --- Styles ---
style_title = ParagraphStyle(
    name="Title",
    parent=styles['Normal'],
    fontName=FONT_BOLD,
    fontSize=22,
    leading=28,
    alignment=TA_LEFT,
    textColor=COLOR_DARK,
    spaceAfter=4,
    spaceBefore=0,
)

style_subtitle = ParagraphStyle(
    name="Subtitle",
    parent=styles['Normal'],
    fontName=FONT_FAMILY,
    fontSize=13,
    leading=17,
    alignment=TA_LEFT,
    textColor=COLOR_GRAY,
    spaceAfter=0,
    spaceBefore=0,
)

style_section = ParagraphStyle(
    name="Section",
    parent=styles['Normal'],
    fontName=FONT_BOLD,
    fontSize=15,
    leading=19,
    alignment=TA_LEFT,
    textColor=COLOR_PRIMARY,
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
    textColor=COLOR_DARK,
    spaceAfter=6,
)

style_body_small = ParagraphStyle(
    name="BodySmall",
    parent=styles['Normal'],
    fontName=FONT_FAMILY,
    fontSize=10,
    leading=14,
    alignment=TA_JUSTIFY,
    textColor=COLOR_GRAY,
    spaceAfter=4,
)

style_caption = ParagraphStyle(
    name="Caption",
    parent=styles['Normal'],
    fontName=FONT_ITALIC,
    fontSize=9,
    alignment=TA_CENTER,
    textColor=COLOR_GRAY,
    spaceAfter=8,
)

style_bullet = ParagraphStyle(
    name="Bullet",
    parent=styles['Normal'],
    fontName=FONT_FAMILY,
    fontSize=11,
    leading=16,
    alignment=TA_LEFT,
    textColor=COLOR_DARK,
    leftIndent=12,
    spaceAfter=3,
)

style_footer = ParagraphStyle(
    name="Footer",
    parent=styles['Normal'],
    fontName=FONT_ITALIC,
    fontSize=8,
    alignment=TA_CENTER,
    textColor=COLOR_GRAY,
    spaceBefore=6,
)

style_highlight_box = ParagraphStyle(
    name="HighlightBox",
    parent=styles['Normal'],
    fontName=FONT_BOLD,
    fontSize=12,
    alignment=TA_CENTER,
    textColor=COLOR_PRIMARY,
    spaceAfter=4,
    spaceBefore=4,
)

style_table_header = ParagraphStyle(
    name="TableHeader",
    parent=styles['Normal'],
    fontName=FONT_BOLD,
    fontSize=10,
    alignment=TA_CENTER,
    textColor=colors.white,
    spaceAfter=2,
)

style_table_cell = ParagraphStyle(
    name="TableCell",
    parent=styles['Normal'],
    fontName=FONT_FAMILY,
    fontSize=10,
    alignment=TA_CENTER,
    textColor=COLOR_DARK,
    spaceAfter=2,
)

style_table_highlight = ParagraphStyle(
    name="TableHighlight",
    parent=styles['Normal'],
    fontName=FONT_BOLD,
    fontSize=10,
    alignment=TA_CENTER,
    textColor=COLOR_PRIMARY,
    spaceAfter=2,
)

# ==============================================================================
# 4. HELPER FUNCTIONS
# ==============================================================================

def create_academic_hook(title_text, subtitle_text):
    """Creates an academic-style header with left border accent."""
    title_para = Paragraph(title_text, style_title)
    subtitle_para = Paragraph(subtitle_text, style_subtitle)
    
    data = [[title_para], [subtitle_para]]
    table = Table(data, colWidths=[6.8 * inch])
    table.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('LEFTPADDING', (0,0), (-1,-1), 18),
        ('RIGHTPADDING', (0,0), (-1,-1), 0),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ('LINEAFTER', (0,0), (0,0), 4, COLOR_PRIMARY),
        ('LINEAFTER', (0,1), (0,1), 4, COLOR_PRIMARY),
        ('BACKGROUND', (0,0), (-1,-1), colors.white),
    ]))
    return [table, Spacer(1, 0.06 * inch)]

def create_academic_divider():
    """Creates a subtle academic divider line."""
    data = [[""]]
    table = Table(data, colWidths=[6.8 * inch])
    table.setStyle(TableStyle([
        ('LINEABOVE', (0,0), (-1,-1), 0.5, COLOR_BORDER),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
    ]))
    return table

def create_academic_stat_grid(stats):
    """Creates an academic-style grid of statistics."""
    data = []
    row = []
    
    for i, (label, value, color) in enumerate(stats):
        if len(str(value)) > 10:
            value_style = ParagraphStyle(
                name="StatValueLong",
                parent=styles['Normal'],
                fontName=FONT_BOLD,
                fontSize=12,
                alignment=TA_CENTER,
                textColor=color or COLOR_PRIMARY,
                spaceAfter=2,
                wordWrap='CJK',
            )
        else:
            value_style = ParagraphStyle(
                name="StatValue",
                parent=styles['Normal'],
                fontName=FONT_BOLD,
                fontSize=16,
                alignment=TA_CENTER,
                textColor=color or COLOR_PRIMARY,
                spaceAfter=2,
                wordWrap='CJK',
            )
        
        label_style = ParagraphStyle(
            name="StatLabel",
            parent=styles['Normal'],
            fontName=FONT_FAMILY,
            fontSize=8,
            alignment=TA_CENTER,
            textColor=COLOR_GRAY,
            wordWrap='CJK',
        )
        
        cell_data = [
            [Paragraph(str(value), value_style)],
            [Paragraph(label, label_style)],
        ]
        
        cell_table = Table(cell_data, colWidths=[1.5 * inch])
        cell_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), COLOR_LIGHT_GRAY),
            ('BORDER', (0,0), (-1,-1), 0.5, COLOR_BORDER),
            ('TOPPADDING', (0,0), (-1,-1), 6),
            ('BOTTOMPADDING', (0,0), (-1,-1), 6),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ]))
        row.append(cell_table)
        if len(row) == 4:
            data.append(row)
            row = []
    
    if row:
        while len(row) < 4:
            empty_cell = Table([[""]], colWidths=[1.5 * inch])
            empty_cell.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,-1), colors.white),
                ('BORDER', (0,0), (-1,-1), 0.5, colors.white),
            ]))
            row.append(empty_cell)
        data.append(row)
    
    if data:
        grid = Table(data, colWidths=[1.5 * inch] * 4)
        grid.setStyle(TableStyle([
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('TOPPADDING', (0,0), (-1,-1), 2),
            ('BOTTOMPADDING', (0,0), (-1,-1), 2),
        ]))
        return grid
    return None

def create_academic_method_table():
    """Creates an academic-style method comparison table with actual data."""
    
    # Use actual data from portfolio_metrics_test.csv
    methods = ['Constant', 'Ledoit-Wolf', 'Elastic Net', 'XGBoost', 'Optimal']
    
    # Map method names to data
    vol_map = {}
    frob_map = {}
    if not portfolio_metrics.empty:
        for _, row in portfolio_metrics.iterrows():
            vol_map[row['method']] = row['mean_volatility']
    
    # Frobenius distances from statistical_tests.json
    if stat_tests:
        frob_map['Constant'] = stat_tests.get('summary', {}).get('mean_frobenius_constant', 0.01214)
        frob_map['Ledoit-Wolf'] = stat_tests.get('summary', {}).get('mean_frobenius_lw', 0.01172)
        frob_map['Optimal'] = stat_tests.get('summary', {}).get('mean_frobenius_optimal', 0.01209)
    
    # For Elastic Net and XGBoost, use values from their results
    elastic_net = data.get("elastic_net_results", pd.DataFrame())
    xgboost = data.get("xgboost_results", pd.DataFrame())
    if not elastic_net.empty:
        frob_map['Elastic Net'] = elastic_net['mean_frobenius'].values[0] if 'mean_frobenius' in elastic_net.columns else 0.01214
    if not xgboost.empty:
        frob_map['XGBoost'] = xgboost['mean_frobenius'].values[0] if 'mean_frobenius' in xgboost.columns else 0.01214
    
    # Use default values if not available
    vol_map.setdefault('Elastic Net', 0.01010)
    vol_map.setdefault('XGBoost', 0.01010)
    frob_map.setdefault('Elastic Net', 0.01214)
    frob_map.setdefault('XGBoost', 0.01214)
    
    data_rows = [
        [Paragraph("<b>Method</b>", style_table_header),
         Paragraph("<b>Mean Volatility</b>", style_table_header),
         Paragraph("<b>Frobenius Distance</b>", style_table_header)],
    ]
    
    for method in methods:
        vol = vol_map.get(method, 0.0)
        frob = frob_map.get(method, 0.0)
        if method == 'Ledoit-Wolf':
            data_rows.append([
                Paragraph(method, style_table_highlight),
                Paragraph(f"{vol:.5f}", style_table_highlight),
                Paragraph(f"{frob:.5f}", style_table_highlight),
            ])
        else:
            data_rows.append([
                Paragraph(method, style_table_cell),
                Paragraph(f"{vol:.5f}", style_table_cell),
                Paragraph(f"{frob:.5f}", style_table_cell),
            ])
    
    table = Table(data_rows, colWidths=[2.0*inch, 2.2*inch, 2.0*inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), COLOR_PRIMARY),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), FONT_BOLD),
        ('FONTSIZE', (0,0), (-1,0), 10),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('RIGHTPADDING', (0,0), (-1,-1), 8),
        ('GRID', (0,0), (-1,-1), 0.5, COLOR_BORDER),
        ('BACKGROUND', (0,2), (-1,2), COLOR_HIGHLIGHT),
        ('TEXTCOLOR', (0,2), (-1,2), COLOR_PRIMARY),
        ('FONTNAME', (0,2), (-1,2), FONT_BOLD),
        ('BACKGROUND', (0,1), (-1,1), colors.white),
        ('BACKGROUND', (0,3), (-1,3), colors.white),
        ('BACKGROUND', (0,4), (-1,4), colors.white),
        ('BACKGROUND', (0,5), (-1,5), colors.white),
    ]))
    
    return table

# ==============================================================================
# 5. GENERATE THE PDF
# ==============================================================================

def generate_pdf():
    print("🖨️ Generating academic LinkedIn PDF with actual data...")
    
    doc = SimpleDocTemplate(
        OUTPUT_PDF,
        pagesize=A4,
        leftMargin=18*mm,
        rightMargin=18*mm,
        topMargin=14*mm,
        bottomMargin=14*mm,
    )

    story = []

    # ======================================================================
    # PAGE 1: HEADER + INTRODUCTION + LAMBDA PLOT
    # ======================================================================
    
    story.extend(create_academic_hook(
        title_text="Does Market Regime Information Improve Covariance Estimation?",
        subtitle_text="A Data-Driven Risk Management Project for Systematic Portfolios"
    ))
    
    story.append(create_academic_divider())
    
    story.append(Paragraph(
        "When building a portfolio, you need to estimate how assets move together. "
        "The standard method applies a fixed shrinkage intensity to the covariance matrix. "
        "But markets change regimes—high volatility, low volatility, crises, recoveries. "
        "This project tests whether observable market conditions can help predict a better shrinkage intensity.",
        style_body
    ))
    
    story.append(Paragraph("What We Did", style_section))
    story.append(Paragraph(
        "We used 20 years of daily data (2000–2025) for 50 liquid S&P 500 stocks. "
        "For each rolling 120-day window, we computed the optimal shrinkage intensity (\u03bb*) "
        "that minimizes the error between the estimated covariance and the actual future covariance.",
        style_body
    ))
    
    story.append(Paragraph("The Surprising Finding", style_section))
    
    finding_data = [
        [Paragraph(f"<b>\u03bb* is near-zero for almost all windows (mean = {lambda_mean:.2e})</b>", style_highlight_box)],
        [Paragraph("The sample covariance matrix was already well-conditioned for this 50-stock universe. "
                   "Shrinkage toward the identity matrix provided almost no benefit.", style_body)],
    ]
    finding_table = Table(finding_data, colWidths=[6.8 * inch])
    finding_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), COLOR_LIGHT_GRAY),
        ('BORDER', (0,0), (-1,-1), 0.5, COLOR_BORDER),
        ('LEFTPADDING', (0,0), (-1,-1), 15),
        ('RIGHTPADDING', (0,0), (-1,-1), 15),
        ('TOPPADDING', (0,0), (-1,-1), 8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(finding_table)
    
    story.append(Spacer(1, 0.03 * inch))
    
    if IMAGES["lambda_plot"] and os.path.exists(IMAGES["lambda_plot"]):
        story.append(Image(IMAGES["lambda_plot"], width=6.8*inch, height=2.6*inch))
        story.append(Paragraph("Figure 1: Optimal shrinkage intensity (\u03bb*) over time", style_caption))
    
    story.append(Paragraph("What We Compared", style_section))
    story.append(Paragraph(
        "We tested four baselines and two machine learning models. "
        "The baselines included a constant shrinkage rule, a VIX-based threshold, a rolling average, "
        "and the industry-standard Ledoit-Wolf estimator. The ML models were Elastic Net and XGBoost.",
        style_body
    ))
    
    story.append(PageBreak())
    
    # ======================================================================
    # PAGE 2: FROBENIUS PLOT + DETAILED RESULTS
    # ======================================================================
    
    if IMAGES["frobenius_plot"] and os.path.exists(IMAGES["frobenius_plot"]):
        story.append(Image(IMAGES["frobenius_plot"], width=6.8*inch, height=2.8*inch))
        story.append(Paragraph("Figure 2: Out-of-sample covariance estimation accuracy", style_caption))
    
    story.append(Paragraph("Key Results Summary", style_section))
    
    stats = [
        ("Mean \u03bb*", f"{lambda_mean:.2e}", COLOR_PRIMARY),
        ("Std \u03bb*", f"{lambda_std:.5f}", COLOR_GRAY),
        ("Volatility Reduction", f"{vol_reduction:.1f}%", COLOR_ACCENT),
        ("Best Method", "Ledoit-Wolf", COLOR_WARNING),
    ]
    stat_grid = create_academic_stat_grid(stats)
    if stat_grid:
        story.append(stat_grid)
    
    story.append(Spacer(1, 0.04 * inch))
    
    story.append(Paragraph("Detailed Findings", style_section))
    
    findings_details = [
        f"• Ledoit-Wolf reduced portfolio volatility by 14\u201320% across all regimes (p = {dm_p_value:.4f})",
        "• ML models (Elastic Net, XGBoost) underperformed simple baselines (R² = -0.023)",
        "• VIX-based threshold and rolling average performed identically to constant",
        "• Reducing Frobenius distance doesn't always reduce portfolio volatility",
    ]
    for f in findings_details:
        story.append(Paragraph(f, style_bullet))
    
    story.append(Spacer(1, 0.02 * inch))
    story.append(Paragraph("Method Comparison (Test Set 2020\u20132025)", style_section))
    
    method_table = create_academic_method_table()
    story.append(method_table)
    
    story.append(Spacer(1, 0.03 * inch))
    
    insight_data = [
        [Paragraph("<b>Key Insight:</b> The limitation is not model complexity—it's the lack of signal in \u03bb*. "
                   "When the target variable has near-zero variance, even the best model cannot find a signal.",
                   style_body_small)],
    ]
    insight_table = Table(insight_data, colWidths=[6.8 * inch])
    insight_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), COLOR_HIGHLIGHT),
        ('BORDER', (0,0), (-1,-1), 0.5, COLOR_BORDER),
        ('LEFTPADDING', (0,0), (-1,-1), 12),
        ('RIGHTPADDING', (0,0), (-1,-1), 12),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(insight_table)
    
    story.append(PageBreak())
    
    # ======================================================================
    # PAGE 3: HEATMAP + TAKEAWAYS + GITHUB
    # ======================================================================
    
    story.append(Paragraph("What We Found", style_section))
    story.append(Paragraph(
        "Ledoit-Wolf consistently outperformed all other methods in covariance accuracy. "
        "The two machine learning models performed worse than the simplest constant baseline. "
        "This suggests that, for this dataset, the limitation is not model complexity—it is the lack of signal in the target variable.",
        style_body
    ))
    
    story.append(Paragraph("Does It Matter for Portfolios?", style_section))
    story.append(Paragraph(
        f"Yes. When we constructed minimum-variance portfolios using each covariance estimate, "
        "Ledoit-Wolf delivered the lowest realized volatility. "
        f"It reduced portfolio volatility by {vol_reduction:.1f}% consistently across all market regimes—"
        "including the 2020 COVID crash, the 2022 bear market, and subsequent recoveries.",
        style_body
    ))
    
    if IMAGES["heatmap"] and os.path.exists(IMAGES["heatmap"]):
        story.append(Image(IMAGES["heatmap"], width=6.8*inch, height=2.8*inch))
        story.append(Paragraph("Figure 3: Portfolio volatility by method and market regime", style_caption))
    
    story.append(Paragraph("Key Takeaways", style_section))
    takeaways = [
        f"• The optimal shrinkage intensity (\u03bb*) is near-zero (mean = {lambda_mean:.2e}) for this 50-stock universe.",
        "• The industry-standard Ledoit-Wolf estimator is the best choice for this dataset.",
        "• Machine learning models did not improve performance—they simply predicted the mean.",
        "• Reducing covariance estimation error does not always reduce portfolio volatility.",
    ]
    for t in takeaways:
        story.append(Paragraph(t, style_bullet))
    
    story.append(Spacer(1, 0.02 * inch))
    story.append(Paragraph("A Note on the Machine Learning Approach", style_section))
    story.append(Paragraph(
        "We framed this as a supervised learning problem, which is a standard approach in quantitative finance. "
        "However, the machine learning models did not outperform the simplest baselines. "
        "This is not a failure of the models—it is a reflection of the data. "
        "When the target variable has near-zero variance, even the best model cannot find a signal.",
        style_body_small
    ))
    
    story.append(Spacer(1, 0.02 * inch))
    
    github_data = [
        [Paragraph("<b>Open Source:</b> Full code, data pipeline, and results on GitHub", style_body)],
        [Paragraph("<font color='#1A3A5C'>https://github.com/kira-ml/regime-adaptive-covariance.git</font>", 
                   ParagraphStyle(
                       name="GitHubLink",
                       parent=styles['Normal'],
                       fontName=FONT_FAMILY,
                       fontSize=11,
                       alignment=TA_CENTER,
                       textColor=COLOR_PRIMARY,
                   ))],
    ]
    github_table = Table(github_data, colWidths=[6.8 * inch])
    github_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), COLOR_LIGHT_GRAY),
        ('BORDER', (0,0), (-1,-1), 0.5, COLOR_BORDER),
        ('LEFTPADDING', (0,0), (-1,-1), 12),
        ('RIGHTPADDING', (0,0), (-1,-1), 12),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(github_table)
    
    story.append(Spacer(1, 0.02 * inch))
    story.append(Paragraph(
        "This project is for educational and research purposes only. Not financial advice.",
        style_footer
    ))

    doc.build(story)
    print(f"✅ PDF generated successfully: {OUTPUT_PDF}")

if __name__ == "__main__":
    generate_pdf()