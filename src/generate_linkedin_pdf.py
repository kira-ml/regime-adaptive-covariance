import os
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm, inch
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak,
    KeepTogether
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY, TA_RIGHT
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

# ==============================================================================
# 2. ACADEMIC STYLES - FIXED FONT HANDLING
# ==============================================================================

# Try to register Times New Roman fonts from system
def register_times_fonts():
    """Register Times New Roman fonts with fallback to built-in fonts."""
    try:
        # Try common Windows font locations
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
        
        # Try bold
        bold_paths = [
            "C:/Windows/Fonts/timesbd.ttf",
            "C:/Windows/Fonts/times new roman bold.ttf",
            "C:/Windows/Fonts/ttimesbd.ttf",
            "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman_Bold.ttf",
            "/Library/Fonts/Times New Roman Bold.ttf",
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
        
        # Try italic
        italic_paths = [
            "C:/Windows/Fonts/timesi.ttf",
            "C:/Windows/Fonts/times new roman italic.ttf",
            "C:/Windows/Fonts/ttimesi.ttf",
            "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman_Italic.ttf",
            "/Library/Fonts/Times New Roman Italic.ttf",
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
        
        # Try bold italic
        bi_paths = [
            "C:/Windows/Fonts/timesbi.ttf",
            "C:/Windows/Fonts/times new roman bold italic.ttf",
            "C:/Windows/Fonts/ttimesbi.ttf",
            "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman_Bold_Italic.ttf",
            "/Library/Fonts/Times New Roman Bold Italic.ttf",
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

# Register fonts
font_registered = register_times_fonts()

# Define font family names
FONT_FAMILY = "Times-Roman"
FONT_BOLD = "Times-Bold"
FONT_ITALIC = "Times-Italic"
FONT_BOLD_ITALIC = "Times-BoldItalic"

styles = getSampleStyleSheet()

# --- Academic Color Palette ---
COLOR_PRIMARY = colors.HexColor("#1A3A5C")      # Dark Navy Blue (academic)
COLOR_DARK = colors.HexColor("#1A1A2E")         # Almost Black
COLOR_GRAY = colors.HexColor("#4A4A4A")         # Dark Gray
COLOR_LIGHT_GRAY = colors.HexColor("#F5F6F8")   # Light Gray Background
COLOR_ACCENT = colors.HexColor("#2E7D32")       # Forest Green
COLOR_BORDER = colors.HexColor("#C0C8D0")       # Muted Gray Border
COLOR_WARNING = colors.HexColor("#B76E2E")      # Academic Orange/Brown
COLOR_HIGHLIGHT = colors.HexColor("#E8EDF3")    # Very Light Blue for highlights

# --- Academic Title Style ---
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

style_bullet_em = ParagraphStyle(
    name="BulletEm",
    parent=styles['Normal'],
    fontName=FONT_BOLD,
    fontSize=11,
    leading=16,
    alignment=TA_LEFT,
    textColor=COLOR_PRIMARY,
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

style_stat = ParagraphStyle(
    name="Stat",
    parent=styles['Normal'],
    fontName=FONT_BOLD,
    fontSize=16,  # Reduced from 18 to fit better
    alignment=TA_CENTER,
    textColor=COLOR_PRIMARY,
    spaceAfter=2,
)

style_stat_label = ParagraphStyle(
    name="StatLabel",
    parent=styles['Normal'],
    fontName=FONT_FAMILY,
    fontSize=8,  # Reduced from 9 to fit better
    alignment=TA_CENTER,
    textColor=COLOR_GRAY,
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
# 3. ACADEMIC HOOK WITH LEFT BORDER
# ==============================================================================
def create_academic_hook(title_text, subtitle_text):
    """
    Creates an academic-style header with left border accent.
    """
    
    title_para = Paragraph(title_text, style_title)
    subtitle_para = Paragraph(subtitle_text, style_subtitle)
    
    data = [
        [title_para],
        [subtitle_para],
    ]
    
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

# ==============================================================================
# 4. HELPER: ACADEMIC DIVIDER
# ==============================================================================
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

# ==============================================================================
# 5. HELPER: ACADEMIC STATISTICS GRID - FIXED FOR BETTER TEXT FIT
# ==============================================================================
def create_academic_stat_grid(stats):
    """
    Creates an academic-style grid of statistics.
    stats: list of (label, value, color) tuples
    """
    data = []
    row = []
    
    # Define a style for the value that prevents splitting
    stat_value_style = ParagraphStyle(
        name="StatValue",
        parent=styles['Normal'],
        fontName=FONT_BOLD,
        fontSize=16,
        alignment=TA_CENTER,
        textColor=COLOR_PRIMARY,
        spaceAfter=2,
        wordWrap='CJK',  # Better word wrapping
    )
    
    stat_label_style = ParagraphStyle(
        name="StatLabel",
        parent=styles['Normal'],
        fontName=FONT_FAMILY,
        fontSize=8,
        alignment=TA_CENTER,
        textColor=COLOR_GRAY,
        wordWrap='CJK',  # Better word wrapping
    )
    
    for i, (label, value, color) in enumerate(stats):
        # For the value, if it's a long text like "Ledoit-Wolf", use a smaller font or split
        if len(str(value)) > 10:
            # Use smaller font for long text
            value_style = ParagraphStyle(
                name="StatValueLong",
                parent=styles['Normal'],
                fontName=FONT_BOLD,
                fontSize=12,  # Smaller for long text
                alignment=TA_CENTER,
                textColor=color or COLOR_PRIMARY,
                spaceAfter=2,
                wordWrap='CJK',
            )
        else:
            value_style = stat_value_style
            value_style.textColor = color or COLOR_PRIMARY
        
        cell_data = [
            [Paragraph(str(value), value_style)],
            [Paragraph(label, stat_label_style)],
        ]
        
        # Wider cells to prevent text breaking
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

# ==============================================================================
# 6. HELPER: ACADEMIC METHOD TABLE
# ==============================================================================
def create_academic_method_table():
    """Creates an academic-style method comparison table."""
    
    # Define table data with proper formatting
    data = [
        [Paragraph("<b>Method</b>", style_table_header),
         Paragraph("<b>Mean Volatility</b>", style_table_header),
         Paragraph("<b>Frobenius Distance</b>", style_table_header)],
        [Paragraph("Constant", style_table_cell),
         Paragraph("0.01010", style_table_cell),
         Paragraph("0.01214", style_table_cell)],
        [Paragraph("Ledoit-Wolf", style_table_highlight),
         Paragraph("0.00839", style_table_highlight),
         Paragraph("0.01172", style_table_highlight)],
        [Paragraph("Elastic Net", style_table_cell),
         Paragraph("0.01010", style_table_cell),
         Paragraph("0.01214", style_table_cell)],
        [Paragraph("XGBoost", style_table_cell),
         Paragraph("0.01010", style_table_cell),
         Paragraph("0.01214", style_table_cell)],
    ]
    
    table = Table(data, colWidths=[2.0*inch, 2.2*inch, 2.0*inch])
    table.setStyle(TableStyle([
        # Header style
        ('BACKGROUND', (0,0), (-1,0), COLOR_PRIMARY),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), FONT_BOLD),
        ('FONTSIZE', (0,0), (-1,0), 10),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        # Padding
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('RIGHTPADDING', (0,0), (-1,-1), 8),
        # Grid
        ('GRID', (0,0), (-1,-1), 0.5, COLOR_BORDER),
        # Highlight Ledoit-Wolf row
        ('BACKGROUND', (0,2), (-1,2), COLOR_HIGHLIGHT),
        ('TEXTCOLOR', (0,2), (-1,2), COLOR_PRIMARY),
        ('FONTNAME', (0,2), (-1,2), FONT_BOLD),
        # Alternating row colors
        ('BACKGROUND', (0,1), (-1,1), colors.white),
        ('BACKGROUND', (0,3), (-1,3), colors.white),
        ('BACKGROUND', (0,4), (-1,4), colors.white),
    ]))
    
    return table

# ==============================================================================
# 7. GENERATE THE PDF
# ==============================================================================
def generate_pdf():
    print("🖨️ Generating academic LinkedIn PDF with Times New Roman...")
    
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
    
    # --- Academic Hook ---
    story.extend(create_academic_hook(
        title_text="Does Market Regime Information Improve Covariance Estimation?",
        subtitle_text="A Data-Driven Risk Management Project for Systematic Portfolios"
    ))
    
    story.append(create_academic_divider())
    
    # --- Introduction Paragraph ---
    story.append(Paragraph(
        "When building a portfolio, you need to estimate how assets move together. "
        "The standard method applies a fixed shrinkage intensity to the covariance matrix. "
        "But markets change regimes—high volatility, low volatility, crises, recoveries. "
        "This project tests whether observable market conditions can help predict a better shrinkage intensity.",
        style_body
    ))
    
    # --- What We Did ---
    story.append(Paragraph("What We Did", style_section))
    story.append(Paragraph(
        "We used 20 years of daily data (2000–2025) for 50 liquid S&P 500 stocks. "
        "For each rolling 120-day window, we computed the optimal shrinkage intensity (\u03bb*) "
        "that minimizes the error between the estimated covariance and the actual future covariance.",
        style_body
    ))
    
    # --- The Surprising Finding with Highlight Box ---
    story.append(Paragraph("The Surprising Finding", style_section))
    
    finding_data = [
        [Paragraph("<b>\u03bb* is near-zero for almost all windows</b>", style_highlight_box)],
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
    
    # --- Lambda Plot ---
    if IMAGES["lambda_plot"] and os.path.exists(IMAGES["lambda_plot"]):
        story.append(Image(IMAGES["lambda_plot"], width=6.8*inch, height=2.6*inch))
        story.append(Paragraph("Figure 1: Optimal shrinkage intensity (\u03bb*) over time", style_caption))
    
    # --- What We Compared ---
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
    
    # --- Frobenius Plot ---
    if IMAGES["frobenius_plot"] and os.path.exists(IMAGES["frobenius_plot"]):
        story.append(Image(IMAGES["frobenius_plot"], width=6.8*inch, height=2.8*inch))
        story.append(Paragraph("Figure 2: Out-of-sample covariance estimation accuracy", style_caption))
    
    # --- Key Results Summary ---
    story.append(Paragraph("Key Results Summary", style_section))
    
    # Statistics grid - using shorter labels for better fit
    stats = [
        ("Mean \u03bb*", "3.4e-05", COLOR_PRIMARY),
        ("Std \u03bb*", "0.00035", COLOR_GRAY),
        ("Volatility Reduction", "17%", COLOR_ACCENT),
        ("Best Method", "Ledoit-Wolf", COLOR_WARNING),
    ]
    stat_grid = create_academic_stat_grid(stats)
    if stat_grid:
        story.append(stat_grid)
    
    story.append(Spacer(1, 0.04 * inch))
    
    # --- Detailed Findings ---
    story.append(Paragraph("Detailed Findings", style_section))
    
    findings_details = [
        "• Ledoit-Wolf reduced portfolio volatility by 14\u201320% across all regimes",
        "• ML models (Elastic Net, XGBoost) underperformed simple baselines",
        "• VIX-based threshold and rolling average performed identically to constant",
        "• Reducing Frobenius distance doesn't always reduce portfolio volatility",
    ]
    for f in findings_details:
        story.append(Paragraph(f, style_bullet))
    
    # --- Method Comparison Table ---
    story.append(Spacer(1, 0.02 * inch))
    story.append(Paragraph("Method Comparison (Test Set 2020\u20132025)", style_section))
    
    method_table = create_academic_method_table()
    story.append(method_table)
    
    story.append(Spacer(1, 0.03 * inch))
    
    # --- Key Insight Box ---
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
    
    # --- What We Found ---
    story.append(Paragraph("What We Found", style_section))
    story.append(Paragraph(
        "Ledoit-Wolf consistently outperformed all other methods in covariance accuracy. "
        "The two machine learning models performed worse than the simplest constant baseline. "
        "This suggests that, for this dataset, the limitation is not model complexity—it is the lack of signal in the target variable.",
        style_body
    ))
    
    # --- Does It Matter for Portfolios? ---
    story.append(Paragraph("Does It Matter for Portfolios?", style_section))
    story.append(Paragraph(
        "Yes. When we constructed minimum-variance portfolios using each covariance estimate, "
        "Ledoit-Wolf delivered the lowest realized volatility. "
        "It reduced portfolio volatility by 14% to 20% consistently across all market regimes—"
        "including the 2020 COVID crash, the 2022 bear market, and subsequent recoveries.",
        style_body
    ))
    
    # --- Heatmap ---
    if IMAGES["heatmap"] and os.path.exists(IMAGES["heatmap"]):
        story.append(Image(IMAGES["heatmap"], width=6.8*inch, height=2.8*inch))
        story.append(Paragraph("Figure 3: Portfolio volatility by method and market regime", style_caption))
    
    # --- Key Takeaways ---
    story.append(Paragraph("Key Takeaways", style_section))
    takeaways = [
        "• The optimal shrinkage intensity (\u03bb*) is near-zero for this 50-stock universe.",
        "• The industry-standard Ledoit-Wolf estimator is the best choice for this dataset.",
        "• Machine learning models did not improve performance—they simply predicted the mean.",
        "• Reducing covariance estimation error does not always reduce portfolio volatility.",
    ]
    for t in takeaways:
        story.append(Paragraph(t, style_bullet))
    
    # --- A Note on Machine Learning ---
    story.append(Spacer(1, 0.02 * inch))
    story.append(Paragraph("A Note on the Machine Learning Approach", style_section))
    story.append(Paragraph(
        "We framed this as a supervised learning problem, which is a standard approach in quantitative finance. "
        "However, the machine learning models did not outperform the simplest baselines. "
        "This is not a failure of the models—it is a reflection of the data. "
        "When the target variable has near-zero variance, even the best model cannot find a signal.",
        style_body_small
    ))
    
    # --- GitHub Link ---
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
    
    # --- Footer ---
    story.append(Spacer(1, 0.02 * inch))
    story.append(Paragraph(
        "This project is for educational and research purposes only. Not financial advice.",
        style_footer
    ))

    # Build PDF
    doc.build(story)
    print(f"✅ PDF generated successfully: {OUTPUT_PDF}")

if __name__ == "__main__":
    generate_pdf()