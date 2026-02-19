"""
NautiCAI - Professional PDF Inspection Report
Internship-quality layout with strong visual hierarchy.
"""

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                 TableStyle, HRFlowable, PageBreak, KeepTogether)
from reportlab.platypus import Image as RLImage
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from PIL import Image as PILImage
import io, datetime

# ── Palette ──────────────────────────────────────────────────────
DARK_NAVY   = colors.HexColor('#0A1628')
NAVY_MID    = colors.HexColor('#1A3355')
TEAL        = colors.HexColor('#00B4AA')
RED         = colors.HexColor('#D62839')
RED_BG      = colors.HexColor('#FDF0F1')
AMBER       = colors.HexColor('#E07B39')
AMBER_BG    = colors.HexColor('#FEF6EE')
GREEN       = colors.HexColor('#1A8C6E')
GREEN_BG    = colors.HexColor('#EEF8F5')
GREY_BG     = colors.HexColor('#F5F7FA')
GREY_BORDER = colors.HexColor('#DDE3EC')
GREY_TEXT   = colors.HexColor('#5A6478')
WHITE       = colors.white

SEVERITY_MAP = {
    'corrosion':     ('CRITICAL', RED,   RED_BG,   '#D62839'),
    'damage':        ('CRITICAL', RED,   RED_BG,   '#D62839'),
    'free_span':     ('CRITICAL', RED,   RED_BG,   '#D62839'),
    'debris':        ('WARNING',  AMBER, AMBER_BG, '#E07B39'),
    'marine_growth': ('WARNING',  AMBER, AMBER_BG, '#E07B39'),
    'healthy':       ('NORMAL',   GREEN, GREEN_BG, '#1A8C6E'),
    'anode':         ('NORMAL',   GREEN, GREEN_BG, '#1A8C6E'),
}

PAGE_W = A4[0] - 3*cm


# ── Image helper ─────────────────────────────────────────────────
def get_rl_image(item, max_width=PAGE_W - 1*cm, max_height=9*cm):
    try:
        if item.get('frame_bytes'):
            pil_img = PILImage.open(io.BytesIO(item['frame_bytes'])).convert('RGB')
        elif item.get('frame') is not None:
            pil_img = PILImage.fromarray(item['frame'][:, :, :3])
        else:
            return None
        buf = io.BytesIO()
        pil_img.save(buf, format='JPEG', quality=92)
        buf.seek(0)
        w, h  = pil_img.size
        ratio = min(max_width / w, max_height / h)
        return RLImage(buf, width=w * ratio, height=h * ratio)
    except Exception:
        return None


# ── Styles ───────────────────────────────────────────────────────
def make_styles():
    return {
        'hero':    ParagraphStyle('hero',    fontSize=32, textColor=WHITE,
                       alignment=TA_CENTER, fontName='Helvetica-Bold'),
        'hero_sub':ParagraphStyle('hero_sub',fontSize=11, textColor=TEAL,
                       alignment=TA_CENTER, fontName='Helvetica'),
        'section': ParagraphStyle('section', fontSize=12, textColor=DARK_NAVY,
                       fontName='Helvetica-Bold', spaceBefore=10, spaceAfter=5),
        'body':    ParagraphStyle('body',    fontSize=9,  fontName='Helvetica', leading=14),
        'bold':    ParagraphStyle('bold',    fontSize=9,  fontName='Helvetica-Bold', leading=14),
        'caption': ParagraphStyle('caption', fontSize=8,  textColor=GREY_TEXT,
                       fontName='Helvetica-Bold', alignment=TA_CENTER),
        'footer':  ParagraphStyle('footer',  fontSize=8,  textColor=GREY_TEXT,
                       fontName='Helvetica', alignment=TA_CENTER),
    }


def generate_report(
    anomaly_log,
    mission_name  = "Subsea Inspection Mission",
    operator_name = "NautiCAI Operator",
    vessel_id     = "ROV-NautiCAI-01",
    location      = "Offshore Location",
    output_path   = "nauticai_report.pdf"
):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
        leftMargin=1.5*cm, rightMargin=1.5*cm,
        topMargin=1.5*cm,  bottomMargin=1.5*cm)
    ST    = make_styles()
    story = []
    now   = datetime.datetime.now()

    # ── 1. HERO BANNER ───────────────────────────────────────────
    banner = Table([
        [Paragraph('NautiCAI', ST['hero'])],
        [Paragraph('Underwater Hazard Detection Report — Explore Safer Seas Now', ST['hero_sub'])],
        [Paragraph(f'{now.strftime("%d %B %Y")}  ·  {mission_name}  ·  {location}',
            ParagraphStyle('meta', fontSize=8, textColor=colors.HexColor('#6AAFAF'),
                alignment=TA_CENTER, fontName='Helvetica'))],
    ], colWidths=[PAGE_W])
    banner.setStyle(TableStyle([
        ('BACKGROUND',    (0,0),(-1,-1), DARK_NAVY),
        ('TOPPADDING',    (0,0),(-1,0),  26),
        ('BOTTOMPADDING', (0,0),(-1,0),  4),
        ('TOPPADDING',    (0,1),(-1,1),  2),
        ('BOTTOMPADDING', (0,1),(-1,1),  6),
        ('TOPPADDING',    (0,2),(-1,2),  2),
        ('BOTTOMPADDING', (0,2),(-1,2),  20),
        ('ALIGN',         (0,0),(-1,-1), 'CENTER'),
        ('LINEBELOW',     (0,-1),(-1,-1), 4, TEAL),
    ]))
    story.append(banner)
    story.append(Spacer(1, 0.5*cm))

    # ── 2. MISSION DETAILS ───────────────────────────────────────
    story.append(Paragraph('Mission Details', ST['section']))
    story.append(HRFlowable(width='100%', thickness=2, color=TEAL, spaceAfter=8))

    def kv(k, v):
        return [Paragraph(f'<b>{k}</b>', ST['body']), Paragraph(str(v), ST['body'])]

    md = Table([
        kv('Mission Name', mission_name)  + kv('Date',      now.strftime('%Y-%m-%d')),
        kv('Operator',     operator_name) + kv('Time',      now.strftime('%H:%M:%S')),
        kv('Vessel / ROV', vessel_id)     + kv('Location',  location),
        kv('AI Model',     'YOLOv8s')     + kv('Framework', 'Ultralytics + Streamlit'),
    ], colWidths=[3*cm, 6*cm, 3*cm, 6*cm])
    md.setStyle(TableStyle([
        ('BACKGROUND',    (0,0),(0,-1), GREY_BG),
        ('BACKGROUND',    (2,0),(2,-1), GREY_BG),
        ('GRID',          (0,0),(-1,-1), 0.5, GREY_BORDER),
        ('TOPPADDING',    (0,0),(-1,-1), 8),
        ('BOTTOMPADDING', (0,0),(-1,-1), 8),
        ('LEFTPADDING',   (0,0),(-1,-1), 10),
        ('FONTSIZE',      (0,0),(-1,-1), 9),
        ('ROWBACKGROUNDS',(0,0),(-1,-1), [WHITE, GREY_BG, WHITE, GREY_BG]),
    ]))
    story.append(md)
    story.append(Spacer(1, 0.5*cm))

    # ── 3. EXECUTIVE SUMMARY ─────────────────────────────────────
    story.append(Paragraph('Executive Summary', ST['section']))
    story.append(HRFlowable(width='100%', thickness=2, color=TEAL, spaceAfter=8))

    class_counts = {}
    crit = warn = norm = 0
    for item in anomaly_log:
        cls = item.get('class_name', 'unknown')
        class_counts[cls] = class_counts.get(cls, 0) + 1
        sev = SEVERITY_MAP.get(cls, ('WARNING',))[0]
        if sev == 'CRITICAL':  crit += 1
        elif sev == 'WARNING': warn += 1
        else:                  norm += 1

    def big(n, col):
        return Paragraph(str(n), ParagraphStyle(f'bv{n}',
            fontSize=32, fontName='Helvetica-Bold', textColor=col, alignment=TA_CENTER))
    def lbl(t, col=GREY_TEXT):
        return Paragraph(t, ParagraphStyle(f'lbl{t}',
            fontSize=8, fontName='Helvetica-Bold', textColor=col, alignment=TA_CENTER))

    sum_t = Table([
        [lbl('TOTAL DETECTIONS'), lbl('CRITICAL', RED), lbl('WARNINGS', AMBER), lbl('NORMAL', GREEN)],
        [big(len(anomaly_log), DARK_NAVY), big(crit, RED), big(warn, AMBER), big(norm, GREEN)],
    ], colWidths=[PAGE_W/4]*4)
    sum_t.setStyle(TableStyle([
        ('BACKGROUND',    (0,0),(-1,-1), WHITE),
        ('BACKGROUND',    (1,0),(1,-1),  RED_BG),
        ('BACKGROUND',    (2,0),(2,-1),  AMBER_BG),
        ('BACKGROUND',    (3,0),(3,-1),  GREEN_BG),
        ('GRID',          (0,0),(-1,-1), 0.5, GREY_BORDER),
        ('ALIGN',         (0,0),(-1,-1), 'CENTER'),
        ('VALIGN',        (0,0),(-1,-1), 'MIDDLE'),
        ('TOPPADDING',    (0,0),(-1,-1), 12),
        ('BOTTOMPADDING', (0,0),(-1,-1), 14),
        ('LINEABOVE',     (0,0),(-1,0),  3, TEAL),
        ('LINEBELOW',     (0,-1),(-1,-1), 2, GREY_BORDER),
        ('LINEBEFORE',    (1,0),(1,-1),  2, RED),
        ('LINEBEFORE',    (2,0),(2,-1),  2, AMBER),
        ('LINEBEFORE',    (3,0),(3,-1),  2, GREEN),
    ]))
    story.append(sum_t)
    story.append(Spacer(1, 0.5*cm))

    # ── 4. CLASS BREAKDOWN ───────────────────────────────────────
    if class_counts:
        story.append(Paragraph('Detection Breakdown by Class', ST['section']))
        story.append(HRFlowable(width='100%', thickness=1, color=GREY_BORDER, spaceAfter=6))

        total_det = max(len(anomaly_log), 1)
        hdr = [Paragraph(f'<b>{h}</b>', ST['bold'])
               for h in ['Anomaly Class', 'Count', 'Severity', 'Share %']]
        rows = [hdr]
        for cls, cnt in sorted(class_counts.items(), key=lambda x: -x[1]):
            sev_label, _, _, hex_col = SEVERITY_MAP.get(cls, ('WARNING', None, None, '#E07B39'))
            rows.append([
                Paragraph(cls.replace('_', ' ').title(), ST['body']),
                Paragraph(str(cnt), ST['body']),
                Paragraph(f'<font color="{hex_col}"><b>{sev_label}</b></font>', ST['body']),
                Paragraph(f'{cnt/total_det*100:.1f}%', ST['body']),
            ])

        bd_t = Table(rows, colWidths=[7*cm, 2.5*cm, 4*cm, 4.5*cm])
        bd_style = [
            ('BACKGROUND',    (0,0),(-1,0), NAVY_MID),
            ('TEXTCOLOR',     (0,0),(-1,0), WHITE),
            ('FONTSIZE',      (0,0),(-1,-1), 9),
            ('GRID',          (0,0),(-1,-1), 0.5, GREY_BORDER),
            ('ALIGN',         (1,0),(3,-1),  'CENTER'),
            ('TOPPADDING',    (0,0),(-1,-1), 8),
            ('BOTTOMPADDING', (0,0),(-1,-1), 8),
            ('LEFTPADDING',   (0,0),(-1,-1), 10),
        ]
        for i in range(1, len(rows)):
            bd_style.append(('BACKGROUND', (0,i),(-1,i), WHITE if i % 2 else GREY_BG))
        bd_t.setStyle(TableStyle(bd_style))
        story.append(bd_t)

    # ── 5. DETAILED LOG ──────────────────────────────────────────
    if anomaly_log:
        story.append(PageBreak())
        story.append(Paragraph('Detailed Anomaly Log', ST['section']))
        story.append(HRFlowable(width='100%', thickness=2, color=TEAL, spaceAfter=10))

        for i, item in enumerate(anomaly_log):
            cls       = item.get('class_name', 'unknown')
            conf      = item.get('confidence', 0.0)
            timestamp = item.get('timestamp', 'N/A')
            sev_label, sev_color, sev_bg, hex_col = SEVERITY_MAP.get(
                cls, ('WARNING', AMBER, AMBER_BG, '#E07B39'))

            # Detection header bar
            hdr_t = Table([[
                Paragraph(
                    f'<font color="white"><b>Detection #{i+1:02d} &nbsp;—&nbsp; '
                    f'{cls.replace("_"," ").title()}</b></font>', ST['body']),
                Paragraph(f'<font color="white"><b>{sev_label}</b></font>',
                    ParagraphStyle(f'sr{i}', fontSize=10, fontName='Helvetica-Bold',
                        alignment=TA_RIGHT)),
            ]], colWidths=[PAGE_W * 0.78, PAGE_W * 0.22])
            hdr_t.setStyle(TableStyle([
                ('BACKGROUND',    (0,0),(-1,-1), sev_color),
                ('TOPPADDING',    (0,0),(-1,-1), 10),
                ('BOTTOMPADDING', (0,0),(-1,-1), 10),
                ('LEFTPADDING',   (0,0),(0,0),   14),
                ('RIGHTPADDING',  (1,0),(1,0),   14),
            ]))

            # Metadata
            meta_t = Table([
                [Paragraph('<b>Timestamp</b>',  ST['body']),
                 Paragraph(timestamp,            ST['body']),
                 Paragraph('<b>Class</b>',       ST['body']),
                 Paragraph(cls.replace('_',' ').title(), ST['body'])],
                [Paragraph('<b>Confidence</b>', ST['body']),
                 Paragraph(f'{conf:.1%}',        ST['body']),
                 Paragraph('<b>Severity</b>',   ST['body']),
                 Paragraph(f'<font color="{hex_col}"><b>{sev_label}</b></font>', ST['body'])],
            ], colWidths=[3*cm, 6*cm, 3*cm, 6*cm])
            meta_t.setStyle(TableStyle([
                ('BACKGROUND',    (0,0),(0,-1), GREY_BG),
                ('BACKGROUND',    (2,0),(2,-1), GREY_BG),
                ('GRID',          (0,0),(-1,-1), 0.5, GREY_BORDER),
                ('TOPPADDING',    (0,0),(-1,-1), 9),
                ('BOTTOMPADDING', (0,0),(-1,-1), 9),
                ('LEFTPADDING',   (0,0),(-1,-1), 10),
                ('FONTSIZE',      (0,0),(-1,-1), 9),
                ('ROWBACKGROUNDS',(0,0),(-1,-1), [WHITE, GREY_BG]),
            ]))

            # Image
            rl_img   = get_rl_image(item)
            elements = [hdr_t, meta_t]

            if rl_img:
                img_t = Table([
                    [rl_img],
                    [Paragraph(
                        f'AI-Annotated Frame  ·  {cls.replace("_"," ").title()}'
                        f'  ·  Confidence {conf:.1%}  ·  Detected at {timestamp}',
                        ST['caption'])],
                ], colWidths=[PAGE_W])
                img_t.setStyle(TableStyle([
                    ('ALIGN',         (0,0),(-1,-1), 'CENTER'),
                    ('VALIGN',        (0,0),(0,0),   'MIDDLE'),
                    ('BACKGROUND',    (0,0),(-1,-1), sev_bg),
                    ('TOPPADDING',    (0,0),(0,0),   12),
                    ('BOTTOMPADDING', (0,0),(0,0),   8),
                    ('TOPPADDING',    (0,1),(0,1),   4),
                    ('BOTTOMPADDING', (0,1),(0,1),   10),
                    ('LINEABOVE',     (0,0),(-1,0),  0.5, GREY_BORDER),
                    ('LINEBELOW',     (0,-1),(-1,-1), 0.5, GREY_BORDER),
                    ('LINEBEFORE',    (0,0),(0,-1),  0.5, GREY_BORDER),
                    ('LINEAFTER',     (0,0),(0,-1),  0.5, GREY_BORDER),
                    ('LINEABOVE',     (0,0),(-1,0),  3, sev_color),
                ]))
                elements.append(img_t)

            elements.append(Spacer(1, 0.5*cm))
            story.append(KeepTogether(elements))

    # ── 6. FOOTER ────────────────────────────────────────────────
    story.append(Spacer(1, 0.4*cm))
    story.append(HRFlowable(width='100%', thickness=1, color=GREY_BORDER))
    story.append(Spacer(1, 0.2*cm))
    story.append(Paragraph(
        f'Generated by NautiCAI &nbsp;|&nbsp; {now.strftime("%Y-%m-%d %H:%M:%S")}'
        f'&nbsp;|&nbsp; Confidential Inspection Report &nbsp;|&nbsp; www.nauticai-ai.com',
        ST['footer']))

    doc.build(story)
    result = buf.getvalue()
    buf.close()
    return result