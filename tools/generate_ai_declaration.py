import os
import shutil
import docx
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import parse_xml, OxmlElement
from docx.oxml.ns import nsdecls, qn

def set_cell_margins(cell, top=100, bottom=100, left=140, right=140):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = OxmlElement('w:tcMar')
    for m, val in [('top', top), ('bottom', bottom), ('left', left), ('right', right)]:
        node = OxmlElement(f'w:{m}')
        node.set(qn('w:w'), str(val))
        node.set(qn('w:type'), 'dxa')
        tcMar.append(node)
    tcPr.append(tcMar)

def set_cell_shading(cell, color_hex):
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{color_hex}"/>')
    cell._tc.get_or_add_tcPr().append(shd)

def set_table_borders(table, color="CCCCCC", sz="4", val="single"):
    tblPr = table._tbl.tblPr
    borders = parse_xml(f'''
        <w:tblBorders {nsdecls("w")}>
            <w:top w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>
            <w:bottom w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>
            <w:left w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>
            <w:right w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>
            <w:insideH w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>
            <w:insideV w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>
        </w:tblBorders>
    ''')
    tblPr.append(borders)

def build_declaration_docx(output_path):
    doc = Document()

    # Page setup: Standard Letter with 1 inch margins
    section = doc.sections[0]
    section.page_width = Inches(8.5)
    section.page_height = Inches(11.0)
    section.top_margin = Inches(0.9)
    section.bottom_margin = Inches(0.9)
    section.left_margin = Inches(1.0)
    section.right_margin = Inches(1.0)

    # Configure default style font
    doc.styles['Normal'].font.name = 'Arial'
    doc.styles['Normal'].font.size = Pt(11)
    doc.styles['Normal'].font.color.rgb = RGBColor(0x11, 0x11, 0x11)

    # Title: ANNEX 1
    p0 = doc.add_paragraph()
    p0.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p0.paragraph_format.space_before = Pt(0)
    p0.paragraph_format.space_after = Pt(2)
    r0 = p0.add_run('ANNEX 1')
    r0.bold = True
    r0.font.size = Pt(12)
    r0.font.name = 'Arial'

    # Heading: DECLARATION ON THE USE OF ARTIFICIAL INTELLIGENCE
    p1 = doc.add_paragraph()
    p1.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p1.paragraph_format.space_before = Pt(0)
    p1.paragraph_format.space_after = Pt(12)
    r1 = p1.add_run('DECLARATION ON THE USE OF ARTIFICIAL INTELLIGENCE')
    r1.bold = True
    r1.font.size = Pt(13)
    r1.font.name = 'Arial'

    # Declaration statement paragraph
    p2 = doc.add_paragraph()
    p2.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p2.paragraph_format.line_spacing = 1.15
    p2.paragraph_format.space_after = Pt(8)

    p2.add_run('We, ')
    r_names = p2.add_run('Shikina Y. Cabral, John Christian B. Caparas, Matan John B. Exconde, and Geuel John D. Rivera')
    r_names.bold = True
    p2.add_run(', students of the Polytechnic University of the Philippines, hereby declare that we ')
    r_have = p2.add_run('HAVE')
    r_have.bold = True
    p2.add_run(' used Artificial Intelligence to complete this research project/thesis titled ')
    quote_open = chr(8220)
    quote_close = chr(8221)
    r_title = p2.add_run(f'{quote_open}A Multimodal Deepfake Detection Framework Leveraging Bilinear Pooling and Emotion Mismatch{quote_close}')
    r_title.bold = True
    p2.add_run('.')

    p3 = doc.add_paragraph()
    p3.paragraph_format.space_after = Pt(6)
    p3.add_run('The use of Artificial Intelligence is in the following manner:')

    # Add main table
    table = doc.add_table(rows=4, cols=3)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(table, color="B0B0B0", sz="4")

    col_widths = [Inches(1.45), Inches(2.65), Inches(2.40)]
    for row in table.rows:
        for i, w in enumerate(col_widths):
            row.cells[i].width = w

    headers = ['AI Tool Used', 'Detailed Prompts and AI Outputs', 'How the Output was Incorporated in the Project']
    hdr_row = table.rows[0]
    for i, title in enumerate(headers):
        cell = hdr_row.cells[i]
        set_cell_margins(cell, top=100, bottom=100, left=120, right=120)
        set_cell_shading(cell, 'EAEAEA')
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(2)
        p.paragraph_format.space_after = Pt(2)
        run = p.add_run(title)
        run.bold = True
        run.font.size = Pt(9.5)
        run.font.name = 'Arial'

    # -------------------------------------------------------------
    # Row 1: Language & Grammar
    # -------------------------------------------------------------
    cell0 = table.rows[1].cells[0]
    set_cell_margins(cell0, top=90, bottom=90, left=110, right=110)
    p = cell0.paragraphs[0]
    p.paragraph_format.space_after = Pt(2)
    r = p.add_run('DeepSeek / Claude / Gemini\n')
    r.bold = True
    r.font.size = Pt(9)
    r2 = p.add_run('(Language, Grammar & Manuscript Styling)')
    r2.font.italic = True
    r2.font.size = Pt(8.5)

    cell1 = table.rows[1].cells[1]
    set_cell_margins(cell1, top=90, bottom=90, left=110, right=110)
    p = cell1.paragraphs[0]
    p.paragraph_format.space_after = Pt(2)
    r = p.add_run('Prompts:\n')
    r.bold = True
    r.font.size = Pt(9)
    r_p1 = p.add_run(
        '• "Proofread this methodology section on Compact Bilinear Pooling (Z_at ⊗ Z_v → 256D) for grammatical correctness, academic tone, and active-voice flow."\n'
        '• "Check for consistency in mathematical notation throughout Chapter 3 (e.g., notation for discrepancy vector Δ = |P_A - P_B|, Kronecker product ⊗, and feature tensors)."\n'
        '• "Help condense this literature review paragraph on affective neurobiology to meet conference length limits without omitting key citations."\n\n'
    )
    r_p1.font.size = Pt(8)
    r_out = p.add_run('Outputs:\n')
    r_out.bold = True
    r_out.font.size = Pt(9)
    r_out_text = p.add_run('Grammar and spelling corrections, academic vocabulary refinements, sentence structure tightening, and LaTeX typography suggestions.')
    r_out_text.font.size = Pt(8)

    cell2 = table.rows[1].cells[2]
    set_cell_margins(cell2, top=90, bottom=90, left=110, right=110)
    p = cell2.paragraphs[0]
    p.paragraph_format.space_after = Pt(2)
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    r_inc1 = p.add_run(
        '• All conceptual arguments, scientific hypotheses, mathematical derivations, and analytical evaluations were originally authored by the researchers.\n'
        '• AI recommendations were reviewed individually, and only stylistic and grammatical improvements were manually applied.\n'
        '• Final manuscript phrasing, academic claims, and conclusions remain strictly the original intellectual work of the proponents.'
    )
    r_inc1.font.size = Pt(8)

    # -------------------------------------------------------------
    # Row 2: Code Review & Debugging
    # -------------------------------------------------------------
    cell0 = table.rows[2].cells[0]
    set_cell_margins(cell0, top=90, bottom=90, left=110, right=110)
    p = cell0.paragraphs[0]
    p.paragraph_format.space_after = Pt(2)
    r = p.add_run('Claude / Antigravity / Gemini\n')
    r.bold = True
    r.font.size = Pt(9)
    r2 = p.add_run('(Code Review, Linting & Pipeline Debugging)')
    r2.font.italic = True
    r2.font.size = Pt(8.5)

    cell1 = table.rows[2].cells[1]
    set_cell_margins(cell1, top=90, bottom=90, left=110, right=110)
    p = cell1.paragraphs[0]
    p.paragraph_format.space_after = Pt(2)
    r = p.add_run('Prompts:\n')
    r.bold = True
    r.font.size = Pt(9)
    r_p2 = p.add_run(
        '• "Review our PyTorch feature extraction pipeline and suggest DataLoader pin_memory and CUDA cache clearing strategies to prevent VRAM bottlenecks."\n'
        '• "Help troubleshoot why torchaudio.load() threw TorchCodec runtime errors on Windows, and verify our soundfile fallback routine."\n'
        '• "Check our FastAPI async detection endpoints for unhandled exceptions and Pydantic validation syntax."\n'
        '• "Suggest WCAG 2.1 AA-compliant color contrast values and CSS layout rules for the forensic web analyst dashboard."\n\n'
    )
    r_p2.font.size = Pt(8)
    r_out = p.add_run('Outputs:\n')
    r_out.bold = True
    r_out.font.size = Pt(9)
    r_out_text = p.add_run('Syntax linting, memory leak warnings (torch.cuda.empty_cache), Windows C++ backend DLL workarounds, and CSS contrast/flexbox suggestions.')
    r_out_text.font.size = Pt(8)

    cell2 = table.rows[2].cells[2]
    set_cell_margins(cell2, top=90, bottom=90, left=110, right=110)
    p = cell2.paragraphs[0]
    p.paragraph_format.space_after = Pt(2)
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    r_inc2 = p.add_run(
        '• All core neural network architectures (299D Hybrid Bottleneck, Compact Bilinear Pooling, Multi-Head Emotion Classifiers), training loss formulations, data preprocessors, and evaluation scripts were designed and implemented by the proponents.\n'
        '• AI was utilized strictly as an interactive static code reviewer, linter, and runtime error troubleshooter.\n'
        '• No machine learning algorithms, dataset pipelines, or experimental benchmark results were autonomously generated.'
    )
    r_inc2.font.size = Pt(8)

    # -------------------------------------------------------------
    # Row 3: Defense Prep & Statistical Validation
    # -------------------------------------------------------------
    cell0 = table.rows[3].cells[0]
    set_cell_margins(cell0, top=90, bottom=90, left=110, right=110)
    p = cell0.paragraphs[0]
    p.paragraph_format.space_after = Pt(2)
    r = p.add_run('Claude / DeepSeek\n')
    r.bold = True
    r.font.size = Pt(9)
    r2 = p.add_run('(Defense Rehearsal & Statistical Verification)')
    r2.font.italic = True
    r2.font.size = Pt(8.5)

    cell1 = table.rows[3].cells[1]
    set_cell_margins(cell1, top=90, bottom=90, left=110, right=110)
    p = cell1.paragraphs[0]
    p.paragraph_format.space_after = Pt(2)
    r = p.add_run('Prompts:\n')
    r.bold = True
    r.font.size = Pt(9)
    r_p3 = p.add_run(
        '• "Act as a strict thesis panelist and ask probing technical questions regarding our multi-modal feature gating under silent-speech conditions."\n'
        '• "What potential edge cases might reviewers ask concerning our speaker-disjoint cross-dataset evaluation protocol on FakeAVCeleb?"\n'
        '• "Verify the standard reporting conventions for paired DeLong non-parametric test statistics (Z = 9.87, p < 0.001) and 95% bootstrap confidence intervals."\n\n'
    )
    r_p3.font.size = Pt(8)
    r_out = p.add_run('Outputs:\n')
    r_out.bold = True
    r_out.font.size = Pt(9)
    r_out_text = p.add_run('Simulated panelist examination questions, edge-case challenge scenarios, presentation clarity critiques, and verification of statistical reporting terminology.')
    r_out_text.font.size = Pt(8)

    cell2 = table.rows[3].cells[2]
    set_cell_margins(cell2, top=90, bottom=90, left=110, right=110)
    p = cell2.paragraphs[0]
    p.paragraph_format.space_after = Pt(2)
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    r_inc3 = p.add_run(
        '• Used solely as an interactive practice tool to anticipate defense questions, challenge assumptions, and refine oral presentation clarity.\n'
        '• The theoretical justifications, paired DeLong statistical comparisons, and experimental benchmark tables are based exclusively on our empirical trials, dataset splits, and codebase.'
    )
    r_inc3.font.size = Pt(8)

    # -------------------------------------------------------------
    # Signatures
    # -------------------------------------------------------------
    p_sig_head = doc.add_paragraph()
    p_sig_head.paragraph_format.space_before = Pt(12)
    p_sig_head.paragraph_format.space_after = Pt(8)
    r_sig = p_sig_head.add_run('Dated and Signed:')
    r_sig.bold = True
    r_sig.font.size = Pt(10.5)

    sig_table = doc.add_table(rows=2, cols=2)
    sig_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for row in sig_table.rows:
        row.cells[0].width = Inches(3.25)
        row.cells[1].width = Inches(3.25)

    # Row 1: JC Caparas & Shikina Cabral
    c0 = sig_table.rows[0].cells[0]
    set_cell_margins(c0, top=40, bottom=60, left=40, right=40)
    p = c0.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(2)
    p.add_run('__________________________________________\n').font.size = Pt(9.5)
    r_name = p.add_run('John Christian B. Caparas\n')
    r_name.bold = True
    r_name.font.size = Pt(10)
    r_sub = p.add_run('Signature Above Printed Name / Date')
    r_sub.font.size = Pt(8)
    r_sub.font.italic = True

    c1 = sig_table.rows[0].cells[1]
    set_cell_margins(c1, top=40, bottom=60, left=40, right=40)
    p = c1.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(2)
    p.add_run('__________________________________________\n').font.size = Pt(9.5)
    r_name = p.add_run('Shikina Y. Cabral\n')
    r_name.bold = True
    r_name.font.size = Pt(10)
    r_sub = p.add_run('Signature Above Printed Name / Date')
    r_sub.font.size = Pt(8)
    r_sub.font.italic = True

    # Row 2: Matan Exconde & Geuel Rivera
    c0 = sig_table.rows[1].cells[0]
    set_cell_margins(c0, top=100, bottom=40, left=40, right=40)
    p = c0.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(2)
    p.add_run('__________________________________________\n').font.size = Pt(9.5)
    r_name = p.add_run('Matan John B. Exconde\n')
    r_name.bold = True
    r_name.font.size = Pt(10)
    r_sub = p.add_run('Signature Above Printed Name / Date')
    r_sub.font.size = Pt(8)
    r_sub.font.italic = True

    c1 = sig_table.rows[1].cells[1]
    set_cell_margins(c1, top=100, bottom=40, left=40, right=40)
    p = c1.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(2)
    p.add_run('__________________________________________\n').font.size = Pt(9.5)
    r_name = p.add_run('Geuel John D. Rivera\n')
    r_name.bold = True
    r_name.font.size = Pt(10)
    r_sub = p.add_run('Signature Above Printed Name / Date')
    r_sub.font.size = Pt(8)
    r_sub.font.italic = True

    doc.save(output_path)
    print(f'Successfully wrote declaration document to: {output_path}')

if __name__ == '__main__':
    target = r'c:\Users\capar\Downloads\Group10_AIDeclaration-1.docx'
    build_declaration_docx(target)
