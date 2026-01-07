"""
Generate comprehensive documentation for causal chain analysis
with physical explanations and scoring methodology
"""

from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.style import WD_STYLE_TYPE

def create_causal_chain_documentation():
    """
    Create detailed documentation of causal chains with physics and scoring
    """
    doc = Document()
    
    # ========================================================================
    # TITLE PAGE
    # ========================================================================
    
    title = doc.add_heading('PCB Defect Detection System', 0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    
    subtitle = doc.add_paragraph('Causal Chain Analysis & Scoring Methodology')
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    subtitle_format = subtitle.runs[0]
    subtitle_format.font.size = Pt(16)
    subtitle_format.font.color.rgb = RGBColor(0, 0, 0)
    
    doc.add_paragraph()
    
    version = doc.add_paragraph('Version 1.0')
    version.alignment = WD_ALIGN_PARAGRAPH.CENTER
    
    date = doc.add_paragraph('January 2026')
    date.alignment = WD_ALIGN_PARAGRAPH.CENTER
    
    doc.add_page_break()
    
    # ========================================================================
    # TABLE OF CONTENTS
    # ========================================================================
    
    doc.add_heading('Table of Contents', 1)
    
    toc_items = [
        '1. System Overview',
        '2. Physical Understanding',
        '   2.1 Mechanism 1: Poor Paste Transfer → Open Circuits',
        '   2.2 Mechanism 2: Aperture Overfill → Solder Bridging',
        '3. Three-Level Causal Chain Architecture',
        '   3.1 Level 3: Parameter Violations (Root Causes)',
        '   3.2 Level 2: Failure Mechanisms (Intermediate Causes)',
        '   3.3 Level 1: Observable Defects (Final Outcomes)',
        '4. Scoring Methodology',
        '   4.1 Violation Probability Calculation',
        '   4.2 MLP Confidence Scores',
        '   4.3 Chain Strength Aggregation',
        '5. Complete Examples',
        '   5.1 Scenario 1: Open Circuit Chain',
        '   5.2 Scenario 2: Solder Bridging Chain',
        '   5.3 Scenario 3: Nominal Process (No Defects)',
        '6. Evidence Strength for Mode C Integration',
        '7. References'
    ]
    
    for item in toc_items:
        doc.add_paragraph(item, style='List Number')
    
    doc.add_page_break()
    
    # ========================================================================
    # 1. SYSTEM OVERVIEW
    # ========================================================================
    
    doc.add_heading('1. System Overview', 1)
    
    doc.add_paragraph(
        'The PCB Defect Detection System uses a hybrid approach combining '
        'rule-based violation detection with machine learning (Multi-Layer Perceptron) '
        'to predict defects through causal chains. The system operates on three levels:'
    )
    
    levels = [
        'Level 3 (Root Causes): Parameter violations detected via deterministic rules',
        'Level 2 (Mechanisms): Failure modes predicted by MLP based on parameter patterns',
        'Level 1 (Defects): Observable outcomes predicted by MLP based on mechanisms and parameters'
    ]
    
    for level in levels:
        p = doc.add_paragraph(level, style='List Bullet')
        p.paragraph_format.left_indent = Inches(0.5)
    
    doc.add_paragraph()
    
    doc.add_paragraph(
        'This architecture provides complete diagnostic information: '
        'what went wrong (defect), why it happened (mechanism), and '
        'what to fix (parameter violations).'
    )
    
    doc.add_page_break()
    
    # ========================================================================
    # 2. PHYSICAL UNDERSTANDING
    # ========================================================================
    
    doc.add_heading('2. Physical Understanding', 1)
    
    doc.add_paragraph(
        'The system models two primary failure mechanisms in PCB assembly, '
        'each leading to different defect types based on fundamental physics '
        'of solder paste behavior.'
    )
    
    # ------------------------------------------------------------------------
    # 2.1 Poor Paste Transfer
    # ------------------------------------------------------------------------
    
    doc.add_heading('2.1 Mechanism 1: Poor Paste Transfer → Open Circuits', 2)
    
    doc.add_paragraph(
        'Poor paste transfer occurs when insufficient solder paste is deposited '
        'from the stencil onto the PCB pads. This results in incomplete solder joints '
        'after reflow, causing open circuits (electrical disconnections).'
    )
    
    doc.add_heading('Physical Causes:', 3)
    
    causes_poor = [
        ('High Paste Viscosity', 
         'Thick, viscous paste does not release cleanly from stencil apertures. '
         'The paste adheres to the stencil rather than transferring to the board.'),
        
        ('Low Humidity', 
         'Dry conditions cause paste to lose moisture and become tacky/sticky. '
         'This increases adhesion to the stencil, reducing transfer efficiency.'),
        
        ('Low Stencil Thickness', 
         'Thin stencils contain less paste volume in their apertures, resulting in '
         'insufficient paste available for transfer to pads.'),
        
        ('Low Paste Volume', 
         'Inadequate paste dispensed means less material available overall, '
         'contributing to insufficient deposition on pads.')
    ]
    
    for cause, explanation in causes_poor:
        doc.add_paragraph(f'{cause}:', style='List Bullet').bold = True
        p = doc.add_paragraph(explanation)
        p.paragraph_format.left_indent = Inches(0.75)
        p.paragraph_format.space_after = Pt(6)
    
    doc.add_heading('Defect Outcome:', 3)
    
    doc.add_paragraph(
        'Open Circuits - During reflow, insufficient paste means inadequate solder '
        'to form complete electrical connections. The result is incomplete or missing '
        'joints between component leads and PCB pads.'
    )
    
    doc.add_paragraph()
    
    # Add parameter table for poor transfer
    doc.add_heading('Parameter Pattern for Poor Paste Transfer:', 3)
    
    table = doc.add_table(rows=6, cols=3)
    table.style = 'Light Grid Accent 1'
    
    # Header row
    header_cells = table.rows[0].cells
    header_cells[0].text = 'Parameter'
    header_cells[1].text = 'Direction'
    header_cells[2].text = 'Effect on Transfer'
    
    # Make header bold
    for cell in header_cells:
        for paragraph in cell.paragraphs:
            for run in paragraph.runs:
                run.font.bold = True
    
    # Data rows
    data_poor = [
        ('Paste Volume', 'LOW', 'Less material available'),
        ('Stencil Thickness', 'LOW', 'Less paste in apertures'),
        ('Paste Viscosity', 'HIGH', 'Paste sticks to stencil'),
        ('Ambient Humidity', 'LOW', 'Paste becomes tacky'),
        ('Temperature', 'LOW', 'Increases effective viscosity')
    ]
    
    for i, (param, direction, effect) in enumerate(data_poor, start=1):
        row_cells = table.rows[i].cells
        row_cells[0].text = param
        row_cells[1].text = direction
        row_cells[2].text = effect
    
    doc.add_paragraph()
    
    # ------------------------------------------------------------------------
    # 2.2 Aperture Overfill
    # ------------------------------------------------------------------------
    
    doc.add_heading('2.2 Mechanism 2: Aperture Overfill → Solder Bridging', 2)
    
    doc.add_paragraph(
        'Aperture overfill occurs when excessive solder paste is deposited on PCB pads, '
        'and the paste spreads beyond pad boundaries. During reflow, this excess solder '
        'can bridge between adjacent pads, creating electrical shorts.'
    )
    
    doc.add_heading('Physical Causes:', 3)
    
    causes_over = [
        ('Low Paste Viscosity', 
         'Thin, fluid paste spreads easily and slumps after printing. '
         'Low viscosity allows paste to flow beyond intended pad areas.'),
        
        ('High Humidity', 
         'Moisture absorption reduces effective paste viscosity, promoting '
         'spreading and making the paste more prone to slumping.'),
        
        ('High Stencil Thickness', 
         'Thick stencils contain more paste volume in apertures, depositing '
         'excess material that can spread beyond pad boundaries.'),
        
        ('High Paste Volume', 
         'Excessive paste dispensed provides more material than needed, '
         'increasing likelihood of overfill and spreading.'),
        
        ('High Temperature',
         'Elevated temperatures reduce paste viscosity further, increasing '
         'flow and spreading tendencies.')
    ]
    
    for cause, explanation in causes_over:
        doc.add_paragraph(f'{cause}:', style='List Bullet').bold = True
        p = doc.add_paragraph(explanation)
        p.paragraph_format.left_indent = Inches(0.75)
        p.paragraph_format.space_after = Pt(6)
    
    doc.add_heading('Defect Outcome:', 3)
    
    doc.add_paragraph(
        'Solder Bridging - Excess paste spreads between adjacent pads. During reflow, '
        'this creates unwanted solder connections (bridges) between pads, resulting in '
        'electrical shorts that can cause circuit malfunction.'
    )
    
    doc.add_paragraph()
    
    # Add parameter table for overfill
    doc.add_heading('Parameter Pattern for Aperture Overfill:', 3)
    
    table2 = doc.add_table(rows=6, cols=3)
    table2.style = 'Light Grid Accent 1'
    
    # Header row
    header_cells2 = table2.rows[0].cells
    header_cells2[0].text = 'Parameter'
    header_cells2[1].text = 'Direction'
    header_cells2[2].text = 'Effect on Overfill'
    
    # Make header bold
    for cell in header_cells2:
        for paragraph in cell.paragraphs:
            for run in paragraph.runs:
                run.font.bold = True
    
    # Data rows
    data_over = [
        ('Paste Volume', 'HIGH', 'Excess material available'),
        ('Stencil Thickness', 'HIGH', 'More paste in apertures'),
        ('Paste Viscosity', 'LOW', 'Paste spreads easily'),
        ('Ambient Humidity', 'HIGH', 'Reduces effective viscosity'),
        ('Temperature', 'HIGH', 'Further reduces viscosity')
    ]
    
    for i, (param, direction, effect) in enumerate(data_over, start=1):
        row_cells = table2.rows[i].cells
        row_cells[0].text = param
        row_cells[1].text = direction
        row_cells[2].text = effect
    
    doc.add_page_break()
    
    # ========================================================================
    # 3. THREE-LEVEL CAUSAL CHAIN ARCHITECTURE
    # ========================================================================
    
    doc.add_heading('3. Three-Level Causal Chain Architecture', 1)
    
    doc.add_paragraph(
        'The system analyzes PCB quality through three hierarchical levels, '
        'moving from root causes (parameter violations) through intermediate '
        'mechanisms to final observable outcomes (defects).'
    )
    
    # ------------------------------------------------------------------------
    # 3.1 Level 3: Parameter Violations
    # ------------------------------------------------------------------------
    
    doc.add_heading('3.1 Level 3: Parameter Violations (Root Causes)', 2)
    
    doc.add_paragraph(
        'Parameter violations are detected using deterministic rule-based calculations. '
        'Each process parameter has defined specification limits (Upper Specification Limit '
        'and Lower Specification Limit). Violations occur when parameters exceed these limits.'
    )
    
    doc.add_heading('Parameters Monitored:', 3)
    
    params = [
        ('Paste Volume', '0.036 - 0.044 mm³', 'Amount of solder paste dispensed'),
        ('Stencil Thickness', '95 - 105 µm', 'Thickness of printing stencil'),
        ('Paste Viscosity', '150 - 250 Pa·s', 'Flowability of solder paste'),
        ('Ambient Humidity', '30 - 50 %', 'Moisture level in assembly area'),
        ('Ambient Temperature', '20 - 26 °C', 'Temperature in assembly area')
    ]
    
    param_table = doc.add_table(rows=6, cols=3)
    param_table.style = 'Light Grid Accent 1'
    
    # Header
    header = param_table.rows[0].cells
    header[0].text = 'Parameter'
    header[1].text = 'Specification Range'
    header[2].text = 'Description'
    
    for cell in header:
        for paragraph in cell.paragraphs:
            for run in paragraph.runs:
                run.font.bold = True
    
    # Data
    for i, (param, spec, desc) in enumerate(params, start=1):
        row = param_table.rows[i].cells
        row[0].text = param
        row[1].text = spec
        row[2].text = desc
    
    doc.add_paragraph()
    
    doc.add_heading('Violation Detection Method:', 3)
    
    doc.add_paragraph(
        'Source: Rule-Based (Deterministic)\n'
        'Accuracy: 100% (mathematical comparison)\n'
        'Output: Binary (violated/not violated) + Risk probability'
    )
    
    doc.add_heading('Risk Probability Calculation:', 3)
    
    doc.add_paragraph(
        'Rather than simple binary detection, the system calculates a risk probability '
        'based on how close a parameter is to its limit and how far it exceeds the limit. '
        'This models the actual physics: parameters near or beyond limits have progressively '
        'higher probability of causing mechanisms and defects.'
    )
    
    doc.add_paragraph(
        'The probability ranges from 0.0 (far from limit, minimal risk) to 1.0 '
        '(way beyond limit, critical risk). Parameters within specification but '
        'approaching limits receive moderate probabilities (e.g., 0.3-0.5), providing '
        'early warning of potential issues.'
    )
    
    doc.add_page_break()
    
    # ------------------------------------------------------------------------
    # 3.2 Level 2: Failure Mechanisms
    # ------------------------------------------------------------------------
    
    doc.add_heading('3.2 Level 2: Failure Mechanisms (Intermediate Causes)', 2)
    
    doc.add_paragraph(
        'Failure mechanisms represent the physical processes that link parameter '
        'violations to observable defects. These are predicted using a Multi-Layer '
        'Perceptron (MLP) trained on 200,000 samples of historical process data.'
    )
    
    doc.add_heading('Mechanisms Predicted:', 3)
    
    mechanisms = [
        ('poorpastetransfer', 
         'Insufficient paste transfer from stencil to board pads',
         'Open Circuits'),
        
        ('apertureoverfill', 
         'Excessive paste deposition and spreading beyond pad boundaries',
         'Solder Bridging')
    ]
    
    mech_table = doc.add_table(rows=3, cols=3)
    mech_table.style = 'Light Grid Accent 1'
    
    # Header
    mech_header = mech_table.rows[0].cells
    mech_header[0].text = 'Mechanism'
    mech_header[1].text = 'Description'
    mech_header[2].text = 'Leads To'
    
    for cell in mech_header:
        for paragraph in cell.paragraphs:
            for run in paragraph.runs:
                run.font.bold = True
    
    # Data
    for i, (mech, desc, leads) in enumerate(mechanisms, start=1):
        row = mech_table.rows[i].cells
        row[0].text = mech
        row[1].text = desc
        row[2].text = leads
    
    doc.add_paragraph()
    
    doc.add_heading('Prediction Method:', 3)
    
    doc.add_paragraph(
        'Source: Multi-Layer Perceptron (Learned from Data)\n'
        'Architecture: 18 input features → 256 → 128 → 64 → 1 output\n'
        'Training: 140,000 samples, validated on 30,000\n'
        'Performance: 82% F1 score on test set\n'
        'Output: Probability (0.0-1.0) that mechanism is present'
    )
    
    doc.add_heading('How MLP Makes Predictions:', 3)
    
    doc.add_paragraph(
        'The MLP receives 18 engineered features derived from the 5 raw process parameters. '
        'These features include ratios, interactions, and transformations that capture '
        'complex relationships between parameters. The network learned patterns such as:'
    )
    
    patterns = [
        'Low paste volume + thin stencil + high viscosity → poorpastetransfer (78% prob)',
        'High paste volume + thick stencil + low viscosity → apertureoverfill (86% prob)',
        'Combined effects: multiple violations amplify mechanism probability'
    ]
    
    for pattern in patterns:
        p = doc.add_paragraph(pattern, style='List Bullet')
        p.paragraph_format.left_indent = Inches(0.5)
    
    doc.add_paragraph()
    
    doc.add_paragraph(
        'The MLP captures subtle interactions that rule-based systems cannot, such as '
        'how humidity and temperature jointly affect paste viscosity, or how stencil '
        'thickness and paste volume interact to determine transfer efficiency.'
    )
    
    doc.add_page_break()
    
    # ------------------------------------------------------------------------
    # 3.3 Level 1: Observable Defects
    # ------------------------------------------------------------------------
    
    doc.add_heading('3.3 Level 1: Observable Defects (Final Outcomes)', 2)
    
    doc.add_paragraph(
        'Observable defects are the final outcomes that can be detected through visual '
        'inspection or Automated Optical Inspection (AOI). These are also predicted using '
        'the MLP, which learned associations between parameter patterns, mechanisms, and '
        'defect types from historical data.'
    )
    
    doc.add_heading('Defects Detected:', 3)
    
    defects = [
        ('No Defect', 
         'Board passes quality inspection with no issues detected'),
        
        ('Open Circuit', 
         'Incomplete or missing solder joints resulting in electrical disconnections'),
        
        ('Solder Bridging',
         'Unwanted solder connections between adjacent pads causing electrical shorts')
    ]
    
    defect_table = doc.add_table(rows=4, cols=2)
    defect_table.style = 'Light Grid Accent 1'
    
    # Header
    defect_header = defect_table.rows[0].cells
    defect_header[0].text = 'Defect Type'
    defect_header[1].text = 'Description'
    
    for cell in defect_header:
        for paragraph in cell.paragraphs:
            for run in paragraph.runs:
                run.font.bold = True
    
    # Data
    for i, (defect, desc) in enumerate(defects, start=1):
        row = defect_table.rows[i].cells
        row[0].text = defect
        row[1].text = desc
    
    doc.add_paragraph()
    
    doc.add_heading('Prediction Method:', 3)
    
    doc.add_paragraph(
        'Source: Multi-Layer Perceptron (Learned from Data)\n'
        'Architecture: 18 input features → 256 → 128 → 64 → 3 outputs (softmax)\n'
        'Training: 140,000 samples, validated on 30,000\n'
        'Performance: 98% F1 score on test set\n'
        'Output: Probability distribution over 3 defect classes'
    )
    
    doc.add_heading('How MLP Makes Predictions:', 3)
    
    doc.add_paragraph(
        'The defect prediction head uses the same shared encoder as the mechanism prediction. '
        'This means it leverages learned representations about parameter patterns and their '
        'effects. The key learned associations include:'
    )
    
    associations = [
        'poorpastetransfer + low parameter pattern → Open Circuit (82% confidence)',
        'apertureoverfill + high parameter pattern → Solder Bridging (91% confidence)',
        'Nominal parameters + no mechanism → No Defect (95% confidence)'
    ]
    
    for assoc in associations:
        p = doc.add_paragraph(assoc, style='List Bullet')
        p.paragraph_format.left_indent = Inches(0.5)
    
    doc.add_paragraph()
    
    doc.add_paragraph(
        'The model outputs a probability distribution over all three classes using softmax, '
        'ensuring probabilities sum to 1.0. The class with highest probability is the '
        'predicted defect type.'
    )
    
    doc.add_page_break()
    
    # ========================================================================
    # 4. SCORING METHODOLOGY
    # ========================================================================
    
    doc.add_heading('4. Scoring Methodology', 1)
    
    doc.add_paragraph(
        'Each level in the causal chain produces numerical scores that quantify '
        'confidence and risk. Understanding how these scores are calculated and '
        'aggregated is essential for interpreting results and making decisions.'
    )
    
    # ------------------------------------------------------------------------
    # 4.1 Violation Probability Calculation
    # ------------------------------------------------------------------------
    
    doc.add_heading('4.1 Violation Probability Calculation (Level 3)', 2)
    
    doc.add_paragraph(
        'Violation probabilities represent the likelihood that a parameter state '
        'will contribute to a mechanism or defect. The calculation considers both '
        'proximity to limits and exceedance beyond limits.'
    )
    
    doc.add_heading('Calculation Formula:', 3)
    
    doc.add_paragraph(
        'For parameters approaching but within specification:'
    )
    
    doc.add_paragraph(
        'probability = 0.05 + 0.65 × (distance_to_limit / limit_range)²'
    )
    
    doc.add_paragraph(
        'This quadratic function starts at ~5% (minimal risk at nominal) and reaches '
        '~70% at the specification limit.'
    )
    
    doc.add_paragraph()
    
    doc.add_paragraph(
        'For parameters exceeding specification limits:'
    )
    
    doc.add_paragraph(
        'probability = 0.70 + 0.30 × (1 - exp(-exceedance × 2))'
    )
    
    doc.add_paragraph(
        'This exponential function starts at 70% (at limit) and asymptotically '
        'approaches 100% as exceedance increases.'
    )
    
    doc.add_heading('Risk Level Classification:', 3)
    
    risk_levels = [
        ('< 10%', 'Minimal', 'Process stable, normal operation'),
        ('10-30%', 'Low', 'Monitor parameter trends'),
        ('30-50%', 'Moderate', 'Attention needed, approaching warning zone'),
        ('50-70%', 'High', 'Adjust process parameters soon'),
        ('> 70%', 'Critical', 'Immediate corrective action required')
    ]
    
    risk_table = doc.add_table(rows=6, cols=3)
    risk_table.style = 'Light Grid Accent 1'
    
    # Header
    risk_header = risk_table.rows[0].cells
    risk_header[0].text = 'Probability Range'
    risk_header[1].text = 'Risk Level'
    risk_header[2].text = 'Action Required'
    
    for cell in risk_header:
        for paragraph in cell.paragraphs:
            for run in paragraph.runs:
                run.font.bold = True
    
    # Data
    for i, (prob, level, action) in enumerate(risk_levels, start=1):
        row = risk_table.rows[i].cells
        row[0].text = prob
        row[1].text = level
        row[2].text = action
    
    doc.add_paragraph()
    
    doc.add_heading('Aggregated Violation Score:', 3)
    
    doc.add_paragraph(
        'When multiple parameters show high risk, an overall violation score is '
        'calculated as the average of all high-risk (probability > 30%) parameters. '
        'This provides a single metric representing the severity of process deviation.'
    )
    
    doc.add_paragraph(
        'Example: If paste_volume_high = 0.85, paste_viscosity_low = 0.52, and '
        'ambient_rh_high = 0.78, the overall violation score = (0.85 + 0.52 + 0.78) / 3 = 0.72'
    )
    
    doc.add_page_break()
    
    # ------------------------------------------------------------------------
    # 4.2 MLP Confidence Scores
    # ------------------------------------------------------------------------
    
    doc.add_heading('4.2 MLP Confidence Scores (Levels 2 & 1)', 2)
    
    doc.add_paragraph(
        'The MLP outputs represent learned probabilities based on training data patterns. '
        'These are fundamentally different from rule-based violation probabilities:'
    )
    
    mlp_points = [
        ('Violation scores are deterministic', 
         'Given a parameter value, the probability is always the same (mathematical calculation)'),
        
        ('MLP scores are probabilistic', 
         'Learned from data, capturing complex patterns and interactions the model discovered'),
        
        ('MLP captures context',
         'The same violation might have different mechanism probabilities depending on other parameters'),
        
        ('MLP incorporates uncertainty',
         'Confidence reflects both training data patterns and model uncertainty')
    ]
    
    for title, explanation in mlp_points:
        doc.add_paragraph(f'{title}:', style='List Bullet').bold = True
        p = doc.add_paragraph(explanation)
        p.paragraph_format.left_indent = Inches(0.75)
        p.paragraph_format.space_after = Pt(6)
    
    doc.add_heading('Mechanism Scores (Binary Classification):', 3)
    
    doc.add_paragraph(
        'The mechanism head outputs a single value between 0 and 1 using sigmoid '
        'activation. This represents P(mechanism present | parameters).'
    )
    
    doc.add_paragraph(
        'Score > 0.5: Mechanism predicted as PRESENT\n'
        'Score ≤ 0.5: Mechanism predicted as ABSENT'
    )
    
    doc.add_heading('Defect Scores (Multi-Class Classification):', 3)
    
    doc.add_paragraph(
        'The defect head outputs three values that sum to 1.0 using softmax activation. '
        'These represent the probability distribution over defect types:'
    )
    
    doc.add_paragraph(
        'P(No Defect) + P(Open Circuit) + P(Solder Bridging) = 1.0'
    )
    
    doc.add_paragraph(
        'The defect with the highest probability is the predicted class. The probability '
        'of the predicted class serves as the confidence score.'
    )
    
    doc.add_heading('Calibration:', 3)
    
    doc.add_paragraph(
        'The MLP was trained with proper calibration techniques to ensure its confidence '
        'scores are well-calibrated. This means:'
    )
    
    calibration = [
        'If the model predicts 80% confidence, it should be correct ~80% of the time',
        'Expected Calibration Error (ECE) measured at 1.2% (excellent calibration)',
        'Confidence scores can be trusted for decision-making'
    ]
    
    for point in calibration:
        p = doc.add_paragraph(point, style='List Bullet')
        p.paragraph_format.left_indent = Inches(0.5)
    
    doc.add_page_break()
    
    # ------------------------------------------------------------------------
    # 4.3 Chain Strength Aggregation
    # ------------------------------------------------------------------------
    
    doc.add_heading('4.3 Chain Strength Aggregation', 2)
    
    doc.add_paragraph(
        'A complete causal chain connects violations → mechanism → defect with scores '
        'at each level. Multiple aggregation methods provide different insights:'
    )
    
    doc.add_heading('Method 1: Average Path Strength', 3)
    
    doc.add_paragraph(
        'Average all three levels to get overall chain confidence:'
    )
    
    doc.add_paragraph(
        'Chain Strength = (Violation Score + Mechanism Score + Defect Score) / 3'
    )
    
    doc.add_paragraph(
        'Example: (0.72 + 0.86 + 0.91) / 3 = 0.83 (83% confidence in complete causal path)'
    )
    
    doc.add_paragraph(
        'Interpretation: Represents the average confidence across all causal levels. '
        'A high score (>0.7) indicates strong evidence throughout the entire chain.'
    )
    
    doc.add_heading('Method 2: Minimum Link Strength (Weakest Link)', 3)
    
    doc.add_paragraph(
        'Use the minimum score across all levels:'
    )
    
    doc.add_paragraph(
        'Chain Strength = min(Violation Score, Mechanism Score, Defect Score)'
    )
    
    doc.add_paragraph(
        'Example: min(0.72, 0.86, 0.91) = 0.72'
    )
    
    doc.add_paragraph(
        'Interpretation: Conservative measure - the chain is only as strong as its weakest link. '
        'Useful for high-stakes decisions where you need confidence at every level.'
    )
    
    doc.add_heading('Method 3: Maximum Link Strength (Strongest Evidence)', 3)
    
    doc.add_paragraph(
        'Use the maximum score across all levels:'
    )
    
    doc.add_paragraph(
        'Chain Strength = max(Violation Score, Mechanism Score, Defect Score)'
    )
    
    doc.add_paragraph(
        'Example: max(0.72, 0.86, 0.91) = 0.91'
    )
    
    doc.add_paragraph(
        'Interpretation: Optimistic measure - highlights the strongest evidence in the chain. '
        'Useful for identifying where confidence is highest.'
    )
    
    doc.add_heading('Method 4: Weighted Average (Recommended)', 3)
    
    doc.add_paragraph(
        'Weight levels by importance and reliability:'
    )
    
    doc.add_paragraph(
        'Chain Strength = 0.4 × Violation + 0.3 × Mechanism + 0.3 × Defect'
    )
    
    doc.add_paragraph(
        'Example: 0.4 × 0.72 + 0.3 × 0.86 + 0.3 × 0.91 = 0.819'
    )
    
    doc.add_paragraph(
        'Rationale: Violations (40% weight) are most reliable (rule-based, deterministic). '
        'Mechanism and defect predictions (30% each) are learned and probabilistic, weighted equally.'
    )
    
    doc.add_heading('Choosing the Right Method:', 3)
    
    method_guide = [
        ('Average Path', 'General reporting, balanced view'),
        ('Minimum Link', 'Conservative decisions, safety-critical applications'),
        ('Maximum Link', 'Identifying strongest evidence, research'),
        ('Weighted Average', 'Production decisions, balances reliability and confidence')
    ]
    
    method_table = doc.add_table(rows=5, cols=2)
    method_table.style = 'Light Grid Accent 1'
    
    # Header
    method_header = method_table.rows[0].cells
    method_header[0].text = 'Aggregation Method'
    method_header[1].text = 'Best Use Case'
    
    for cell in method_header:
        for paragraph in cell.paragraphs:
            for run in paragraph.runs:
                run.font.bold = True
    
    # Data
    for i, (method, use_case) in enumerate(method_guide, start=1):
        row = method_table.rows[i].cells
        row[0].text = method
        row[1].text = use_case
    
    doc.add_page_break()
    
    # ========================================================================
    # 5. COMPLETE EXAMPLES
    # ========================================================================
    
    doc.add_heading('5. Complete Examples', 1)
    
    doc.add_paragraph(
        'The following examples demonstrate complete causal chain analysis '
        'for different process conditions, showing how scores flow from '
        'violations through mechanisms to defects.'
    )
    
    # ------------------------------------------------------------------------
    # 5.1 Scenario 1: Open Circuit Chain
    # ------------------------------------------------------------------------
    
    doc.add_heading('5.1 Scenario 1: Open Circuit Causal Chain', 2)
    
    doc.add_heading('Input Parameters:', 3)
    
    scenario1_params = [
        ('Paste Volume', '0.037 mm³', 'Low (approaching LSL of 0.036)'),
        ('Stencil Thickness', '93 µm', 'Low (VIOLATED - LSL is 95)'),
        ('Paste Viscosity', '240 Pa·s', 'High (approaching USL of 250)'),
        ('Ambient Humidity', '32%', 'Low (approaching LSL of 30)'),
        ('Ambient Temperature', '23°C', 'Normal (within spec)')
    ]
    
    s1_table = doc.add_table(rows=6, cols=3)
    s1_table.style = 'Light Grid Accent 1'
    
    # Header
    s1_header = s1_table.rows[0].cells
    s1_header[0].text = 'Parameter'
    s1_header[1].text = 'Value'
    s1_header[2].text = 'Status'
    
    for cell in s1_header:
        for paragraph in cell.paragraphs:
            for run in paragraph.runs:
                run.font.bold = True
    
    # Data
    for i, (param, value, status) in enumerate(scenario1_params, start=1):
        row = s1_table.rows[i].cells
        row[0].text = param
        row[1].text = value
        row[2].text = status
    
    doc.add_paragraph()
    
    doc.add_heading('Level 3: Violation Analysis', 3)
    
    s1_violations = [
        ('paste_volume_low (approaching)', '0.55', 'High Risk'),
        ('stencil_thickness_low (VIOLATED)', '0.72', 'Critical'),
        ('paste_viscosity_high (approaching)', '0.68', 'High Risk'),
        ('ambient_rh_low (approaching)', '0.58', 'High Risk')
    ]
    
    doc.add_paragraph('Detected Violations and Risk Scores:')
    
    for violation, score, level in s1_violations:
        doc.add_paragraph(
            f'{violation}: Probability = {score} ({level})',
            style='List Bullet'
        )
    
    doc.add_paragraph()
    doc.add_paragraph(f'Overall Violation Score: 0.63 (average of high-risk parameters)')
    
    doc.add_paragraph()
    
    doc.add_paragraph(
        'Physical Pattern: Low paste volume, thin stencil, high viscosity, and low '
        'humidity create conditions where paste does not release cleanly from stencil.'
    )
    
    doc.add_paragraph()
    
    doc.add_heading('Level 2: Mechanism Prediction', 3)
    
    doc.add_paragraph('Mechanism: poorpastetransfer')
    doc.add_paragraph('MLP Probability: 0.78 (78% confidence)')
    doc.add_paragraph('Status: PRESENT')
    
    doc.add_paragraph()
    
    doc.add_paragraph(
        'Explanation: The MLP learned that this combination of low/high parameters '
        'leads to insufficient paste transfer. The paste sticks to the stencil apertures '
        'rather than transferring to board pads.'
    )
    
    doc.add_paragraph()
    
    doc.add_heading('Level 1: Defect Prediction', 3)
    
    doc.add_paragraph('Predicted Defect: Open Circuit')
    doc.add_paragraph('MLP Confidence: 0.82 (82%)')
    
    doc.add_paragraph()
    
    s1_defect_probs = [
        ('No Defect', '0.12', '12%'),
        ('Open Circuit', '0.82', '82% (PREDICTED)'),
        ('Solder Bridging', '0.06', '6%')
    ]
    
    doc.add_paragraph('Probability Distribution:')
    
    for defect, prob, pct in s1_defect_probs:
        doc.add_paragraph(f'{defect}: {prob} ({pct})', style='List Bullet')
    
    doc.add_paragraph()
    
    doc.add_paragraph(
        'Explanation: Poor paste transfer results in insufficient solder on pads. '
        'During reflow, there is not enough solder to form complete joints, leading '
        'to open circuits (incomplete electrical connections).'
    )
    
    doc.add_paragraph()
    
    doc.add_heading('Complete Causal Chain:', 3)
    
    doc.add_paragraph(
        'Violations (0.63) → poorpastetransfer (0.78) → Open Circuit (0.82)'
    )
    
    doc.add_paragraph()
    
    chain_strength = [
        ('Average Path Strength', '(0.63 + 0.78 + 0.82) / 3 = 0.74'),
        ('Minimum Link (Weakest)', 'min(0.63, 0.78, 0.82) = 0.63'),
        ('Maximum Link (Strongest)', 'max(0.63, 0.78, 0.82) = 0.82'),
        ('Weighted Average', '0.4 × 0.63 + 0.3 × 0.78 + 0.3 × 0.82 = 0.73')
    ]
    
    doc.add_paragraph('Chain Strength Metrics:')
    
    for metric, calc in chain_strength:
        doc.add_paragraph(f'{metric}: {calc}', style='List Bullet')
    
    doc.add_paragraph()
    
    doc.add_paragraph(
        'Interpretation: Strong causal evidence (average 74%) linking parameter violations '
        'through poor paste transfer mechanism to open circuit defect. Recommend immediate '
        'adjustment of paste volume, stencil thickness, and humidity.'
    )
    
    doc.add_page_break()
    
    # ------------------------------------------------------------------------
    # 5.2 Scenario 2: Solder Bridging Chain
    # ------------------------------------------------------------------------
    
    doc.add_heading('5.2 Scenario 2: Solder Bridging Causal Chain', 2)
    
    doc.add_heading('Input Parameters:', 3)
    
    scenario2_params = [
        ('Paste Volume', '0.046 mm³', 'High (VIOLATED - USL is 0.044)'),
        ('Stencil Thickness', '108 µm', 'High (VIOLATED - USL is 105)'),
        ('Paste Viscosity', '160 Pa·s', 'Low (approaching LSL of 150)'),
        ('Ambient Humidity', '53%', 'High (approaching USL of 50)'),
        ('Ambient Temperature', '27°C', 'High (VIOLATED - USL is 26)')
    ]
    
    s2_table = doc.add_table(rows=6, cols=3)
    s2_table.style = 'Light Grid Accent 1'
    
    # Header
    s2_header = s2_table.rows[0].cells
    s2_header[0].text = 'Parameter'
    s2_header[1].text = 'Value'
    s2_header[2].text = 'Status'
    
    for cell in s2_header:
        for paragraph in cell.paragraphs:
            for run in paragraph.runs:
                run.font.bold = True
    
    # Data
    for i, (param, value, status) in enumerate(scenario2_params, start=1):
        row = s2_table.rows[i].cells
        row[0].text = param
        row[1].text = value
        row[2].text = status
    
    doc.add_paragraph()
    
    doc.add_heading('Level 3: Violation Analysis', 3)
    
    s2_violations = [
        ('paste_volume_high (VIOLATED)', '0.85', 'Critical'),
        ('stencil_thickness_high (VIOLATED)', '0.78', 'Critical'),
        ('paste_viscosity_low (approaching)', '0.58', 'High Risk'),
        ('ambient_rh_high (approaching)', '0.72', 'Critical'),
        ('ambient_temperature_high (VIOLATED)', '0.68', 'High Risk')
    ]
    
    doc.add_paragraph('Detected Violations and Risk Scores:')
    
    for violation, score, level in s2_violations:
        doc.add_paragraph(
            f'{violation}: Probability = {score} ({level})',
            style='List Bullet'
        )
    
    doc.add_paragraph()
    doc.add_paragraph(f'Overall Violation Score: 0.72 (average of high-risk parameters)')
    
    doc.add_paragraph()
    
    doc.add_paragraph(
        'Physical Pattern: High paste volume, thick stencil, low viscosity, high humidity, '
        'and high temperature create conditions for excessive paste deposition and spreading.'
    )
    
    doc.add_paragraph()
    
    doc.add_heading('Level 2: Mechanism Prediction', 3)
    
    doc.add_paragraph('Mechanism: apertureoverfill')
    doc.add_paragraph('MLP Probability: 0.86 (86% confidence)')
    doc.add_paragraph('Status: PRESENT')
    
    doc.add_paragraph()
    
    doc.add_paragraph(
        'Explanation: The MLP learned that this combination of high/low parameters '
        'leads to excessive paste deposition. The thick stencil and high volume provide '
        'too much paste, while low viscosity and environmental conditions promote spreading.'
    )
    
    doc.add_paragraph()
    
    doc.add_heading('Level 1: Defect Prediction', 3)
    
    doc.add_paragraph('Predicted Defect: Solder Bridging')
    doc.add_paragraph('MLP Confidence: 0.91 (91%)')
    
    doc.add_paragraph()
    
    s2_defect_probs = [
        ('No Defect', '0.05', '5%'),
        ('Open Circuit', '0.04', '4%'),
        ('Solder Bridging', '0.91', '91% (PREDICTED)')
    ]
    
    doc.add_paragraph('Probability Distribution:')
    
    for defect, prob, pct in s2_defect_probs:
        doc.add_paragraph(f'{defect}: {prob} ({pct})', style='List Bullet')
    
    doc.add_paragraph()
    
    doc.add_paragraph(
        'Explanation: Aperture overfill results in excess solder paste on and beyond pads. '
        'The paste spreads between adjacent pads. During reflow, this excess solder creates '
        'unwanted connections (bridges) between pads, causing electrical shorts.'
    )
    
    doc.add_paragraph()
    
    doc.add_heading('Complete Causal Chain:', 3)
    
    doc.add_paragraph(
        'Violations (0.72) → apertureoverfill (0.86) → Solder Bridging (0.91)'
    )
    
    doc.add_paragraph()
    
    chain_strength2 = [
        ('Average Path Strength', '(0.72 + 0.86 + 0.91) / 3 = 0.83'),
        ('Minimum Link (Weakest)', 'min(0.72, 0.86, 0.91) = 0.72'),
        ('Maximum Link (Strongest)', 'max(0.72, 0.86, 0.91) = 0.91'),
        ('Weighted Average', '0.4 × 0.72 + 0.3 × 0.86 + 0.3 × 0.91 = 0.82')
    ]
    
    doc.add_paragraph('Chain Strength Metrics:')
    
    for metric, calc in chain_strength2:
        doc.add_paragraph(f'{metric}: {calc}', style='List Bullet')
    
    doc.add_paragraph()
    
    doc.add_paragraph(
        'Interpretation: Very strong causal evidence (average 83%) linking parameter violations '
        'through aperture overfill mechanism to solder bridging defect. Recommend immediate '
        'reduction of paste volume and stencil thickness, plus environmental controls.'
    )
    
    doc.add_page_break()
    
    # ------------------------------------------------------------------------
    # 5.3 Scenario 3: Nominal Process
    # ------------------------------------------------------------------------
    
    doc.add_heading('5.3 Scenario 3: Nominal Process (No Defects)', 2)
    
    doc.add_heading('Input Parameters:', 3)
    
    scenario3_params = [
        ('Paste Volume', '0.040 mm³', 'Nominal (optimal value)'),
        ('Stencil Thickness', '100 µm', 'Nominal (optimal value)'),
        ('Paste Viscosity', '200 Pa·s', 'Nominal (optimal value)'),
        ('Ambient Humidity', '40%', 'Nominal (optimal value)'),
        ('Ambient Temperature', '23°C', 'Nominal (optimal value)')
    ]
    
    s3_table = doc.add_table(rows=6, cols=3)
    s3_table.style = 'Light Grid Accent 1'
    
    # Header
    s3_header = s3_table.rows[0].cells
    s3_header[0].text = 'Parameter'
    s3_header[1].text = 'Value'
    s3_header[2].text = 'Status'
    
    for cell in s3_header:
        for paragraph in cell.paragraphs:
            for run in paragraph.runs:
                run.font.bold = True
    
    # Data
    for i, (param, value, status) in enumerate(scenario3_params, start=1):
        row = s3_table.rows[i].cells
        row[0].text = param
        row[1].text = value
        row[2].text = status
    
    doc.add_paragraph()
    
    doc.add_heading('Level 3: Violation Analysis', 3)
    
    doc.add_paragraph('All parameters at nominal values.')
    doc.add_paragraph('Overall Violation Score: 0.08 (8% - minimal risk)')
    
    doc.add_paragraph()
    
    doc.add_paragraph(
        'Risk probabilities for all parameters are in the minimal range (5-10%), '
        'indicating stable process operation well within specifications.'
    )
    
    doc.add_paragraph()
    
    doc.add_heading('Level 2: Mechanism Prediction', 3)
    
    doc.add_paragraph('Mechanism: poorpastetransfer / apertureoverfill')
    doc.add_paragraph('MLP Probability: 0.15 (15% - unlikely)')
    doc.add_paragraph('Status: NOT PRESENT')
    
    doc.add_paragraph()
    
    doc.add_paragraph(
        'Explanation: With all parameters at nominal values, the MLP predicts very low '
        'probability of any failure mechanism. The process is operating in the stable region '
        'where paste transfer is optimal.'
    )
    
    doc.add_paragraph()
    
    doc.add_heading('Level 1: Defect Prediction', 3)
    
    doc.add_paragraph('Predicted Defect: No Defect')
    doc.add_paragraph('MLP Confidence: 0.95 (95%)')
    
    doc.add_paragraph()
    
    s3_defect_probs = [
        ('No Defect', '0.95', '95% (PREDICTED)'),
        ('Open Circuit', '0.03', '3%'),
        ('Solder Bridging', '0.02', '2%')
    ]
    
    doc.add_paragraph('Probability Distribution:')
    
    for defect, prob, pct in s3_defect_probs:
        doc.add_paragraph(f'{defect}: {prob} ({pct})', style='List Bullet')
    
    doc.add_paragraph()
    
    doc.add_paragraph(
        'Explanation: With nominal parameters and no mechanism active, the MLP predicts '
        'very high probability of defect-free production. The small residual probabilities '
        'for defects (3% and 2%) represent normal process variation and model uncertainty.'
    )
    
    doc.add_paragraph()
    
    doc.add_heading('Complete Causal Chain:', 3)
    
    doc.add_paragraph(
        'Violations (0.08) → No Mechanism (0.15) → No Defect (0.95)'
    )
    
    doc.add_paragraph()
    
    chain_strength3 = [
        ('Average Path Strength', '(0.08 + 0.15 + 0.95) / 3 = 0.39'),
        ('Note', 'Low average reflects minimal violations, not lack of confidence')
    ]
    
    doc.add_paragraph('Chain Strength Metrics:')
    
    for metric, value in chain_strength3:
        doc.add_paragraph(f'{metric}: {value}', style='List Bullet')
    
    doc.add_paragraph()
    
    doc.add_paragraph(
        'Interpretation: Process operating normally. Low violation and mechanism scores '
        'indicate stable conditions. High defect confidence (95%) for No Defect class '
        'confirms expected outcome. No corrective action required - continue monitoring.'
    )
    
    doc.add_page_break()
    
    # ========================================================================
    # 6. EVIDENCE STRENGTH FOR MODE C INTEGRATION
    # ========================================================================
    
    doc.add_heading('6. Evidence Strength for Mode C Integration', 1)
    
    doc.add_paragraph(
        'Mode C (Causal Graph) integration requires understanding evidence types and '
        'how to weight different sources of information. The system produces two types '
        'of evidence with different reliability characteristics.'
    )
    
    doc.add_heading('Evidence Types:', 2)
    
    evidence_types = [
        ('HARD Evidence (Violations)',
         'Rule-based, deterministic calculations',
         'Source: Mathematical comparison to specification limits',
         'Reliability: 100% accurate (no uncertainty in detection)',
         'Weight: 1.0 (fully trusted)',
         'Use: Ground truth for parameter states'),
        
        ('SOFT Evidence (Mechanisms & Defects)',
         'MLP predictions, probabilistic',
         'Source: Learned patterns from training data',
         'Reliability: 82-98% F1 scores (validated on test set)',
         'Weight: Equal to MLP probability (0.0-1.0)',
         'Use: Predicted outcomes based on learned associations')
    ]
    
    for title, desc, source, reliability, weight, use in evidence_types:
        doc.add_heading(title, 3)
        doc.add_paragraph(f'Description: {desc}')
        doc.add_paragraph(f'{source}')
        doc.add_paragraph(f'{reliability}')
        doc.add_paragraph(f'{weight}')
        doc.add_paragraph(f'{use}')
        doc.add_paragraph()
    
    doc.add_heading('Node Weighting Strategy:', 2)
    
    doc.add_paragraph(
        'Each node in the causal graph should be weighted based on evidence type:'
    )
    
    doc.add_paragraph()
    
    weighting_strategy = [
        ('Parameter Violation Nodes (Violated)',
         'Weight = 1.0',
         'These are hard facts - parameter definitely exceeds limit'),
        
        ('Parameter Warning Nodes (Approaching)',
         'Weight = violation probability (e.g., 0.52)',
         'Probability represents risk, not certainty of violation'),
        
        ('Mechanism Nodes',
         'Weight = MLP probability (e.g., 0.82)',
         'Reflects model confidence in mechanism presence'),
        
        ('Defect Nodes',
         'Weight = MLP probability (e.g., 0.91)',
         'Reflects model confidence in defect prediction')
    ]
    
    for node_type, weight, explanation in weighting_strategy:
        doc.add_paragraph(f'{node_type}:', style='List Bullet').bold = True
        doc.add_paragraph(f'   {weight}')
        doc.add_paragraph(f'   {explanation}')
        doc.add_paragraph()
    
    doc.add_heading('Edge Weighting Strategy:', 2)
    
    doc.add_paragraph(
        'Edges (connections) between nodes represent causal relationships. '
        'Edge weights reflect the strength of influence:'
    )
    
    doc.add_paragraph()
    
    edge_strategy = [
        ('Violations → Mechanism',
         'Weight = average of contributing violation probabilities',
         'Example: (0.85 + 0.52) / 2 = 0.685'),
        
        ('Mechanism → Defect',
         'Weight = mechanism probability',
         'Example: 0.82 (mechanism confidence propagates forward)'),
        
        ('Violations → Defect (direct)',
         'Weight = min(violation avg, defect probability)',
         'Conservative estimate of direct influence')
    ]
    
    for edge_type, weight, example in edge_strategy:
        doc.add_paragraph(f'{edge_type}:', style='List Bullet').bold = True
        doc.add_paragraph(f'   {weight}')
        doc.add_paragraph(f'   {example}')
        doc.add_paragraph()
    
    doc.add_heading('Example Mode C Graph Structure:', 2)
    
    doc.add_paragraph('For the bridging scenario (Scenario 2):')
    
    doc.add_paragraph()
    
    graph_nodes = [
        'Node: paste_volume_high (weight=1.0, type=violation, evidence=HARD)',
        'Node: stencil_thickness_high (weight=1.0, type=violation, evidence=HARD)',
        'Node: paste_viscosity_low (weight=0.58, type=warning, evidence=HARD)',
        'Node: ambient_rh_high (weight=0.72, type=warning, evidence=HARD)',
        'Node: ambient_temperature_high (weight=1.0, type=violation, evidence=HARD)',
        'Node: apertureoverfill (weight=0.86, type=mechanism, evidence=SOFT)',
        'Node: SolderBridging (weight=0.91, type=defect, evidence=SOFT)'
    ]
    
    for node in graph_nodes:
        doc.add_paragraph(node, style='List Bullet')
    
    doc.add_paragraph()
    
    graph_edges = [
        'Edge: paste_volume_high → apertureoverfill (weight=0.85)',
        'Edge: stencil_thickness_high → apertureoverfill (weight=0.78)',
        'Edge: paste_viscosity_low → apertureoverfill (weight=0.58)',
        'Edge: ambient_rh_high → apertureoverfill (weight=0.72)',
        'Edge: ambient_temperature_high → apertureoverfill (weight=0.68)',
        'Edge: apertureoverfill → SolderBridging (weight=0.86)'
    ]
    
    for edge in graph_edges:
        doc.add_paragraph(edge, style='List Bullet')
    
    doc.add_paragraph()
    
    doc.add_paragraph(
        'This structure allows Mode C to perform Bayesian inference, considering '
        'evidence strength at each level and propagating uncertainty through the causal chain.'
    )
    
    doc.add_page_break()
    
    # ========================================================================
    # 7. REFERENCES
    # ========================================================================
    
    doc.add_heading('7. References', 1)
    
    doc.add_paragraph(
        'This system builds on established principles in manufacturing quality control, '
        'machine learning, and causal inference:'
    )
    
    doc.add_paragraph()
    
    references = [
        'Statistical Process Control (SPC) principles for specification limit setting',
        'Multi-task learning architecture for joint prediction of related outcomes',
        'Calibrated probability estimation for reliable confidence scores',
        'Bayesian Networks for causal reasoning under uncertainty',
        'Physics-based process models for PCB assembly and solder paste behavior'
    ]
    
    for ref in references:
        doc.add_paragraph(ref, style='List Bullet')
    
    doc.add_paragraph()
    doc.add_paragraph()
    
    doc.add_paragraph(
        'For technical questions or implementation details, refer to the system '
        'code documentation and training notebooks.'
    )
    
    # ========================================================================
    # SAVE DOCUMENT
    # ========================================================================
    
    output_path = './Causal_Chain_Documentation.docx'
    doc.save(output_path)
    
    return output_path


if __name__ == "__main__":
    print("Generating Causal Chain Documentation...")
    output_file = create_causal_chain_documentation()
    print(f"\n✓ Documentation created: {output_file}")
    print("\nDocument includes:")
    print("  - System overview")
    print("  - Physical understanding of mechanisms")
    print("  - Three-level causal chain architecture")
    print("  - Scoring methodology")
    print("  - Complete worked examples")
    print("  - Mode C integration guidelines")