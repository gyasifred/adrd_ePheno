#!/usr/bin/env python3
"""
ADRD ePhenotyping Paper Structure and Code Review PDF Generator
Using ReportLab for reliable PDF generation
"""

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, ListFlowable, ListItem
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from datetime import datetime

def create_pdf():
    output_path = "/home/user/adrd_ePheno/ADRD_ePhenotyping_Paper_Code_Review.pdf"
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        rightMargin=0.75*inch,
        leftMargin=0.75*inch,
        topMargin=0.75*inch,
        bottomMargin=0.75*inch
    )

    # Colors
    navy = HexColor('#003366')
    dark_blue = HexColor('#333366')
    dark_gray = HexColor('#333333')
    light_gray = HexColor('#F5F5F5')

    # Styles
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=navy,
        alignment=TA_CENTER,
        spaceAfter=20,
        spaceBefore=30
    )

    subtitle_style = ParagraphStyle(
        'CustomSubtitle',
        parent=styles['Normal'],
        fontSize=14,
        textColor=dark_gray,
        alignment=TA_CENTER,
        spaceAfter=10
    )

    h1_style = ParagraphStyle(
        'H1',
        parent=styles['Heading1'],
        fontSize=16,
        textColor=navy,
        spaceBefore=20,
        spaceAfter=10
    )

    h2_style = ParagraphStyle(
        'H2',
        parent=styles['Heading2'],
        fontSize=13,
        textColor=dark_blue,
        spaceBefore=15,
        spaceAfter=8
    )

    h3_style = ParagraphStyle(
        'H3',
        parent=styles['Heading3'],
        fontSize=11,
        textColor=dark_gray,
        spaceBefore=10,
        spaceAfter=6
    )

    body_style = ParagraphStyle(
        'Body',
        parent=styles['Normal'],
        fontSize=10,
        alignment=TA_JUSTIFY,
        spaceAfter=8,
        leading=14
    )

    bullet_style = ParagraphStyle(
        'Bullet',
        parent=styles['Normal'],
        fontSize=10,
        leftIndent=20,
        spaceAfter=4,
        leading=13
    )

    code_style = ParagraphStyle(
        'Code',
        parent=styles['Normal'],
        fontSize=9,
        fontName='Courier',
        leftIndent=10,
        backColor=light_gray,
        spaceAfter=10,
        leading=12
    )

    # Build content
    content = []

    # Title Page
    content.append(Spacer(1, 2*inch))
    content.append(Paragraph("ADRD ePhenotyping", title_style))
    content.append(Paragraph("Paper Structure & Pipeline Code Review", title_style))
    content.append(Spacer(1, 0.5*inch))
    content.append(Paragraph("Deep Learning-Based ADRD Detection with<br/>Comprehensive Fairness Analysis", subtitle_style))
    content.append(Spacer(1, 1*inch))
    content.append(Paragraph(f"Generated: {datetime.now().strftime('%B %d, %Y')}", subtitle_style))
    content.append(Paragraph("Authors: Frederick Gyasi, Jihad Obeid, Paul Heider", subtitle_style))
    content.append(Paragraph("Medical University of South Carolina", subtitle_style))
    content.append(PageBreak())

    # PART 1: Paper Structure
    content.append(Paragraph("PART 1: WHAT A PAPER WOULD LOOK LIKE", h1_style))
    content.append(Paragraph(
        "This section outlines the structure and content of a research paper based on the ADRD ePhenotyping work. "
        "The paper would be suitable for submission to AMIA, JAMIA, or similar health informatics venues.",
        body_style
    ))

    # Abstract
    content.append(Paragraph("1. Abstract (~250 words)", h2_style))
    content.append(Paragraph(
        "<b>OBJECTIVE:</b> Evaluate the performance and algorithmic fairness of a pre-trained convolutional neural network (CNN) "
        "for automated detection of Alzheimer's Disease and Related Dementias (ADRD) from unstructured clinical notes in electronic health records.",
        body_style
    ))
    content.append(Paragraph(
        "<b>METHODS:</b> We evaluated Knox and Obeid's pre-trained CNN model on 1,460 patients (657 ADRD, 803 controls) from MUSC EHR. "
        "Aim 1 assessed demographic fairness using approximate randomization testing (10,000 permutations) across gender, race, and ethnicity. "
        "Aim 2 identified discriminative features using chi-squared testing and TF-IDF analysis.",
        body_style
    ))
    content.append(Paragraph(
        "<b>RESULTS:</b> The CNN achieved AUC=0.9867 (95% CI: 0.9818-0.9916), 97.26% sensitivity, and 91.78% specificity. "
        "No statistically significant performance disparities were detected across demographic subgroups (all p>0.05). "
        "Feature analysis revealed 70-90% overlap in discriminative terms across demographics.",
        body_style
    ))
    content.append(Paragraph(
        "<b>CONCLUSIONS:</b> The CNN demonstrates excellent classification performance with algorithmic fairness across demographic groups, "
        "supporting its potential for equitable ADRD screening in clinical practice.",
        body_style
    ))

    # Introduction
    content.append(Paragraph("2. Introduction", h2_style))
    content.append(Paragraph("2.1 Background", h3_style))
    content.append(Paragraph("• ADRD affects 6+ million Americans; early detection is critical", bullet_style))
    content.append(Paragraph("• EHR clinical notes contain rich phenotypic information", bullet_style))
    content.append(Paragraph("• Deep learning enables automated analysis of unstructured text", bullet_style))
    content.append(Paragraph("• Algorithmic fairness is essential for health equity", bullet_style))

    content.append(Paragraph("2.2 Research Questions", h3_style))
    content.append(Paragraph("• RQ1: Does the CNN model perform equitably across demographic groups?", bullet_style))
    content.append(Paragraph("• RQ2: Do discriminative features differ across demographics?", bullet_style))

    content.append(Paragraph("2.3 Contributions", h3_style))
    content.append(Paragraph("• First comprehensive fairness evaluation of CNN for ADRD ePhenotyping", bullet_style))
    content.append(Paragraph("• Three-dimensional bias framework: Clinical, Algorithmic, Linguistic", bullet_style))
    content.append(Paragraph("• Rigorous statistical testing with approximate randomization", bullet_style))

    content.append(PageBreak())

    # Methods
    content.append(Paragraph("3. Methods", h2_style))

    content.append(Paragraph("3.1 Study Design & Data", h3_style))
    content.append(Paragraph(
        "Retrospective evaluation study using MUSC EHR data. Cohort: 1,460 patients (657 ADRD cases identified via ICD-10 codes G30.x, F01-F03.x, G31.x; "
        "803 matched controls). Demographics: 56.7% female, 69.4% White, 27.9% Black.",
        body_style
    ))

    content.append(Paragraph("3.2 Pre-trained CNN Model", h3_style))
    content.append(Paragraph(
        "Knox and Obeid's CNN architecture: 40,000-term vocabulary, 200-dimension embeddings, convolutional filters (3,4,5 kernels), "
        "max pooling, dense layers. Ten-fold cross-validation with early stopping. Best model selected via median AUC + max F1 criteria.",
        body_style
    ))

    content.append(Paragraph("3.3 Aim 1: Demographic Fairness Analysis", h3_style))
    content.append(Paragraph("• Stratification by: Gender, Race, Ethnicity, Intersectional (Gender x Race)", bullet_style))
    content.append(Paragraph("• Metrics: AUC, Accuracy, Sensitivity, Specificity, F1, NPV, PPV", bullet_style))
    content.append(Paragraph("• Statistical testing: Approximate randomization (10,000 permutations)", bullet_style))
    content.append(Paragraph("• Multiple testing correction: Benjamini-Hochberg FDR (alpha=0.05)", bullet_style))
    content.append(Paragraph("• Effect sizes: Cohen's d for clinical significance", bullet_style))

    content.append(Paragraph("3.4 Aim 2: Feature Analysis", h3_style))
    content.append(Paragraph("• Chi-squared keyness test: 13,889 terms tested", bullet_style))
    content.append(Paragraph("• TF-IDF analysis: Identify distinctive ADRD terminology", bullet_style))
    content.append(Paragraph("• Demographic-stratified analysis: Feature consistency across groups", bullet_style))
    content.append(Paragraph("• LIME explainability: Local model interpretations", bullet_style))

    content.append(PageBreak())

    # Results
    content.append(Paragraph("4. Results", h2_style))

    content.append(Paragraph("4.1 Overall Model Performance", h3_style))
    content.append(Paragraph("Table 1: Classification Performance Metrics (N=1,460)", body_style))

    # Performance table
    perf_data = [
        ['Metric', 'Value', '95% CI', 'Interpretation'],
        ['AUC', '0.9867', '0.9818-0.9916', 'Excellent discrimination'],
        ['Accuracy', '94.25%', '92.9%-95.4%', 'High overall accuracy'],
        ['Sensitivity', '97.26%', '95.8%-98.4%', 'Few missed ADRD cases'],
        ['Specificity', '91.78%', '89.7%-93.6%', 'Good true negative rate'],
        ['Precision', '90.64%', '88.4%-92.7%', 'High positive predictive value'],
        ['F1 Score', '0.9383', '0.926-0.949', 'Balanced precision/recall'],
        ['Brier Score', '0.0440', '-', 'Well-calibrated probabilities'],
    ]
    perf_table = Table(perf_data, colWidths=[1.2*inch, 0.9*inch, 1.2*inch, 2.2*inch])
    perf_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), navy),
        ('TEXTCOLOR', (0, 0), (-1, 0), HexColor('#FFFFFF')),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 9),
        ('FONTSIZE', (0, 1), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#CCCCCC')),
    ]))
    content.append(perf_table)
    content.append(Spacer(1, 0.2*inch))

    content.append(Paragraph("4.2 Aim 1: Demographic Fairness Results", h3_style))
    content.append(Paragraph("Table 2: Performance by Demographic Subgroup", body_style))

    # Demographics table
    demo_data = [
        ['Subgroup', 'N', 'AUC', 'Sensitivity', 'Specificity', 'p-value'],
        ['Female', '828', '0.9867', '98.40%', '90.71%', '-'],
        ['Male', '632', '0.9875', '95.73%', '93.16%', '0.432'],
        ['White', '1,013', '0.9855', '97.04%', '91.43%', '-'],
        ['Black', '407', '0.9893', '97.83%', '93.22%', '0.089'],
        ['Non-Hispanic', '1,441', '0.9864', '97.23%', '91.66%', '-'],
    ]
    demo_table = Table(demo_data, colWidths=[1*inch, 0.6*inch, 0.7*inch, 0.9*inch, 0.9*inch, 0.7*inch])
    demo_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), navy),
        ('TEXTCOLOR', (0, 0), (-1, 0), HexColor('#FFFFFF')),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 9),
        ('FONTSIZE', (0, 1), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#CCCCCC')),
    ]))
    content.append(demo_table)
    content.append(Spacer(1, 0.1*inch))

    content.append(Paragraph(
        "<b>KEY FINDING:</b> No statistically significant performance disparities across any demographic subgroup (all p>0.05). "
        "AUC variance &lt;0.01 across groups.",
        body_style
    ))

    content.append(Paragraph("4.3 Aim 2: Feature Analysis Results", h3_style))
    content.append(Paragraph("Table 3: Top Discriminative Terms for ADRD", body_style))

    # Features table
    feat_data = [
        ['Rank', 'Term', 'Chi-sq', 'p-value', 'Clinical Category'],
        ['1', 'goal', '4,623', '<0.001', 'Care planning'],
        ['2', 'outcome', '4,401', '<0.001', 'Care planning'],
        ['3', 'ongoing', '3,717', '<0.001', 'Disease management'],
        ['4', 'progressing', '2,753', '<0.001', 'Disease progression'],
        ['5', 'dementia', '1,858', '<0.001', 'Diagnosis'],
    ]
    feat_table = Table(feat_data, colWidths=[0.5*inch, 0.9*inch, 0.7*inch, 0.7*inch, 1.5*inch])
    feat_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), navy),
        ('TEXTCOLOR', (0, 0), (-1, 0), HexColor('#FFFFFF')),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 9),
        ('FONTSIZE', (0, 1), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#CCCCCC')),
    ]))
    content.append(feat_table)
    content.append(Spacer(1, 0.1*inch))

    content.append(Paragraph(
        "Feature consistency: 90% term overlap (Female vs Male), 70% term overlap (Black vs White) - indicating robust, generalizable features.",
        body_style
    ))

    content.append(PageBreak())

    # Discussion
    content.append(Paragraph("5. Discussion", h2_style))

    content.append(Paragraph("5.1 Principal Findings", h3_style))
    content.append(Paragraph("• CNN achieves state-of-the-art performance (AUC=0.987)", bullet_style))
    content.append(Paragraph("• No statistically significant demographic bias detected", bullet_style))
    content.append(Paragraph("• Discriminative features align with clinical ADRD phenotypes", bullet_style))
    content.append(Paragraph("• 70-90% feature consistency across demographics", bullet_style))

    content.append(Paragraph("5.2 Clinical Implications", h3_style))
    content.append(Paragraph("• 97.3% sensitivity supports population-level screening", bullet_style))
    content.append(Paragraph("• Equitable performance enables fair deployment across demographics", bullet_style))
    content.append(Paragraph("• Interpretable features support clinical validation", bullet_style))

    content.append(Paragraph("5.3 Limitations", h3_style))
    content.append(Paragraph("• Single-site data (MUSC) - external validation needed", bullet_style))
    content.append(Paragraph("• Small subgroups for Hispanic (n=14), Asian (n=10)", bullet_style))
    content.append(Paragraph("• Retrospective design with ICD-based labels", bullet_style))
    content.append(Paragraph("• Temporal validation not performed", bullet_style))

    content.append(Paragraph("5.4 Future Directions", h3_style))
    content.append(Paragraph("• Multi-site external validation", bullet_style))
    content.append(Paragraph("• Prospective clinical trial with real-time alerts", bullet_style))
    content.append(Paragraph("• Longitudinal progression modeling", bullet_style))
    content.append(Paragraph("• Integration with structured EHR data", bullet_style))

    content.append(PageBreak())

    # PART 2: Code Review
    content.append(Paragraph("PART 2: CODE REVIEW OF THE PIPELINE", h1_style))
    content.append(Paragraph(
        "This section provides a comprehensive code review of the ADRD ePhenotyping analysis pipeline, "
        "examining each script's purpose, methodology, and quality.",
        body_style
    ))

    # Pipeline Overview
    content.append(Paragraph("1. Pipeline Architecture Overview", h2_style))
    content.append(Paragraph(
        "The pipeline consists of 6 main R scripts executed sequentially, with utility scripts for shared functions.",
        body_style
    ))

    content.append(Paragraph(
        "<font name='Courier' size='9'>"
        "EXECUTION ORDER:<br/>"
        "&nbsp;&nbsp;01_prepare_data.R&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-> Data preparation (optional)<br/>"
        "&nbsp;&nbsp;02_train_cnnr.R&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-> Training reference (NOT USED)<br/>"
        "&nbsp;&nbsp;03_evaluate_models.R&nbsp;&nbsp;&nbsp;&nbsp;-> Model evaluation (START HERE)<br/>"
        "&nbsp;&nbsp;04_demographic_analysis.R -> Aim 1: Fairness analysis<br/>"
        "&nbsp;&nbsp;05_aim2_feature_analysis.R -> Aim 2: Feature analysis<br/>"
        "&nbsp;&nbsp;06_integration_analysis.R&nbsp;-> Integration: Aim 1 + Aim 2"
        "</font>",
        body_style
    ))

    content.append(PageBreak())

    # Script 03 Review
    content.append(Paragraph("2. 03_evaluate_models.R - Model Evaluation", h2_style))

    content.append(Paragraph("2.1 Purpose", h3_style))
    content.append(Paragraph(
        "Evaluates Knox/Obeid's 10 pre-trained CNN models on the test set, calculates comprehensive metrics, "
        "selects best model, and generates visualizations.",
        body_style
    ))

    content.append(Paragraph("2.2 Key Functions", h3_style))
    content.append(Paragraph(
        "<font name='Courier' size='9'>"
        "calculate_comprehensive_metrics(y_true, y_pred_prob, threshold)<br/>"
        "&nbsp;&nbsp;- Confusion matrix calculation (TP, TN, FP, FN)<br/>"
        "&nbsp;&nbsp;- Classification metrics: accuracy, sens, spec, precision, NPV, F1, F2<br/>"
        "&nbsp;&nbsp;- ROC/AUC with DeLong confidence intervals<br/>"
        "&nbsp;&nbsp;- Calibration metrics: Brier score, log loss<br/>"
        "&nbsp;&nbsp;- Optimal threshold via Youden's Index"
        "</font>",
        body_style
    ))

    content.append(Paragraph("2.3 Model Selection Criteria", h3_style))
    content.append(Paragraph(
        "Best model selected using median AUC + maximum F1 criteria (following Jihad Obeid's methodology). "
        "This approach balances discrimination ability with precision-recall trade-off.",
        body_style
    ))

    content.append(Paragraph("2.4 Code Quality Assessment", h3_style))
    content.append(Paragraph("• STRENGTH: Well-documented, modular functions, comprehensive metrics", bullet_style))
    content.append(Paragraph("• STRENGTH: Auto-detects model naming conventions (CL07_* or current)", bullet_style))
    content.append(Paragraph("• STRENGTH: Produces publication-ready visualizations", bullet_style))
    content.append(Paragraph("• CONSIDERATION: Heavy memory usage when loading all 10 models", bullet_style))

    content.append(Paragraph("2.5 Outputs Generated", h3_style))
    content.append(Paragraph("• results/predictions_df.csv - Main predictions file", bullet_style))
    content.append(Paragraph("• results/evaluation_summary.csv - All metrics by cycle", bullet_style))
    content.append(Paragraph("• figures/AUC_CNNr.png - ROC curves for all models", bullet_style))
    content.append(Paragraph("• figures/calibration_plot.png - Probability calibration", bullet_style))
    content.append(Paragraph("• figures/confusion_matrix.png - Best model confusion matrix", bullet_style))

    content.append(PageBreak())

    # Script 04 Review
    content.append(Paragraph("3. 04_demographic_analysis.R - Aim 1 Fairness", h2_style))

    content.append(Paragraph("3.1 Purpose", h3_style))
    content.append(Paragraph(
        "Evaluates CNN model performance across demographic subgroups to assess algorithmic fairness. "
        "Tests statistical significance of performance differences using approximate randomization.",
        body_style
    ))

    content.append(Paragraph("3.2 Statistical Methods", h3_style))
    content.append(Paragraph(
        "<font name='Courier' size='9'>"
        "PERMUTATION TESTING (Approximate Randomization):<br/>"
        "&nbsp;&nbsp;1. Calculate observed AUC difference between groups<br/>"
        "&nbsp;&nbsp;2. Shuffle demographic labels 10,000 times<br/>"
        "&nbsp;&nbsp;3. Recalculate AUC difference for each permutation<br/>"
        "&nbsp;&nbsp;4. p-value = proportion of permuted diffs >= observed<br/>"
        "&nbsp;&nbsp;5. Apply Benjamini-Hochberg FDR correction<br/><br/>"
        "BOOTSTRAP CONFIDENCE INTERVALS:<br/>"
        "&nbsp;&nbsp;- 10,000 bootstrap samples per subgroup<br/>"
        "&nbsp;&nbsp;- 95% CIs for all metrics"
        "</font>",
        body_style
    ))

    content.append(Paragraph("3.3 Fairness Criteria Evaluated", h3_style))
    content.append(Paragraph("• Equalized Odds: TPR/FPR differences &lt;5% across groups", bullet_style))
    content.append(Paragraph("• Demographic Parity: Positive prediction rate variance &lt;10%", bullet_style))
    content.append(Paragraph("• AUC Parity: AUC variance &lt;0.05 across groups", bullet_style))

    content.append(Paragraph("3.4 Code Quality Assessment", h3_style))
    content.append(Paragraph("• STRENGTH: Comprehensive statistical testing framework", bullet_style))
    content.append(Paragraph("• STRENGTH: Handles missing demographic data gracefully", bullet_style))
    content.append(Paragraph("• STRENGTH: Intersectional analysis (Gender x Race)", bullet_style))
    content.append(Paragraph("• STRENGTH: Visualizes null distributions for each test", bullet_style))
    content.append(Paragraph("• CONSIDERATION: Computationally intensive (10,000 permutations)", bullet_style))

    content.append(PageBreak())

    # Script 05 Review
    content.append(Paragraph("4. 05_aim2_feature_analysis.R - Aim 2 Features", h2_style))

    content.append(Paragraph("4.1 Purpose", h3_style))
    content.append(Paragraph(
        "Identifies discriminative linguistic features using statistical methods and assesses "
        "feature consistency across demographic subgroups.",
        body_style
    ))

    content.append(Paragraph("4.2 Text Analysis Pipeline", h3_style))
    content.append(Paragraph(
        "<font name='Courier' size='9'>"
        "QUANTEDA WORKFLOW:<br/>"
        "&nbsp;&nbsp;1. Create corpus from clinical notes<br/>"
        "&nbsp;&nbsp;2. Tokenize (remove punct, symbols, URLs)<br/>"
        "&nbsp;&nbsp;3. Remove stopwords and de-identification artifacts<br/>"
        "&nbsp;&nbsp;4. Create document-feature matrix (DFM)<br/>"
        "&nbsp;&nbsp;5. Trim rare features (min_termfreq = 10)<br/><br/>"
        "STATISTICAL TESTS:<br/>"
        "&nbsp;&nbsp;- Chi-squared keyness test (textstat_keyness)<br/>"
        "&nbsp;&nbsp;- TF-IDF weighting (textstat_tfidf)<br/>"
        "&nbsp;&nbsp;- Benjamini-Hochberg FDR correction"
        "</font>",
        body_style
    ))

    content.append(Paragraph("4.3 Key Analyses", h3_style))
    content.append(Paragraph("• Chi-squared: Tests 13,889 terms for ADRD vs Control association", bullet_style))
    content.append(Paragraph("• TF-IDF: Ranks terms by discriminative importance", bullet_style))
    content.append(Paragraph("• Demographic stratification: Compares features across subgroups", bullet_style))
    content.append(Paragraph("• Feature overlap: Measures consistency (90% gender, 70% race)", bullet_style))

    content.append(Paragraph("4.4 Code Quality Assessment", h3_style))
    content.append(Paragraph("• STRENGTH: Uses established quanteda library for NLP", bullet_style))
    content.append(Paragraph("• STRENGTH: Proper handling of de-identification artifacts", bullet_style))
    content.append(Paragraph("• STRENGTH: Demographic-stratified TF-IDF analysis", bullet_style))
    content.append(Paragraph("• STRENGTH: Clinical interpretation of top terms", bullet_style))
    content.append(Paragraph("• CONSIDERATION: LIME analysis requires model loading overhead", bullet_style))

    content.append(PageBreak())

    # Script 06 Review
    content.append(Paragraph("5. 06_integration_analysis.R - Integration", h2_style))

    content.append(Paragraph("5.1 Purpose", h3_style))
    content.append(Paragraph(
        "Integrates Aim 1 (demographic fairness) with Aim 2 (feature analysis) to provide "
        "comprehensive three-dimensional bias characterization.",
        body_style
    ))

    content.append(Paragraph("5.2 Three-Dimensional Bias Framework", h3_style))
    content.append(Paragraph(
        "<font name='Courier' size='9'>"
        "DIMENSION 1: CLINICAL BIAS<br/>"
        "&nbsp;&nbsp;- How ADRD presentation differs by demographics<br/>"
        "&nbsp;&nbsp;- Analyzed via demographic-stratified features<br/><br/>"
        "DIMENSION 2: ALGORITHMIC BIAS<br/>"
        "&nbsp;&nbsp;- WHERE bias exists in CNN performance<br/>"
        "&nbsp;&nbsp;- Analyzed via Aim 1 performance gaps<br/><br/>"
        "DIMENSION 3: LINGUISTIC BIAS<br/>"
        "&nbsp;&nbsp;- WHY bias exists (feature salience disparities)<br/>"
        "&nbsp;&nbsp;- Analyzed via Aim 2 term overlap"
        "</font>",
        body_style
    ))

    content.append(Paragraph("5.3 Code Quality Assessment", h3_style))
    content.append(Paragraph("• STRENGTH: Novel bias characterization framework", bullet_style))
    content.append(Paragraph("• STRENGTH: Creates integrated dashboard visualization", bullet_style))
    content.append(Paragraph("• STRENGTH: Exports actionable summary tables", bullet_style))
    content.append(Paragraph("• CONSIDERATION: Depends on Aim 1 and Aim 2 outputs", bullet_style))

    # Utility Scripts
    content.append(Paragraph("6. Utility Scripts Review", h2_style))

    content.append(Paragraph("6.1 utils_statistical_tests.R (~800 lines)", h3_style))
    content.append(Paragraph(
        "<font name='Courier' size='9'>"
        "KEY FUNCTIONS:<br/>"
        "&nbsp;&nbsp;permutation_test_auc()&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- Approximate randomization for AUC<br/>"
        "&nbsp;&nbsp;compare_all_metrics_comprehensive() - Test all 8 metrics<br/>"
        "&nbsp;&nbsp;bootstrap_ci_auc()&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- Bootstrap confidence intervals<br/>"
        "&nbsp;&nbsp;calculate_cohens_d()&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- Effect size calculation"
        "</font>",
        body_style
    ))
    content.append(Paragraph("• ASSESSMENT: Well-designed, reusable statistical framework", bullet_style))

    content.append(Paragraph("6.2 utils_model_loader.R (~280 lines)", h3_style))
    content.append(Paragraph(
        "<font name='Courier' size='9'>"
        "KEY FUNCTIONS:<br/>"
        "&nbsp;&nbsp;find_artifact()&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- Detect naming conventions (CL07_* or current)<br/>"
        "&nbsp;&nbsp;load_tokenizer_auto() - Load Keras tokenizer<br/>"
        "&nbsp;&nbsp;load_all_artifacts()&nbsp;- Load all required artifacts<br/>"
        "&nbsp;&nbsp;find_all_models()&nbsp;&nbsp;&nbsp;&nbsp;- Scan for available models"
        "</font>",
        body_style
    ))
    content.append(Paragraph("• ASSESSMENT: Enables backward compatibility with Jihad's artifacts", bullet_style))

    content.append(PageBreak())

    # Overall Assessment
    content.append(Paragraph("7. Overall Code Quality Assessment", h2_style))

    content.append(Paragraph("7.1 Strengths", h3_style))
    content.append(Paragraph("• Well-documented with clear comments and section headers", bullet_style))
    content.append(Paragraph("• Modular design with reusable utility functions", bullet_style))
    content.append(Paragraph("• Comprehensive error handling and validation", bullet_style))
    content.append(Paragraph("• Follows consistent coding style throughout", bullet_style))
    content.append(Paragraph("• Produces publication-ready outputs (tables, figures)", bullet_style))
    content.append(Paragraph("• Rigorous statistical methodology (permutation tests, FDR)", bullet_style))
    content.append(Paragraph("• Backward compatible with original Knox/Obeid artifacts", bullet_style))

    content.append(Paragraph("7.2 Technical Considerations", h3_style))
    content.append(Paragraph("• Memory: Loading 10 CNN models requires significant RAM", bullet_style))
    content.append(Paragraph("• Compute: 10,000 permutations per metric is time-intensive", bullet_style))
    content.append(Paragraph("• Dependencies: Requires TensorFlow, Keras, quanteda packages", bullet_style))
    content.append(Paragraph("• Data: Assumes specific column naming conventions", bullet_style))

    content.append(Paragraph("7.3 Reproducibility", h3_style))
    content.append(Paragraph("• Environment: Conda setup script provided (setup_environment.sh)", bullet_style))
    content.append(Paragraph("• Seeds: Random seeds set for reproducible results", bullet_style))
    content.append(Paragraph("• Outputs: All results saved to CSV/RDS for verification", bullet_style))
    content.append(Paragraph("• Documentation: Comprehensive README and method guides", bullet_style))

    content.append(PageBreak())

    # Summary
    content.append(Paragraph("8. Summary", h2_style))

    content.append(Paragraph("8.1 Paper Readiness", h3_style))
    content.append(Paragraph(
        "The repository contains all necessary components for a high-quality research publication: "
        "rigorous methodology, comprehensive results, statistical validation, and interpretable findings. "
        "The existing AMIA_Paper_Methodology_Results_Discussion.md provides ~12,000 words of publication-ready content.",
        body_style
    ))

    content.append(Paragraph("8.2 Pipeline Quality", h3_style))
    content.append(Paragraph(
        "The code demonstrates professional software engineering practices with clear documentation, "
        "modular design, and comprehensive testing. The pipeline is production-ready for research use, "
        "with appropriate caveats about computational requirements.",
        body_style
    ))

    content.append(Paragraph("8.3 Key Findings", h3_style))
    content.append(Paragraph("• CNN achieves excellent performance (AUC=0.987)", bullet_style))
    content.append(Paragraph("• No significant demographic bias detected", bullet_style))
    content.append(Paragraph("• Features are clinically interpretable and consistent", bullet_style))
    content.append(Paragraph("• Three-dimensional bias framework provides comprehensive analysis", bullet_style))

    content.append(Paragraph("8.4 Recommended Next Steps", h3_style))
    content.append(Paragraph("• External validation at additional sites", bullet_style))
    content.append(Paragraph("• Prospective clinical deployment study", bullet_style))
    content.append(Paragraph("• Expansion of Hispanic/Asian subgroup samples", bullet_style))
    content.append(Paragraph("• Integration into clinical decision support workflow", bullet_style))

    # Build PDF
    doc.build(content)
    return output_path

if __name__ == "__main__":
    output_file = create_pdf()
    print(f"PDF generated successfully: {output_file}")
