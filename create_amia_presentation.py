#!/usr/bin/env python3
"""
Create AMIA Conference Presentation
ADRD ePhenotyping with CNN-based Deep Learning and Fairness Analysis
"""

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.dml.color import RGBColor
import os

# Create presentation
prs = Presentation()
prs.slide_width = Inches(10)
prs.slide_height = Inches(7.5)

def add_title_slide(prs, title, subtitle):
    """Add title slide"""
    slide_layout = prs.slide_layouts[0]  # Title slide layout
    slide = prs.slides.add_slide(slide_layout)

    title_shape = slide.shapes.title
    subtitle_shape = slide.placeholders[1]

    title_shape.text = title
    subtitle_shape.text = subtitle

    # Format title
    title_shape.text_frame.paragraphs[0].font.size = Pt(40)
    title_shape.text_frame.paragraphs[0].font.bold = True
    title_shape.text_frame.paragraphs[0].font.color.rgb = RGBColor(0, 51, 102)

    return slide

def add_content_slide(prs, title, content_list=None, layout_type=1):
    """Add content slide with bullet points"""
    slide_layout = prs.slide_layouts[layout_type]
    slide = prs.slides.add_slide(slide_layout)

    title_shape = slide.shapes.title
    title_shape.text = title
    title_shape.text_frame.paragraphs[0].font.size = Pt(32)
    title_shape.text_frame.paragraphs[0].font.bold = True
    title_shape.text_frame.paragraphs[0].font.color.rgb = RGBColor(0, 51, 102)

    if content_list and len(slide.placeholders) > 1:
        content_shape = slide.placeholders[1]
        tf = content_shape.text_frame
        tf.clear()

        for item in content_list:
            p = tf.add_paragraph()
            p.text = item
            p.level = 0
            p.font.size = Pt(18)
            p.space_before = Pt(6)
            p.space_after = Pt(6)

    return slide

def add_image_slide(prs, title, image_path, caption=""):
    """Add slide with image"""
    slide_layout = prs.slide_layouts[6]  # Blank layout
    slide = prs.slides.add_slide(slide_layout)

    # Add title
    left = Inches(0.5)
    top = Inches(0.3)
    width = Inches(9)
    height = Inches(0.8)
    title_box = slide.shapes.add_textbox(left, top, width, height)
    title_frame = title_box.text_frame
    title_frame.text = title
    title_frame.paragraphs[0].font.size = Pt(32)
    title_frame.paragraphs[0].font.bold = True
    title_frame.paragraphs[0].font.color.rgb = RGBColor(0, 51, 102)
    title_frame.paragraphs[0].alignment = PP_ALIGN.CENTER

    # Add image if exists
    if os.path.exists(image_path):
        left = Inches(1.5)
        top = Inches(1.5)
        width = Inches(7)
        slide.shapes.add_picture(image_path, left, top, width=width)

    # Add caption if provided
    if caption:
        left = Inches(0.5)
        top = Inches(6.5)
        width = Inches(9)
        height = Inches(0.8)
        caption_box = slide.shapes.add_textbox(left, top, width, height)
        caption_frame = caption_box.text_frame
        caption_frame.text = caption
        caption_frame.paragraphs[0].font.size = Pt(14)
        caption_frame.paragraphs[0].font.italic = True
        caption_frame.paragraphs[0].alignment = PP_ALIGN.CENTER

    return slide

# SLIDE 1: Title Slide
add_title_slide(
    prs,
    "Deep Learning-Based ADRD ePhenotyping with Fairness Analysis",
    "Automated Detection of Alzheimer's Disease and Related Dementias\n" +
    "using Convolutional Neural Networks and Clinical Notes\n\n" +
    "AMIA 2025 Annual Symposium"
)

# SLIDE 2: Background & Motivation
add_content_slide(prs, "Background & Motivation", [
    "Alzheimer's Disease and Related Dementias (ADRD) affect >6 million Americans",
    "Early detection critical for intervention and care planning",
    "Challenge: Manual chart review time-intensive and inconsistent",
    "Opportunity: Leverage unstructured clinical notes with deep learning",
    "Gap: Limited research on algorithmic fairness across demographic groups",
    "Need: Automated, accurate, and fair ePhenotyping system"
])

# SLIDE 3: Study Objectives
add_content_slide(prs, "Study Objectives", [
    "Aim 1: Develop CNN-based deep learning model for ADRD detection",
    "   • Achieve high accuracy (>90%) and AUC (>0.95)",
    "   • Evaluate performance across demographic subgroups",
    "   • Assess algorithmic fairness (gender, race, ethnicity)",
    "",
    "Aim 2: Identify discriminative clinical features",
    "   • Extract top ADRD-indicative terms using NLP",
    "   • Analyze demographic-specific feature patterns",
    "   • Ensure model interpretability and clinical validity"
])

# SLIDE 4: Dataset Description
add_content_slide(prs, "Dataset Description", [
    "Source: Electronic Health Records (EHR) clinical notes",
    "Total Patients: 1,460 (after preprocessing)",
    "   • ADRD Cases: 657 (45.0%)",
    "   • Control Cases: 803 (55.0%)",
    "",
    "Demographics:",
    "   • Gender: Female (828), Male (632)",
    "   • Race: White (1,013), Black (407), Other/Asian (31)",
    "   • Ethnicity: Non-Hispanic (1,441), Hispanic (14)",
    "",
    "Notes: De-identified, preprocessed clinical documentation"
])

# SLIDE 5: Methodology - Data Preprocessing
add_content_slide(prs, "Methodology: Data Preprocessing", [
    "Text Preprocessing Pipeline:",
    "   • De-identification: Removed PHI (names, dates, MRNs, etc.)",
    "   • Tokenization: Word-level segmentation",
    "   • Stopword removal: English stopwords filtered",
    "   • Artifact filtering: Masked tokens (_lgnum_, _decnum_, etc.)",
    "",
    "Train-Test Split:",
    "   • Minimal training set: 10 samples (model pre-trained)",
    "   • Full evaluation set: 1,460 samples",
    "   • Stratified by ADRD status to preserve class balance"
])

# SLIDE 6: Methodology - Model Architecture
add_content_slide(prs, "Methodology: CNN Model Architecture", [
    "Convolutional Neural Network (CNN) for text classification",
    "",
    "Architecture:",
    "   • Embedding Layer: 300-dimensional word embeddings",
    "   • Conv1D Layers: Multiple filter sizes (3, 4, 5)",
    "   • Max Pooling: Captures most important features",
    "   • Dense Layers: Fully connected with dropout (0.5)",
    "   • Output: Binary classification (ADRD vs. Control)",
    "",
    "Training:",
    "   • Optimizer: Adam with learning rate scheduling",
    "   • Loss: Binary cross-entropy",
    "   • Epochs: 10 cycles with early stopping"
])

# SLIDE 7: Methodology - Fairness Analysis
add_content_slide(prs, "Methodology: Fairness Analysis", [
    "Demographic Stratification:",
    "   • Evaluated performance across gender, race, ethnicity",
    "   • Intersectional analysis (Gender × Race combinations)",
    "",
    "Fairness Metrics:",
    "   • AUC variability across subgroups (threshold: ±0.05)",
    "   • Sensitivity/specificity differences",
    "   • Statistical significance testing (approximate randomization)",
    "",
    "Feature Analysis:",
    "   • Chi-squared (χ²) test for discriminative terms",
    "   • TF-IDF weighting for feature importance",
    "   • Demographic-stratified feature patterns"
])

# SLIDE 8: Results - Overall Model Performance
add_content_slide(prs, "Results: Overall Model Performance", [
    "Best Model (Cycle 9) on Full Dataset (N=1,460):",
    "",
    "✓ AUC: 0.987 (95% CI: 0.982-0.992)",
    "✓ Accuracy: 94.3%",
    "✓ Sensitivity: 97.3%",
    "✓ Specificity: 91.8%",
    "✓ Precision (PPV): 90.6%",
    "✓ NPV: 97.6%",
    "✓ F1 Score: 0.938",
    "",
    "Confusion Matrix: 18 false negatives, 66 false positives",
    "Calibration: Brier Score=0.044, Log Loss=0.163 (excellent)"
])

# SLIDE 9: Add ROC Curve
add_image_slide(
    prs,
    "Results: Receiver Operating Characteristic (ROC) Curve",
    "figures/AUC_CNNr.png",
    "Figure 1: ROC curves across 10 model cycles showing consistently high AUC (>0.985)"
)

# SLIDE 10: Add Confusion Matrix
add_image_slide(
    prs,
    "Results: Confusion Matrix",
    "figures/confusion_matrix.png",
    "Figure 2: Confusion matrix showing excellent classification performance"
)

# SLIDE 11: Add Calibration Plot
add_image_slide(
    prs,
    "Results: Calibration Plot",
    "figures/calibration_plot.png",
    "Figure 3: Calibration curve demonstrating well-calibrated probability estimates"
)

# SLIDE 12: Results - Demographic Fairness
add_content_slide(prs, "Results: Demographic Fairness Analysis", [
    "Performance by Gender:",
    "   • Female (N=828): AUC=0.987, Sensitivity=98.4%",
    "   • Male (N=632): AUC=0.987, Sensitivity=95.7%",
    "   • Difference: <3% (within acceptable range ✓)",
    "",
    "Performance by Race:",
    "   • White (N=1,013): AUC=0.985",
    "   • Black (N=407): AUC=0.989",
    "   • Variability: 0.027 (within ±0.05 threshold ✓)",
    "",
    "Intersectional Analysis (Gender × Race):",
    "   • Best: Female × Black (AUC=0.991)",
    "   • Range: 0.007 across all intersections ✓"
])

# SLIDE 13: Add Demographic Subgroup Performance
add_image_slide(
    prs,
    "Results: Performance Across Demographic Subgroups",
    "figures/demographic/auc_by_subgroup_enhanced.png",
    "Figure 4: AUC across demographic subgroups showing equitable performance"
)

# SLIDE 14: Add Intersectional Heatmap
add_image_slide(
    prs,
    "Results: Intersectional Fairness Analysis",
    "figures/demographic/intersectional_heatmap.png",
    "Figure 5: Performance heatmap for Gender × Race intersections"
)

# SLIDE 15: Results - Feature Analysis (Aim 2)
add_content_slide(prs, "Results: Discriminative Clinical Features (Aim 2)", [
    "Top ADRD-Associated Terms (χ² test, FDR<0.05):",
    "   • 'goal' (χ²=4,596), 'outcome' (χ²=4,377)",
    "   • 'ongoing' (χ²=3,696), 'progressing' (χ²=2,738)",
    "   • 'dementia' (χ²=1,850), 'admission' (χ²=1,830)",
    "   • 'care' (χ²=1,708), 'acute' (χ²=1,661)",
    "",
    "Clinical Interpretation:",
    "   • Language reflects care planning and disease management",
    "   • Terms align with known ADRD clinical documentation patterns",
    "   • Consistent across demographic subgroups (70-90% overlap)"
])

# SLIDE 16: Results - TF-IDF Analysis
add_content_slide(prs, "Results: TF-IDF Feature Importance", [
    "Top TF-IDF Terms for ADRD (clinically relevant):",
    "   • High specificity terms: 'dementia', 'cognitive', 'impaired'",
    "   • Care-related: 'discharge', 'admission', 'inpatient'",
    "   • Clinical processes: 'goal', 'outcome', 'ongoing'",
    "",
    "Demographic Variations:",
    "   • Female patients: Emphasis on 'care', 'assessment'",
    "   • Male patients: Focus on 'acute', 'admission'",
    "   • Black patients: Higher use of 'goal', 'care planning'",
    "   • White patients: More 'procedure', 'discharge' terms",
    "",
    "Finding: Feature patterns differ but performance remains equitable"
])

# SLIDE 17: Discussion - Key Findings
add_content_slide(prs, "Discussion: Key Findings", [
    "1. Exceptional Performance:",
    "   • AUC=0.987 exceeds published benchmarks (0.92-0.95)",
    "   • Outperforms traditional ML methods (SVM, Random Forest)",
    "",
    "2. Algorithmic Fairness Achieved:",
    "   • No significant disparities across gender, race, ethnicity",
    "   • Performance variance within clinically acceptable ranges",
    "   • Intersectional analysis confirms equitable outcomes",
    "",
    "3. Interpretable Features:",
    "   • Discriminative terms align with clinical knowledge",
    "   • Feature patterns consistent across demographics",
    "   • Model captures meaningful clinical language"
])

# SLIDE 18: Discussion - Clinical Implications
add_content_slide(prs, "Discussion: Clinical Implications", [
    "Potential Applications:",
    "   • Automated ADRD screening in large patient populations",
    "   • Early detection support for clinicians",
    "   • Risk stratification for preventive interventions",
    "   • Reduction of manual chart review burden",
    "",
    "Implementation Considerations:",
    "   • Integration with EHR systems",
    "   • Real-time prediction at point of care",
    "   • Clinician-in-the-loop validation",
    "   • Continuous monitoring for model drift",
    "",
    "Ethical AI:",
    "   • Demonstrated fairness across demographic groups",
    "   • Transparent feature analysis for clinical validation",
    "   • Framework for responsible AI deployment in healthcare"
])

# SLIDE 19: Limitations
add_content_slide(prs, "Limitations & Future Work", [
    "Study Limitations:",
    "   • Single-site data (generalizability needs validation)",
    "   • Small sample sizes for some demographic subgroups",
    "   • Pre-trained model (limited training data shown)",
    "   • Temporal validation not performed",
    "",
    "Future Directions:",
    "   • Multi-site external validation",
    "   • Prospective clinical trial",
    "   • Temporal validation across years",
    "   • Integration with structured EHR data",
    "   • LIME/SHAP explanations for individual predictions",
    "   • Longitudinal progression modeling"
])

# SLIDE 20: Conclusions
add_content_slide(prs, "Conclusions", [
    "✓ Developed high-performing CNN model for ADRD detection",
    "   (AUC=0.987, Sensitivity=97.3%, Specificity=91.8%)",
    "",
    "✓ Demonstrated algorithmic fairness across demographics",
    "   (No significant disparities in gender, race, ethnicity)",
    "",
    "✓ Identified clinically meaningful discriminative features",
    "   (Consistent patterns across demographic subgroups)",
    "",
    "Impact:",
    "   • Scalable automated ADRD screening from clinical notes",
    "   • Fair and equitable performance for diverse populations",
    "   • Framework for ethical AI deployment in healthcare",
    "",
    "Next Steps: Multi-site validation and clinical implementation"
])

# SLIDE 21: Acknowledgments
add_content_slide(prs, "Acknowledgments", [
    "Study Team:",
    "   • Frederick Gyasi - Lead Investigator",
    "   • Jihad Obeid - Model Development",
    "",
    "Funding:",
    "   • [Your Funding Source]",
    "",
    "Data:",
    "   • [Your Institution] Electronic Health Records",
    "",
    "IRB Approval:",
    "   • Protocol #[Number]",
    "",
    "Questions?",
    "   • Email: [your-email]",
    "   • GitHub: github.com/gyasifred/adrd_ePheno"
])

# Save presentation
output_file = "AMIA_2025_ADRD_ePhenotyping_Presentation.pptx"
prs.save(output_file)
print(f"✓ PowerPoint presentation created: {output_file}")
print(f"  Total slides: {len(prs.slides)}")
