from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
                                Image, KeepTogether, PageBreak)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.fonts import addMapping
from PIL import Image as PILImage
from xml.sax.saxutils import escape as esc

D = "/usr/share/fonts/truetype/dejavu/"
pdfmetrics.registerFont(TTFont("DV", D + "DejaVuSans.ttf"))
pdfmetrics.registerFont(TTFont("DV-B", D + "DejaVuSans-Bold.ttf"))
pdfmetrics.registerFont(TTFont("DV-I", D + "DejaVuSans-Oblique.ttf"))
pdfmetrics.registerFont(TTFont("DV-BI", D + "DejaVuSans-BoldOblique.ttf"))
pdfmetrics.registerFont(TTFont("DVM", D + "DejaVuSansMono.ttf"))
addMapping("DV", 0, 0, "DV"); addMapping("DV", 1, 0, "DV-B")
addMapping("DV", 0, 1, "DV-I"); addMapping("DV", 1, 1, "DV-BI")

NAVY = colors.HexColor("#1f2d4d")
GREY = colors.HexColor("#555555")
RED_BG = colors.HexColor("#fbeeee"); RED_BAR = colors.HexColor("#c0392b")
GRN_BG = colors.HexColor("#eef7ef"); GRN_BAR = colors.HexColor("#2e8b57")
BLU_BG = colors.HexColor("#eef2fa"); BLU_BAR = colors.HexColor("#3b5ba9")

body = ParagraphStyle("body", fontName="DV", fontSize=8.8, leading=12.6, spaceAfter=4)
small = ParagraphStyle("small", parent=body, fontSize=7.8, leading=10.8, textColor=GREY)
box = ParagraphStyle("box", parent=body, fontSize=8.6, leading=12.2, spaceAfter=3)
h1 = ParagraphStyle("h1", fontName="DV-B", fontSize=13, leading=17, textColor=NAVY, spaceBefore=10, spaceAfter=6)
h2 = ParagraphStyle("h2", fontName="DV-B", fontSize=10.5, leading=14, textColor=NAVY, spaceBefore=12, spaceAfter=3)
title = ParagraphStyle("title", fontName="DV-B", fontSize=17, leading=22, textColor=NAVY, spaceAfter=4)
lab = ParagraphStyle("lab", fontName="DV-B", fontSize=7.4, leading=10, spaceAfter=2)
cell = ParagraphStyle("cell", fontName="DV", fontSize=7.9, leading=10.8)
cellb = ParagraphStyle("cellb", parent=cell, fontName="DV-B")

W = letter[0] - 1.5 * inch


def P(t, s=body):
    return Paragraph(t, s)


def eq(name, scale=0.78):
    path = f"/home/claude/eq/{name}.png"
    w, h = PILImage.open(path).size
    wp, hp = w * 72 / 300 * scale, h * 72 / 300 * scale
    maxw = W - 0.7 * inch
    if wp > maxw:
        hp *= maxw / wp; wp = maxw
    im = Image(path, width=wp, height=hp)
    im.hAlign = "CENTER"
    return im


def card(label, items, bg, bar, barlabel):
    """items: list of flowables (Paragraphs / Images)."""
    inner = [Paragraph(f'<font color="{bar.hexval().replace("0x","#")}">{barlabel}</font>', lab)] + items
    t = Table([[inner]], colWidths=[W])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), bg),
        ("LINEBEFORE", (0, 0), (0, -1), 2.5, bar),
        ("LEFTPADDING", (0, 0), (-1, -1), 9), ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return t


def current(*paras):
    return card("CURRENTLY READS", [P(p, box) for p in paras], RED_BG, RED_BAR, "CURRENTLY READS (find this)")


def replace(*items):
    its = [P(i, box) if isinstance(i, str) else i for i in items]
    return card("REPLACE", its, GRN_BG, GRN_BAR, "REPLACE WITH (paste this)")


def note(t):
    return card("NOTE", [P(t, box)], BLU_BG, BLU_BAR, "NOTE")


def fix(num, name, where):
    return [P(f"{num}. {esc(name)}", h2), P(esc(where), small)]


def gap(h=5):
    return Spacer(1, h)


S = []
S.append(P("DeepSentinel — Required Fixes to Chapters 1–3", title))
S.append(P("Revision of Final Manuscript (Chapters 1–3). Page numbers are the <b>printed</b> page numbers "
           "(printed p. 106 = PDF p. 110). Each fix gives the text to find and the text to paste. "
           "Quotes were taken from the manuscript PDF.", small))
S.append(gap(4))
S.append(note("Part A is the 10-item fix list. Part B is five more problems I verified in the manuscript that also "
              "need fixing. Replacement wording is mine. Fix 7 and Fix B2 assume the group adopts the existing "
              "speaker-disjoint 10% test split (printed p. 69) for the emotion and sarcasm heads, and Fix 10 gives "
              "reference details from memory, so check them against the publisher pages before submitting."))

# ---------------- PART A ----------------
S.append(P("Part A — The 10 fixes", h1))

# 1
S += fix(1, "Accuracy formula", "Data Analysis → Statistical Treatment → Accuracy (ACC), p. 107")
S.append(current("ACC is used as the primary metric for evaluating the performance of the speech-text and visual "
                 "encoders on the emotion classification task in Research Questions 1 and 2. It is defined as: "
                 "<b>ACC = (TP + TN) / N</b>, where N is the total number of samples; TP is the number of true "
                 "positives; TN is the number of true negatives."))
S.append(gap())
S.append(replace(
    "ACC is the primary metric for evaluating the audio-text emotion head (Emotion Head A) and the visual emotion "
    "head (Emotion Head B) in Research Questions 1 and 2. Both heads classify six emotion classes, so ACC is the "
    "proportion of clips whose predicted emotion matches the ground-truth emotion:",
    eq("acc_multi"),
    "where N is the number of clips evaluated, y<sub>i</sub> is the ground-truth emotion of clip i, and "
    "ŷ<sub>i</sub> is the predicted emotion. For binary deepfake detection, ACC is computed from the confusion matrix as:",
    eq("acc_bin"),
    "where TP, TN, FP and FN are the true positives, true negatives, false positives and false negatives."))
S.append(gap(2))
S.append(note("Why: TP and TN are binary-confusion terms. A six-class emotion head has no single TP/TN, and N alone "
              "does not say what is counted as correct."))

# 2
S += fix(2, "Add a Balanced Accuracy subsection", "Insert after the new Specificity subsection (Fix 3), before F1-Score, p. 110")
S.append(replace(
    "<b>Balanced Accuracy (BAcc)</b>",
    "Balanced Accuracy is reported for the fused detector in Research Question 3. It is the mean of the true "
    "positive rate (Fake Recall) and the true negative rate (Real Specificity), so each class contributes equally "
    "regardless of its size:",
    eq("bacc"),
    "A value of 0.5 indicates chance-level performance and 1.0 indicates perfect detection of both classes. "
    "Because Evaluation Set B contains 350 real and 350 fake clips, BAcc is numerically equal to ACC on that "
    "partition; it is reported so that results remain comparable with studies that evaluate on the full, "
    "imbalanced FakeAVCeleb distribution."))

# 3
S += fix(3, "Add a Specificity subsection", "Insert after Recall, before Balanced Accuracy, p. 110")
S.append(replace(
    "<b>Specificity</b>",
    "Specificity, reported as Real Video Specificity, is the true negative rate (TNR). It measures how well the "
    "system recognizes authentic clips as authentic:",
    eq("tnr"),
    "A high specificity means that real clips are rarely flagged as deepfakes. It is reported for the fused model "
    "in Research Question 3, and it is the metric most exposed to legitimate audio-visual incongruity such as "
    "sarcasm, which is the concern addressed in Research Question 4."))

# 4
S += fix(4, "Recall sentence (logic error)", "Data Analysis → Recall, p. 110")
S.append(current("Here, a high recall value signifies that the system does not miss many fake videos. In that case, "
                 "however, it may be flagging many real clips as well, which would result in a <b>low false positive "
                 "rate</b> and hence, low precision."))
S.append(gap())
S.append(replace(
    "Here, a high recall value signifies that the system does not miss many fake videos. A system can reach high "
    "recall by flagging nearly every clip as fake, however, which also flags many real clips and results in a "
    "<b>high false positive rate</b> and hence low precision."))
S.append(gap(2))
S.append(note("Flagging many real clips raises the false positive rate. It does not lower it. The old sentence also "
              "starts with “In that case,” which has no antecedent."))

# 5
S += fix(5, "F1-Score: formula and “averaging”", "Data Analysis → F1-Score, p. 110")
S.append(current("It is a great way to evaluate the system because it settles the trade-off between recall and "
                 "precision values by <b>averaging them as one value</b>."))
S.append(gap())
S.append(replace(
    "It is a suitable way to evaluate the system because it combines precision and recall into one value through "
    "their harmonic mean, which is low whenever either component is low:",
    eq("f1"),
    "For the six-class emotion heads, F1 is computed for each class in a one-vs-rest manner and macro-averaged "
    "over the K = 6 classes:",
    eq("f1macro")))
S.append(gap(2))
S.append(note("Macro-averaging is my recommendation because Table 5.A/5.B in the template report a per-class F1 and "
              "an Overall row. If the group prefers weighted averaging, say so in this sentence and use the same "
              "choice in the tables."))

# 6
S += fix(6, "TPR / FPR / AUC formulas", "Data Analysis → ROC Curve and Area Under the Curve (AUC), p. 110")
S.append(current("The receiver operating characteristic (ROC) curve plots the true positive rate against the false "
                 "positive rate across all possible decision thresholds. AUC condenses this curve into a single "
                 "value between 0 and 1, representing the probability that the model ranks a randomly chosen fake "
                 "sample above a randomly chosen real one."))
S.append(gap())
S.append(replace(
    "The receiver operating characteristic (ROC) curve plots the true positive rate (TPR) against the false "
    "positive rate (FPR) at every possible decision threshold, where",
    eq("tpr"), eq("fpr"),
    "AUC is the area under this curve:",
    eq("auc_int"),
    "Equivalently, AUC is the probability that a randomly chosen fake clip receives a higher score than a randomly "
    "chosen real clip, counting ties as one half:",
    eq("auc_rank"),
    "where s<sub>i</sub><super>F</super> is the score of the i-th fake clip, s<sub>j</sub><super>R</super> is the "
    "score of the j-th real clip, and N<sub>F</sub> and N<sub>R</sub> are the numbers of fake and real clips "
    "(N<sub>F</sub> = N<sub>R</sub> = 350 in Evaluation Set B). An AUC near 1.0 …"))
S.append(gap(2))
S.append(note("Keep the existing sentences from “An AUC near 1.0 indicates…” onward unchanged. The manuscript also has "
              "a shorter “Area Under the Curve (AUC)” subsection on p. 107 that repeats the ROC definition. Either "
              "delete the repeated sentence there or point it to this subsection."))

# 7
S += fix(7, "Research Question 4 in Data Analysis", "Data Analysis, opening paragraphs, p. 106")
S.append(current("The data analysis procedure is structured according to the <b>three research questions</b> guiding "
                 "this study."))
S.append(gap())
S.append(replace("The data analysis procedure is structured according to the <b>four research questions</b> guiding "
                 "this study."))
S.append(gap(3))
S.append(P("Then insert this paragraph after the Research Question 3 paragraph (end of p. 106):", body))
S.append(replace(
    "For Research Question 4, the auxiliary Sarcasm Head is evaluated as a binary classifier (sarcastic = 1, "
    "non-sarcastic = 0) on the held-out MUStARD test partition. MUStARD is class-balanced at the corpus level "
    "(345 sarcastic and 345 non-sarcastic utterances). The Sarcasm Head’s confusion matrix is summarized with "
    "Accuracy, Precision, Recall and F1-score, as defined below, with TP, FP, TN and FN counted with respect to the "
    "sarcasm label. These scores indicate how reliably the Sarcasm Head separates rhetorical irony from literal "
    "speech, which the fused detector relies on to avoid flagging legitimate incongruity as manipulation."))
S.append(gap(2))
S.append(note("“Held-out MUStARD test partition” assumes the group confirms the 80/10/10 speaker-disjoint split on "
              "p. 69 as the test set for the heads (see Fix B2). Report the actual MUStARD test count in Table 7."))

# 8
S += fix(8, "Merge the duplicate DeLong sections", "Data Analysis, “DeLong” (p. 111) and “Paired DeLong Test for Correlated ROC Curves” (pp. 112–113)")
S.append(P("<b>Delete both</b> the heading “DeLong” with its paragraph (p. 111) and the heading “Paired DeLong Test for "
           "Correlated ROC Curves” with its text (pp. 112–113). Put this single subsection in their place "
           "(after the Bootstrap subsection):", body))
S.append(replace(
    "<b>DeLong’s Test for Correlated ROC Curves</b>",
    "To formally test Hypothesis 1, which states that the proposed framework significantly outperforms the "
    "replicated multimodal baseline ACE-Net (Yu et al., 2025), the non-parametric DeLong test (DeLong et al., 1988) "
    "is applied using the fast implementation of Sun and Xu (2014). The test compares the AUCs of two correlated "
    "ROC curves obtained from the same samples, which makes it appropriate for paired model outputs, unlike a "
    "standard t-test that assumes independent samples. Both models are scored on the identical clips of "
    "Evaluation Set B (350 real and 350 fake). The variances and the covariance of the two AUCs are estimated from "
    "the structural components of the underlying U-statistics, and the test statistic is:",
    eq("delong"),
    "where AUC<sub>1</sub> is the AUC of the proposed framework; AUC<sub>2</sub> is the AUC of the replicated "
    "ACE-Net baseline; Var(AUC<sub>1</sub>) and Var(AUC<sub>2</sub>) are the variances of the two AUC estimates; "
    "Cov(AUC<sub>1</sub>, AUC<sub>2</sub>) is the covariance between them on the identical paired test set; and Z "
    "follows a standard normal distribution N(0, 1) asymptotically under the null hypothesis. "
    "H<sub>0</sub>: AUC<sub>1</sub> = AUC<sub>2</sub> is rejected at the two-sided 5% level when |Z| &gt; 1.96 "
    "(p &lt; 0.05). The proposed framework is concluded to outperform ACE-Net only if, in addition, "
    "AUC<sub>1</sub> &gt; AUC<sub>2</sub>."))
S.append(gap(2))
S.append(note("The old text says “Z &gt; 1.96 corresponding to a two-tailed significance level” but pairs it with a "
              "one-sided alternative (AUC<sub>1</sub> &gt; AUC<sub>2</sub>). A one-sided 5% test uses 1.645, and 1.96 is "
              "the two-sided value. The wording above uses the two-sided rule plus a direction check, which matches "
              "“p &lt; 0.05” in Hypothesis 1. Also update the Table 5 DeLong row so it points to this one subsection."))

# 9
S += fix(9, "Eight missing in-text citations (Chapter 3)", "Add each citation at the quoted phrase")
rows = [[P("#", cellb), P("Page", cellb), P("Find", cellb), P("Replace with", cellb)]]
cits = [
    ("1", "73", "Using three specialized encoders namely Wav2Vec 2.0 for audio, BERT for text, and Vision Transformer (ViT) for visual",
     "…namely Wav2Vec 2.0 (Baevski et al., 2020) for audio, BERT (Devlin et al., 2019) for text, and Vision Transformer (ViT; Dosovitskiy et al., 2021) for visual"),
    ("2", "76", "using RetinaFace (Facial Alignment &amp; Landmark Detector)",
     "using RetinaFace (Deng et al., 2020) (Facial Alignment &amp; Landmark Detector)"),
    ("3", "78", "Wav2Vec 2.0 serves as the audio encoder",
     "Wav2Vec 2.0 (Baevski et al., 2020) serves as the audio encoder"),
    ("4", "79", "BERT serves as the text encoder",
     "BERT (Devlin et al., 2019) serves as the text encoder"),
    ("5", "79", "The Vision Transformer serves as the visual encoder",
     "The Vision Transformer (Dosovitskiy et al., 2021) serves as the visual encoder"),
    ("6", "93", "Fake content was generated using four methods: FaceSwap, FSGAN, Wav2Lip, and SV2TTS",
     "…FaceSwap, FSGAN (Nirkin et al., 2019), Wav2Lip (Prajwal et al., 2020), and SV2TTS"),
    ("7", "95", "generated using StyleTTS2 and converted to match the original speaker’s voice using RVC",
     "generated using StyleTTS2 (Li et al., 2023) and converted…"),
    ("8", "95", "Wav2Lip rewrites the lip region… / SadTalker takes a single portrait image",
     "Wav2Lip (Prajwal et al., 2020) rewrites… / SadTalker (Zhang et al., 2023) takes…"),
]
for c in cits:
    rows.append([P(c[0], cell), P(c[1], cell), P(c[2], cell), P(c[3], cell)])
t = Table(rows, colWidths=[0.28 * inch, 0.42 * inch, W * 0.38, W - 1.08 * inch - W * 0.38], repeatRows=1)
t.setStyle(TableStyle([
    ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#b5bccb")),
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e3e8f3")),
    ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
]))
S.append(t)
S.append(gap(2))
S.append(note("Rows 1, 3, 4, 5 and 8 each give one of the first-mention citations the fix list asks for. FSGAN "
              "(Nirkin et al., 2019) is already in your reference list, so row 6 needs no new entry. "
              "SV2TTS and FaceSwap are also uncited in Chapter 3; they are not on the fix list."))

# 10
S += fix(10, "Missing references (add in alphabetical position)", "References, from p. 115")
S.append(replace(
    "<b>Deng, J., Guo, J., Ververas, E., Kotsia, I., &amp; Zafeiriou, S. (2020).</b> RetinaFace: Single-shot "
    "multi-level face localisation in the wild. In <i>Proceedings of the IEEE/CVF Conference on Computer Vision and "
    "Pattern Recognition (CVPR)</i> (pp. 5203–5212).",
    "<b>Li, Y. A., Han, C., Raghavan, V. S., Mischler, G., &amp; Mesgarani, N. (2023).</b> StyleTTS 2: Towards "
    "human-level text-to-speech through style diffusion and adversarial training with large speech language models. "
    "In <i>Advances in Neural Information Processing Systems</i> (Vol. 36).",
    "<b>Sun, X., &amp; Xu, W. (2014).</b> Fast implementation of DeLong’s algorithm for comparing the areas under "
    "correlated receiver operating characteristic curves. <i>IEEE Signal Processing Letters, 21</i>(11), 1389–1393. "
    "https://doi.org/10.1109/LSP.2014.2337313",
    "<b>Zhang, W., Cun, X., Wang, X., Zhang, Y., Shen, X., Guo, Y., Shan, Y., &amp; Wang, F. (2023).</b> SadTalker: "
    "Learning realistic 3D motion coefficients for stylized audio-driven single image talking face animation. In "
    "<i>Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)</i> (pp. 8652–8661)."))
S.append(gap(2))
S.append(note("Sun and Xu (2014) is already cited in the text (p. 111) but has no entry. Also required, though not in "
              "the 4-item list: <b>Haliassos, A., Vougioukas, K., Petridis, S., &amp; Pantic, M. (2021).</b> Lips don’t "
              "lie: A generalisable and robust approach to face forgery detection. In <i>Proceedings of the IEEE/CVF "
              "Conference on Computer Vision and Pattern Recognition (CVPR)</i> (pp. 5039–5049). It is cited on "
              "pp. 60–61 and missing from the list. Your list has two other entries beginning “Li, Y.” "
              "(2024); “Li et al., 2023” does not clash with them. Double-check the author lists and page ranges "
              "against the publisher pages."))

# ---------------- PART B ----------------
S.append(PageBreak())
S.append(P("Part B — Contradictions found in the manuscript", h1))

# B1
S += fix("B1", "Test-set size contradicts the balanced Evaluation Set B", "Data Analysis, p. 106 (and “class skew” in MCC, p. 111)")
S.append(current("AUC is chosen over accuracy for this stage because it provides a threshold-independent measure of "
                 "detection performance, making it more suitable for evaluating and comparing deepfake detection "
                 "frameworks across different operating conditions. This threshold-independence is particularly "
                 "relevant given the class distribution of the FakeAVCeleb test set, which contains substantially "
                 "more forged samples (<b>19,500</b>) than authentic ones (<b>500</b>). Under such imbalance, accuracy "
                 "computed at a single fixed threshold can be misleadingly inflated by the majority class, whereas "
                 "AUC evaluates separability across the full range of decision thresholds and is therefore less "
                 "sensitive to this skew."))
S.append(gap())
S.append(replace(
    "AUC is chosen over accuracy for this stage because it provides a threshold-independent measure of detection "
    "performance, making it more suitable for evaluating and comparing deepfake detection frameworks across "
    "different operating conditions. The full FakeAVCeleb v1.2 corpus is heavily skewed toward forged clips "
    "(19,500 fake and 500 real), but all reported metrics in this study are computed on Evaluation Set B, a "
    "class-balanced partition of 700 clips (350 real and 350 fake) drawn from Celebrity Pool B. AUC is the primary "
    "metric because it does not depend on a single decision threshold and because it allows comparison with prior "
    "work that evaluated on the full, imbalanced corpus."))
S.append(gap(3))
S.append(P("In the MCC subsection (p. 111), change one phrase:", body))
S.append(current("To guarantee rigorous forensic evaluation <b>under class skew</b>, the Matthews Correlation "
                 "Coefficient (MCC) is reported as the definitive index of contingency table quality:"))
S.append(gap(3))
S.append(replace("To provide a rigorous summary of all four cells of the confusion matrix, the Matthews Correlation "
                 "Coefficient (MCC) is reported as the primary thresholded index of contingency table quality:"))
S.append(gap(2))
S.append(note("The manuscript says Evaluation Set B is the only set used for reported metrics (pp. 69, 98–99), so the "
              "19,500/500 argument describes the corpus, not the test set. The Table 5 row for MCC can keep "
              "“robust to extreme class skew” as a general property."))

# B2
S += fix("B2", "Emotion and sarcasm heads are said to be evaluated on training data", "Data Analysis, first paragraph, p. 106")
S.append(current("…the audio-text emotion head (EmotionHead A) and visual emotion head (EmotionHead B) are evaluated "
                 "as auxiliary classifiers <b>on the training data</b>, measuring how accurately each head predicts "
                 "the correct emotion class from its respective feature vector."))
S.append(gap())
S.append(replace(
    "…the audio-text emotion head (Emotion Head A) and visual emotion head (Emotion Head B) are evaluated as "
    "auxiliary classifiers on the held-out test partition of the core pool (the 10% speaker-disjoint test split, "
    "N = 1,774 clips across CREMA-D, MELD, CMU-MOSEI and MUStARD), reporting results on its CREMA-D portion for "
    "Research Questions 1 and 2. Each head’s score measures how accurately it predicts the correct emotion class "
    "from its respective feature vector."))
S.append(gap(2))
S.append(note("The split already exists on p. 69 (14,193 / 1,774 / 1,774 clips, speaker-disjoint) and in Section G "
              "(p. 83), so evaluating on training data contradicts the manuscript. State the actual CREMA-D and "
              "MUStARD test counts in the manuscript before filling Tables 4–8, because the split is pooled and the "
              "template’s “Clip 700” rows cannot apply to them."))

# B3
S += fix("B3", "FakeAVCeleb is called “test only”, but 300 clips are used for adaptation", "Datasets, FakeAVCeleb bullet, p. 65")
S.append(current("In this study, FakeAVCeleb serves exclusively as the test set and is <b>not used in any part of the "
                 "training process</b>."))
S.append(gap())
S.append(replace(
    "In this study, FakeAVCeleb is not used in pretraining. Adaptation Set A (150 real and 150 fake clips from "
    "Celebrity Pool A) tunes only the bottleneck projection layer and classifier, and all reported metrics are "
    "computed on Evaluation Set B (350 real and 350 fake clips from Celebrity Pool B), which has zero identity "
    "overlap with Set A."))
S.append(gap(2))
S.append(note("The identity-disjoint protocol is described on pp. 69–70 and p. 99, so this sentence needs to match it."))

# B4
S += fix("B4", "Table 5 (p. 114) is out of date", "Summary of Statistical Tools and Their Application")
rows = [[P("Row", cellb), P("Currently", cellb), P("Replace with", cellb)],
        [P("Accuracy", cell), P("Overall correct classification rate | Final fused model evaluation", cell),
         P("Overall correct classification rate | Emotion Heads A and B (RQ1–2); fused model at the decision threshold (RQ3)", cell)],
        [P("F1-Score (Supplementary)", cell), P("Balance precision and recall; <b>main performance metric</b> | Per-modality and fused model", cell),
         P("Harmonic mean of precision and recall; supplementary to AUC | Emotion heads (macro-averaged), fused model, Sarcasm Head", cell)],
        [P("Balanced Accuracy (new row)", cell), P("—", cell),
         P("Mean of fake recall and real specificity | Fused model (RQ3)", cell)],
        [P("Specificity (new row)", cell), P("—", cell),
         P("Correct acceptance of authentic clips; 1 − FPR | Fused model (RQ3)", cell)],
        [P("Sarcasm Head (new row)", cell), P("—", cell),
         P("Accuracy, Precision, Recall and F1 of the sarcasm label | Sarcasm Head on MUStARD (RQ4)", cell)],
        [P("DeLong’s Test", cell), P("…between the proposed framework and the replicated ACE-Net baseline…", cell),
         P("No change in wording; make sure it points to the single merged DeLong subsection (Fix 8)", cell)]]
t = Table(rows, colWidths=[1.25 * inch, W * 0.36, W - 1.25 * inch - W * 0.36], repeatRows=1)
t.setStyle(TableStyle([
    ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#b5bccb")),
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e3e8f3")),
    ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
]))
S.append(t)
S.append(gap(2))
S.append(note("The text on p. 110 calls F1 a supplementary metric, while Table 5 calls it the “main performance "
              "metric”. The main metrics are AUC (RQ3) and MCC (thresholded)."))

# B5
S += fix("B5", "Group decisions that these fixes depend on", "Settle before editing the tables")
S.append(P("• <b>Emotion and sarcasm test sets.</b> Fix 7 and B2 use the existing 1,774-clip speaker-disjoint split. "
           "If the group would rather build a separate test set, change those two paragraphs.<br/>"
           "• <b>Evaluation set.</b> The manuscript already commits to the balanced 700-clip Set B, which this "
           "document follows (B1, B3, Fix 8).<br/>"
           "• <b>Baselines.</b> Hypothesis 1 tests ACE-Net only, so Fix 8 applies DeLong to ACE-Net only. If the "
           "group wants DeLong tests against MesoNet, XceptionNet or LipForensics, add them to Hypothesis 1 and "
           "describe a multiple-comparison correction.", body))


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("DV", 7.2)
    canvas.setFillColor(GREY)
    canvas.drawString(0.75 * inch, 0.5 * inch, "DeepSentinel — Required fixes to Chapters 1–3")
    canvas.drawRightString(letter[0] - 0.75 * inch, 0.5 * inch, f"Page {doc.page}")
    canvas.restoreState()


doc = SimpleDocTemplate("/mnt/user-data/outputs/DeepSentinel_Required_Fixes.pdf", pagesize=letter,
                        leftMargin=0.75 * inch, rightMargin=0.75 * inch,
                        topMargin=0.7 * inch, bottomMargin=0.8 * inch,
                        title="DeepSentinel — Required Fixes to Chapters 1–3", author="")
doc.build(S, onFirstPage=footer, onLaterPages=footer)
print("built")