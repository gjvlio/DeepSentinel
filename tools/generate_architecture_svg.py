"""
generate_architecture_svg.py
Builds the complete end-to-end DeepSentinel System Architecture diagram in the
iconic DeepSentinel Dark Theme (original design style) with the modular 4-stream
299D Unified Bottleneck Detection module and full congruent multi-stage connectivity.
"""
from pathlib import Path

def create_system_architecture_svg() -> str:
    svg = '''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1680 2380" width="1680" height="2380" style="background-color: #0f172a; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;">
  <defs>
    <!-- Shadows & Glows -->
    <filter id="card-shadow" x="-4%" y="-3%" width="108%" height="108%" filterUnits="userSpaceOnUse">
      <feDropShadow dx="0" dy="8" stdDeviation="12" flood-color="#000000" flood-opacity="0.6"/>
    </filter>
    <filter id="node-shadow" x="-5%" y="-5%" width="110%" height="115%" filterUnits="userSpaceOnUse">
      <feDropShadow dx="0" dy="4" stdDeviation="6" flood-color="#000000" flood-opacity="0.4"/>
    </filter>
    <filter id="glow-cyan" x="-8%" y="-8%" width="116%" height="120%" filterUnits="userSpaceOnUse">
      <feDropShadow dx="0" dy="4" stdDeviation="10" flood-color="#06b6d4" flood-opacity="0.35"/>
    </filter>
    <filter id="glow-bottle" x="-8%" y="-8%" width="116%" height="120%" filterUnits="userSpaceOnUse">
      <feDropShadow dx="0" dy="6" stdDeviation="12" flood-color="#38bdf8" flood-opacity="0.4"/>
    </filter>

    <!-- Original Pastel & Sleek Gradients -->
    <linearGradient id="grad-sarc" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="#fef08a"/>
      <stop offset="100%" stop-color="#fef9c3"/>
    </linearGradient>
    <linearGradient id="grad-cbp" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="#bbf7d0"/>
      <stop offset="100%" stop-color="#dcfce7"/>
    </linearGradient>
    <linearGradient id="grad-emo" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="#fef08a"/>
      <stop offset="100%" stop-color="#fef9c3"/>
    </linearGradient>
    <linearGradient id="grad-cooccur" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="#fed7aa"/>
      <stop offset="100%" stop-color="#ffedd5"/>
    </linearGradient>
    <linearGradient id="grad-diff" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="#fed7aa"/>
      <stop offset="100%" stop-color="#ffedd5"/>
    </linearGradient>
    <linearGradient id="grad-blue" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="#bae6fd"/>
      <stop offset="100%" stop-color="#e0f2fe"/>
    </linearGradient>
    <linearGradient id="grad-purple" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="#e9d5ff"/>
      <stop offset="100%" stop-color="#f3e8ff"/>
    </linearGradient>
    <linearGradient id="grad-red" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="#fecaca"/>
      <stop offset="100%" stop-color="#fee2e2"/>
    </linearGradient>

    <!-- Module Header Gradients -->
    <linearGradient id="grad-hdr-gen" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="#0284c7"/>
      <stop offset="100%" stop-color="#0369a1"/>
    </linearGradient>
    <linearGradient id="grad-hdr-prep" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="#b45309"/>
      <stop offset="100%" stop-color="#92400e"/>
    </linearGradient>
    <linearGradient id="grad-hdr-det" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="#0284c7"/>
      <stop offset="100%" stop-color="#1d4ed8"/>
    </linearGradient>
    <linearGradient id="grad-hdr-train" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="#7e22ce"/>
      <stop offset="100%" stop-color="#581c87"/>
    </linearGradient>

    <!-- Arrow Markers -->
    <marker id="arr-gray" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#94a3b8"/>
    </marker>
    <marker id="arr-cyan" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#38bdf8"/>
    </marker>
    <marker id="arr-green" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#10b981"/>
    </marker>
    <marker id="arr-orange" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#f97316"/>
    </marker>
    <marker id="arr-yellow" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#eab308"/>
    </marker>
    <marker id="arr-purple" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#a855f7"/>
    </marker>
  </defs>

  <!-- ========================================================================= -->
  <!-- 1. DEEPFAKE GENERATION MODULE (TOP PANEL) -->
  <!-- ========================================================================= -->
  <g filter="url(#card-shadow)">
    <!-- Container Card -->
    <rect x="50" y="40" width="1580" height="360" rx="16" fill="#1e293b" stroke="#38bdf8" stroke-width="1.5"/>
    
    <!-- Module Header Banner -->
    <rect x="75" y="60" width="290" height="36" rx="8" fill="url(#grad-hdr-gen)"/>
    <text x="220" y="83" font-size="14.5" font-weight="700" fill="#ffffff" text-anchor="middle" letter-spacing="0.5">Deepfake Generation Module</text>

    <!-- Raw Benchmark Corpora -->
    <rect x="730" y="70" width="220" height="60" rx="8" fill="#0f172a" stroke="#64748b" stroke-width="1.5" filter="url(#node-shadow)"/>
    <text x="840" y="94" font-size="13.5" font-weight="700" fill="#f8fafc" text-anchor="middle">Raw Audio-Visual Clips</text>
    <text x="840" y="113" font-size="12" fill="#94a3b8" text-anchor="middle">(CREMA-D, MELD, MOSEI)</text>

    <!-- Arrow to Generation Dispatcher -->
    <path d="M 840 130 L 840 160" fill="none" stroke="#94a3b8" stroke-width="2" marker-end="url(#arr-gray)"/>

    <!-- Generation Splitter Node -->
    <rect x="700" y="160" width="280" height="34" rx="6" fill="#020617" stroke="#38bdf8" stroke-width="1.2"/>
    <text x="840" y="182" font-size="13" font-weight="700" fill="#38bdf8" text-anchor="middle">Deepfake Generation Pipeline</text>

    <!-- Branch Arrows to Multi-Modal vs Single-Modal -->
    <path d="M 780 194 L 780 215 L 480 215 L 480 235" fill="none" stroke="#94a3b8" stroke-width="2" marker-end="url(#arr-gray)"/>
    <path d="M 900 194 L 900 215 L 1200 215 L 1200 235" fill="none" stroke="#94a3b8" stroke-width="2" marker-end="url(#arr-gray)"/>

    <!-- Multi-Modal Synthesis Card -->
    <rect x="350" y="235" width="260" height="65" rx="8" fill="#0f172a" stroke="#38bdf8" stroke-width="1.5" filter="url(#node-shadow)"/>
    <text x="480" y="258" font-size="13" font-weight="700" fill="#38bdf8" text-anchor="middle">Multi-Modal Generation</text>
    <text x="480" y="276" font-size="11.5" fill="#e2e8f0" text-anchor="middle">Wav2Lip + SadTalker / SimSwap</text>
    <text x="480" y="291" font-size="10.5" fill="#94a3b8" text-anchor="middle">(Cross-Modal Emotion Discordance)</text>

    <!-- Single-Modal Synthesis Card -->
    <rect x="1070" y="235" width="260" height="65" rx="8" fill="#0f172a" stroke="#38bdf8" stroke-width="1.5" filter="url(#node-shadow)"/>
    <text x="1200" y="258" font-size="13" font-weight="700" fill="#38bdf8" text-anchor="middle">Single-Modal Generation</text>
    <text x="1200" y="276" font-size="11.5" fill="#e2e8f0" text-anchor="middle">SimSwap (Face) / Voice Conversion</text>
    <text x="1200" y="291" font-size="10.5" fill="#94a3b8" text-anchor="middle">(Acoustic or Visual Perturbations)</text>

    <!-- Arrows from Generation to Output Ingestion Node -->
    <path d="M 480 300 L 480 320 L 730 320 L 730 335" fill="none" stroke="#94a3b8" stroke-width="2" marker-end="url(#arr-gray)"/>
    <path d="M 1200 300 L 1200 320 L 950 320 L 950 335" fill="none" stroke="#94a3b8" stroke-width="2" marker-end="url(#arr-gray)"/>

    <!-- Synthesized / Pristine Video Node -->
    <rect x="680" y="335" width="320" height="48" rx="8" fill="#0284c7" filter="url(#node-shadow)"/>
    <text x="840" y="356" font-size="13.5" font-weight="700" fill="#ffffff" text-anchor="middle">Input Video Stream (.mp4 / .mov)</text>
    <text x="840" y="373" font-size="11" fill="#e0f2fe" text-anchor="middle">Genuine Clips &amp; Synthetic Manipulations</text>
  </g>

  <!-- Connective Arrow from Stage 1 to Stage 2 -->
  <path d="M 840 400 L 840 440" fill="none" stroke="#38bdf8" stroke-width="3" marker-end="url(#arr-cyan)"/>


  <!-- ========================================================================= -->
  <!-- 2. DATA PREPROCESSING MODULE (MIDDLE PANEL) -->
  <!-- ========================================================================= -->
  <g filter="url(#card-shadow)">
    <!-- Container Card -->
    <rect x="50" y="440" width="1580" height="660" rx="16" fill="#1e293b" stroke="#eab308" stroke-width="1.5"/>
    
    <!-- Module Header Banner -->
    <rect x="75" y="460" width="290" height="36" rx="8" fill="url(#grad-hdr-prep)"/>
    <text x="220" y="483" font-size="14.5" font-weight="700" fill="#ffffff" text-anchor="middle" letter-spacing="0.5">Data Preprocessing Module</text>

    <!-- Branching to Audio vs Visual Pipelines -->
    <path d="M 840 440 L 840 465 L 430 465 L 430 495" fill="none" stroke="#94a3b8" stroke-width="2" marker-end="url(#arr-gray)"/>
    <path d="M 840 440 L 840 465 L 1250 465 L 1250 495" fill="none" stroke="#94a3b8" stroke-width="2" marker-end="url(#arr-gray)"/>

    <!-- AUDIO PREPROCESSING SUB-PIPELINE (LEFT) -->
    <rect x="100" y="495" width="660" height="520" rx="12" fill="#0f172a" stroke="#475569" stroke-width="1.5"/>
    <text x="430" y="525" font-size="14.5" font-weight="700" fill="#f87171" text-anchor="middle">Audio Preprocessing</text>

    <!-- Waveform Extraction -->
    <rect x="290" y="540" width="280" height="42" rx="6" fill="#1e293b" stroke="#64748b" stroke-width="1.2"/>
    <text x="430" y="558" font-size="12" font-weight="700" fill="#f8fafc" text-anchor="middle">Extract Speech Waveform</text>
    <text x="430" y="573" font-size="10.5" fill="#94a3b8" text-anchor="middle">16 kHz Mono Float32 Audio Stream</text>

    <!-- RMS Gatekeeper Check -->
    <path d="M 430 582 L 430 602" fill="none" stroke="#94a3b8" stroke-width="1.5" marker-end="url(#arr-gray)"/>
    <rect x="270" y="602" width="320" height="40" rx="6" fill="#450a0a" stroke="#ef4444" stroke-width="1.2"/>
    <text x="430" y="619" font-size="11.5" font-weight="700" fill="#fca5a5" text-anchor="middle">Gatekeeper: RMS Energy &gt; -42 dBFS</text>
    <text x="430" y="634" font-size="10" fill="#f87171" text-anchor="middle">(Flags has_speech; sets Neutral default on silence)</text>

    <!-- Split to Wav2Vec2 and Whisper/BERT -->
    <path d="M 350 642 L 350 662 L 240 662 L 240 675" fill="none" stroke="#94a3b8" stroke-width="1.5" marker-end="url(#arr-gray)"/>
    <path d="M 510 642 L 510 662 L 620 662 L 620 675" fill="none" stroke="#94a3b8" stroke-width="1.5" marker-end="url(#arr-gray)"/>

    <!-- Wav2Vec 2.0 (Acoustic) -->
    <rect x="140" y="675" width="200" height="60" rx="8" fill="url(#grad-cbp)" stroke="#16a34a" stroke-width="1.5" filter="url(#node-shadow)"/>
    <text x="240" y="698" font-size="12.5" font-weight="700" fill="#14532d" text-anchor="middle">Wav2Vec 2.0 Base</text>
    <text x="240" y="714" font-size="11" fill="#166534" text-anchor="middle">Acoustic Pitch &amp; Prosody</text>
    <text x="240" y="728" font-size="10.5" font-style="italic" fill="#15803d" text-anchor="middle">z_a ∈ ℝ⁷⁶⁸</text>

    <!-- Whisper ASR + BERT (Lexical) -->
    <rect x="520" y="675" width="200" height="60" rx="8" fill="url(#grad-cbp)" stroke="#16a34a" stroke-width="1.5" filter="url(#node-shadow)"/>
    <text x="620" y="698" font-size="12.5" font-weight="700" fill="#14532d" text-anchor="middle">Whisper ASR → BERT</text>
    <text x="620" y="714" font-size="11" fill="#166534" text-anchor="middle">Lexical Semantic Content</text>
    <text x="620" y="728" font-size="10.5" font-style="italic" fill="#15803d" text-anchor="middle">z_t ∈ ℝ⁷⁶⁸</text>

    <!-- Connectors to Concatenation -->
    <path d="M 240 735 L 240 770 L 370 770 L 370 790" fill="none" stroke="#10b981" stroke-width="1.5" marker-end="url(#arr-green)"/>
    <path d="M 620 735 L 620 770 L 490 770 L 490 790" fill="none" stroke="#10b981" stroke-width="1.5" marker-end="url(#arr-green)"/>

    <!-- Concatenation Box -->
    <rect x="290" y="790" width="280" height="50" rx="8" fill="#1e293b" stroke="#64748b" stroke-width="1.5" filter="url(#node-shadow)"/>
    <text x="430" y="812" font-size="12.5" font-weight="700" fill="#f8fafc" text-anchor="middle">Feature Concatenation</text>
    <text x="430" y="829" font-size="11" fill="#94a3b8" text-anchor="middle">[ z_a ∥ z_t ] = Acoustic-Textual Fusion</text>

    <!-- Output Audio-Text Vector Z_AT -->
    <path d="M 430 840 L 430 875" fill="none" stroke="#10b981" stroke-width="2.5" marker-end="url(#arr-green)"/>
    <rect x="250" y="875" width="360" height="60" rx="10" fill="#15803d" stroke="#22c55e" stroke-width="1.5" filter="url(#node-shadow)"/>
    <text x="430" y="900" font-size="14" font-weight="800" fill="#ffffff" text-anchor="middle">Z_AT ∈ ℝ¹⁵³⁶</text>
    <text x="430" y="920" font-size="11.5" font-weight="500" fill="#dcfce7" text-anchor="middle">Synchronized Audio-Text Representation</text>


    <!-- VISUAL PREPROCESSING SUB-PIPELINE (RIGHT) -->
    <rect x="920" y="495" width="660" height="520" rx="12" fill="#0f172a" stroke="#475569" stroke-width="1.5"/>
    <text x="1250" y="525" font-size="14.5" font-weight="700" fill="#38bdf8" text-anchor="middle">Visual Preprocessing</text>

    <!-- Frame Extraction & Flow Filtering -->
    <rect x="1100" y="540" width="300" height="42" rx="6" fill="#1e293b" stroke="#64748b" stroke-width="1.2"/>
    <text x="1250" y="558" font-size="12" font-weight="700" fill="#f8fafc" text-anchor="middle">Extract Keyframe Sequence</text>
    <text x="1250" y="573" font-size="10.5" fill="#94a3b8" text-anchor="middle">Farneback Optical Flow Filtering (Motion &gt; 0.05)</text>

    <!-- RetinaFace Alignment -->
    <path d="M 1250 582 L 1250 602" fill="none" stroke="#94a3b8" stroke-width="1.5" marker-end="url(#arr-gray)"/>
    <rect x="1080" y="602" width="340" height="42" rx="6" fill="#1e293b" stroke="#64748b" stroke-width="1.2"/>
    <text x="1250" y="620" font-size="12" font-weight="700" fill="#f8fafc" text-anchor="middle">RetinaFace &amp; Landmark Alignment</text>
    <text x="1250" y="635" font-size="10.5" fill="#94a3b8" text-anchor="middle">5-Point Facial Crop &amp; Normalized 224×224 Canvas</text>

    <!-- Sharpness Ranking & Top-8 Selection -->
    <path d="M 1250 644 L 1250 664" fill="none" stroke="#94a3b8" stroke-width="1.5" marker-end="url(#arr-gray)"/>
    <rect x="1080" y="664" width="340" height="42" rx="6" fill="#1e293b" stroke="#64748b" stroke-width="1.2"/>
    <text x="1250" y="682" font-size="12" font-weight="700" fill="#f8fafc" text-anchor="middle">Sharpness Ranking (Laplacian Variance)</text>
    <text x="1250" y="697" font-size="10.5" fill="#94a3b8" text-anchor="middle">Top-8 Most Informative Keyframes Selected</text>

    <!-- ViT Feature Extraction -->
    <path d="M 1250 706 L 1250 726" fill="none" stroke="#94a3b8" stroke-width="1.5" marker-end="url(#arr-gray)"/>
    <rect x="1090" y="726" width="320" height="48" rx="8" fill="url(#grad-cbp)" stroke="#16a34a" stroke-width="1.5" filter="url(#node-shadow)"/>
    <text x="1250" y="746" font-size="12.5" font-weight="700" fill="#14532d" text-anchor="middle">Vision Transformer (ViT-Base/16)</text>
    <text x="1250" y="763" font-size="10.5" fill="#166534" text-anchor="middle">Extracts 8 × 768D Spatiotemporal Facial Embeddings</text>

    <!-- Temporal GRU Aggregation -->
    <path d="M 1250 774 L 1250 794" fill="none" stroke="#10b981" stroke-width="1.5" marker-end="url(#arr-green)"/>
    <rect x="1080" y="794" width="340" height="46" rx="8" fill="#1e293b" stroke="#64748b" stroke-width="1.5" filter="url(#node-shadow)"/>
    <text x="1250" y="814" font-size="12.5" font-weight="700" fill="#f8fafc" text-anchor="middle">Cross-Attention + 2-Layer Temporal GRU</text>
    <text x="1250" y="831" font-size="10.5" fill="#94a3b8" text-anchor="middle">Tracks Expression Trajectory &amp; Micro-Movements</text>

    <!-- Output Video Vector Z_V -->
    <path d="M 1250 840 L 1250 875" fill="none" stroke="#10b981" stroke-width="2.5" marker-end="url(#arr-green)"/>
    <rect x="1070" y="875" width="360" height="60" rx="10" fill="#15803d" stroke="#22c55e" stroke-width="1.5" filter="url(#node-shadow)"/>
    <text x="1250" y="900" font-size="14" font-weight="800" fill="#ffffff" text-anchor="middle">Z_V ∈ ℝ⁷⁶⁸</text>
    <text x="1250" y="920" font-size="11.5" font-weight="500" fill="#dcfce7" text-anchor="middle">Aggregated Spatiotemporal Visual Representation</text>
  </g>


  <!-- ========================================================================= -->
  <!-- 3. DETECTION MODULE (BOTTOM-LEFT PANEL - 4-STREAM UNIFIED BOTTLENECK) -->
  <!-- ========================================================================= -->
  <g filter="url(#card-shadow)">
    <!-- Container Card -->
    <rect x="50" y="1130" width="1020" height="1200" rx="16" fill="#1e293b" stroke="#0284c7" stroke-width="1.5"/>
    
    <!-- Module Header Banner -->
    <rect x="75" y="1150" width="260" height="36" rx="8" fill="url(#grad-hdr-det)"/>
    <text x="205" y="1173" font-size="14.5" font-weight="700" fill="#ffffff" text-anchor="middle" letter-spacing="0.5">Detection Module</text>

    <!-- Incoming Feats Connectors from Stage 2 -->
    <path d="M 430 935 L 430 1205" fill="none" stroke="#10b981" stroke-width="3" marker-end="url(#arr-green)"/>
    <path d="M 1250 935 L 1250 1090 L 850 1090 L 850 1205" fill="none" stroke="#10b981" stroke-width="3" marker-end="url(#arr-green)"/>

    <!-- Input Pins in Detection Module -->
    <rect x="330" y="1205" width="200" height="42" rx="8" fill="#15803d" stroke="#22c55e" stroke-width="1.5"/>
    <text x="430" y="1231" font-size="13" font-weight="700" fill="#ffffff" text-anchor="middle">Z_AT (Audio-Text: 1536D)</text>

    <rect x="750" y="1205" width="200" height="42" rx="8" fill="#15803d" stroke="#22c55e" stroke-width="1.5"/>
    <text x="850" y="1231" font-size="13" font-weight="700" fill="#ffffff" text-anchor="middle">Z_V (Visual Video: 768D)</text>

    <!-- ===================================================================== -->
    <!-- 4 CLEAN PARALLEL STREAMS INTO UNIFIED BOTTLENECK -->
    <!-- ===================================================================== -->

    <!-- STREAM 1: SARCASM HEAD (FAR LEFT) -->
    <g filter="url(#node-shadow)">
      <rect x="80" y="1285" width="160" height="75" rx="8" fill="url(#grad-sarc)" stroke="#ca8a04" stroke-width="1.5"/>
      <text x="160" y="1312" font-size="13.5" font-weight="700" fill="#854d0e" text-anchor="middle">Sarcasm Head</text>
      <text x="160" y="1330" font-size="11" fill="#a16207" text-anchor="middle">MLP(Z_AT)</text>
      <text x="160" y="1347" font-size="12" font-style="italic" fill="#713f12" text-anchor="middle">P_sarcasm ∈ ℝ¹</text>
    </g>
    <!-- Route Z_AT to Sarcasm -->
    <path d="M 360 1247 L 360 1265 L 160 1265 L 160 1285" fill="none" stroke="#94a3b8" stroke-width="1.5" marker-end="url(#arr-gray)"/>
    <!-- Stream 1 Downward Line (Straight) -->
    <path d="M 160 1360 L 160 1620" fill="none" stroke="#eab308" stroke-width="2.5" stroke-dasharray="4,4" marker-end="url(#arr-yellow)"/>
    <rect x="125" y="1480" width="70" height="24" rx="12" fill="#0f172a" stroke="#eab308" stroke-width="1"/>
    <text x="160" y="1496" font-size="11" font-weight="700" fill="#fde047" text-anchor="middle">1D Sarc</text>


    <!-- STREAM 2: COMPACT BILINEAR POOLING (CENTER-LEFT) -->
    <g filter="url(#node-shadow)">
      <rect x="270" y="1285" width="220" height="110" rx="8" fill="url(#grad-cbp)" stroke="#16a34a" stroke-width="1.5"/>
      <text x="380" y="1310" font-size="13.5" font-weight="700" fill="#14532d" text-anchor="middle">Compact Bilinear Pooling</text>
      <text x="380" y="1328" font-size="11" fill="#166534" text-anchor="middle">Count Sketch FFT (8,192D)</text>
      <text x="380" y="1345" font-size="11" fill="#166534" text-anchor="middle">Signed Sqrt + L2 Normalization</text>
      <line x1="285" y1="1355" x2="475" y2="1355" stroke="#86efac" stroke-width="1"/>
      <text x="380" y="1373" font-size="12" font-weight="600" fill="#15803d" text-anchor="middle">Linear + LayerNorm + GELU</text>
      <text x="380" y="1389" font-size="11.5" font-style="italic" fill="#166534" text-anchor="middle">fused_proj ∈ ℝ²⁵⁶</text>
    </g>
    <!-- Route Z_AT & Z_V into CBP -->
    <path d="M 410 1247 L 410 1285" fill="none" stroke="#94a3b8" stroke-width="1.5" marker-end="url(#arr-gray)"/>
    <path d="M 800 1247 L 800 1265 L 450 1265 L 450 1285" fill="none" stroke="#94a3b8" stroke-width="1.5" marker-end="url(#arr-gray)"/>
    <!-- Stream 2 Downward Line (Straight) -->
    <path d="M 380 1395 L 380 1620" fill="none" stroke="#10b981" stroke-width="2.5" marker-end="url(#arr-green)"/>
    <rect x="330" y="1480" width="100" height="24" rx="12" fill="#0f172a" stroke="#22c55e" stroke-width="1"/>
    <text x="380" y="1496" font-size="11" font-weight="700" fill="#86efac" text-anchor="middle">256D Features</text>


    <!-- STREAMS 3 & 4: EMOTION HEADS & INTERACTION MODULE (RIGHT) -->
    <rect x="520" y="1275" width="520" height="310" rx="12" fill="#0f172a" fill-opacity="0.8" stroke="#475569" stroke-width="1.5" stroke-dasharray="6,4"/>
    <text x="780" y="1295" font-size="12" font-weight="700" fill="#94a3b8" text-anchor="middle" letter-spacing="1">EMOTION INTERACTION SUBSYSTEM</text>

    <!-- Emotion Head A (Voice) -->
    <g filter="url(#node-shadow)">
      <rect x="545" y="1310" width="180" height="60" rx="8" fill="url(#grad-emo)" stroke="#ca8a04" stroke-width="1.5"/>
      <text x="635" y="1333" font-size="13" font-weight="700" fill="#854d0e" text-anchor="middle">Emotion Head A</text>
      <text x="635" y="1353" font-size="12" font-style="italic" fill="#713f12" text-anchor="middle">P_A ∈ ℝ⁶ (Vocal)</text>
    </g>
    <!-- Route Z_AT to Head A -->
    <path d="M 460 1247 L 460 1270 L 635 1270 L 635 1310" fill="none" stroke="#94a3b8" stroke-width="1.5" marker-end="url(#arr-gray)"/>

    <!-- Emotion Head B (Visual) -->
    <g filter="url(#node-shadow)">
      <rect x="835" y="1310" width="180" height="60" rx="8" fill="url(#grad-emo)" stroke="#ca8a04" stroke-width="1.5"/>
      <text x="925" y="1333" font-size="13" font-weight="700" fill="#854d0e" text-anchor="middle">Emotion Head B</text>
      <text x="925" y="1353" font-size="12" font-style="italic" fill="#713f12" text-anchor="middle">P_B ∈ ℝ⁶ (Facial)</text>
    </g>
    <!-- Route Z_V to Head B -->
    <path d="M 870 1247 L 870 1270 L 925 1270 L 925 1310" fill="none" stroke="#94a3b8" stroke-width="1.5" marker-end="url(#arr-gray)"/>

    <!-- Clean Local Routing into Co-occurrence & Mismatch -->
    <path d="M 605 1370 L 605 1410 L 620 1410" fill="none" stroke="#f97316" stroke-width="1.5" marker-end="url(#arr-orange)"/>
    <path d="M 895 1370 L 895 1410 L 710 1410" fill="none" stroke="#f97316" stroke-width="1.5" marker-end="url(#arr-orange)"/>

    <path d="M 665 1370 L 665 1485 L 835 1485" fill="none" stroke="#f97316" stroke-width="1.5" marker-end="url(#arr-orange)"/>
    <path d="M 955 1370 L 955 1485 L 905 1485" fill="none" stroke="#f97316" stroke-width="1.5" marker-end="url(#arr-orange)"/>

    <!-- Emotion Co-occurrence Box (Stream 3) -->
    <g filter="url(#node-shadow)">
      <rect x="545" y="1390" width="180" height="65" rx="8" fill="url(#grad-cooccur)" stroke="#ea580c" stroke-width="1.5"/>
      <text x="635" y="1413" font-size="12.5" font-weight="700" fill="#9a3412" text-anchor="middle">Emotion Co-occurrence</text>
      <text x="635" y="1430" font-size="11" fill="#c2410c" text-anchor="middle">vec(P_A ⊗ P_B)</text>
      <text x="635" y="1445" font-size="11" font-style="italic" font-weight="600" fill="#9a3412" text-anchor="middle">fused_emo ∈ ℝ³⁶</text>
    </g>

    <!-- Emotion Incongruence Box (Stream 4) -->
    <g filter="url(#node-shadow)">
      <rect x="835" y="1465" width="180" height="65" rx="8" fill="url(#grad-diff)" stroke="#ea580c" stroke-width="1.5"/>
      <text x="925" y="1488" font-size="12.5" font-weight="700" fill="#9a3412" text-anchor="middle">Emotion Incongruence</text>
      <text x="925" y="1505" font-size="11" fill="#c2410c" text-anchor="middle">|P_A − P_B|</text>
      <text x="925" y="1520" font-size="11" font-style="italic" font-weight="600" fill="#9a3412" text-anchor="middle">Δ ∈ ℝ⁶ (Mismatch)</text>
    </g>

    <!-- Streams 3 & 4 Downward Lines (Straight into Bottleneck) -->
    <path d="M 635 1455 L 635 1620" fill="none" stroke="#f97316" stroke-width="2.5" marker-end="url(#arr-orange)"/>
    <rect x="590" y="1530" width="90" height="24" rx="12" fill="#0f172a" stroke="#f97316" stroke-width="1"/>
    <text x="635" y="1546" font-size="11" font-weight="700" fill="#fdba74" text-anchor="middle">36D Outer</text>

    <path d="M 925 1530 L 925 1620" fill="none" stroke="#f97316" stroke-width="2.5" marker-end="url(#arr-orange)"/>
    <rect x="880" y="1560" width="90" height="24" rx="12" fill="#0f172a" stroke="#f97316" stroke-width="1"/>
    <text x="925" y="1576" font-size="11" font-weight="700" fill="#fdba74" text-anchor="middle">6D Delta</text>


    <!-- ===================================================================== -->
    <!-- UNIFIED BOTTLENECK VECTOR CONTAINER -->
    <!-- ===================================================================== -->
    <g filter="url(#glow-bottle)">
      <rect x="80" y="1625" width="960" height="95" rx="12" fill="#020617" stroke="#38bdf8" stroke-width="2.5"/>
      <text x="560" y="1653" font-size="15" font-weight="800" fill="#38bdf8" text-anchor="middle" letter-spacing="1">
        UNIFIED BOTTLENECK VECTOR (x_299 ∈ ℝ²⁹⁹)
      </text>

      <!-- Partition 1: Sarcasm (1D) -->
      <rect x="95" y="1667" width="130" height="40" rx="6" fill="#ca8a04" fill-opacity="0.35" stroke="#eab308" stroke-width="1.5"/>
      <text x="160" y="1692" font-size="12" font-weight="700" fill="#fef08a" text-anchor="middle">Sarcasm [1D]</text>

      <!-- Partition 2: CBP (256D) -->
      <rect x="235" y="1667" width="290" height="40" rx="6" fill="#15803d" fill-opacity="0.35" stroke="#22c55e" stroke-width="1.5"/>
      <text x="380" y="1692" font-size="12" font-weight="700" fill="#bbf7d0" text-anchor="middle">CBP Projection [256D]</text>

      <!-- Partition 3: Co-occurrence (36D) -->
      <rect x="535" y="1667" width="200" height="40" rx="6" fill="#c2410c" fill-opacity="0.35" stroke="#f97316" stroke-width="1.5"/>
      <text x="635" y="1692" font-size="12" font-weight="700" fill="#fed7aa" text-anchor="middle">Co-occurrence [36D]</text>

      <!-- Partition 4: Incongruence (6D) -->
      <rect x="745" y="1667" width="280" height="40" rx="6" fill="#c2410c" fill-opacity="0.35" stroke="#f97316" stroke-width="1.5"/>
      <text x="885" y="1692" font-size="12" font-weight="700" fill="#fed7aa" text-anchor="middle">Incongruence Δ [6D]</text>
    </g>

    <!-- Classifier MLP -->
    <path d="M 560 1720 L 560 1760" fill="none" stroke="#38bdf8" stroke-width="2.5" marker-end="url(#arr-cyan)"/>
    <g filter="url(#node-shadow)">
      <rect x="360" y="1760" width="400" height="60" rx="8" fill="#0f172a" stroke="#38bdf8" stroke-width="2"/>
      <text x="560" y="1784" font-size="13.5" font-weight="700" fill="#38bdf8" text-anchor="middle">3-Layer Classifier MLP</text>
      <text x="560" y="1803" font-size="11.5" fill="#e2e8f0" text-anchor="middle">LN(299) → FC(512) → Squeeze-and-Excitation → FC(128) → FC(1)</text>
    </g>

    <!-- Calibration & Adaptive Memory -->
    <path d="M 560 1820 L 560 1855" fill="none" stroke="#94a3b8" stroke-width="2" marker-end="url(#arr-gray)"/>
    <g filter="url(#node-shadow)">
      <rect x="340" y="1855" width="440" height="60" rx="8" fill="#0f172a" stroke="#64748b" stroke-width="1.5"/>
      <text x="560" y="1878" font-size="13" font-weight="700" fill="#f8fafc" text-anchor="middle">Adaptive Memory &amp; Post-Hoc Calibration</text>
      <text x="560" y="1896" font-size="11" fill="#94a3b8" text-anchor="middle">Asymmetric Sharpening + Gated Biological Harmony Prior</text>
    </g>

    <!-- Final Forensic Output Verdict -->
    <path d="M 560 1915 L 560 1950" fill="none" stroke="#38bdf8" stroke-width="2.5" marker-end="url(#arr-cyan)"/>
    <g filter="url(#glow-bottle)">
      <rect x="300" y="1950" width="520" height="95" rx="12" fill="#020617" stroke="#38bdf8" stroke-width="2.5"/>
      <text x="560" y="1978" font-size="15" font-weight="800" fill="#38bdf8" text-anchor="middle" letter-spacing="0.5">FINAL FORENSIC DETECTION OUTPUT</text>
      <text x="560" y="2000" font-size="13" font-weight="700" fill="#ffffff" text-anchor="middle">P(fake) ∈ [0.0, 1.0] (Calibrated Manipulation Score)</text>
      <text x="560" y="2020" font-size="11.5" fill="#94a3b8" text-anchor="middle">Voice Emotion (P_A) vs Face Emotion (P_B) Breakdown • Sarcasm Level</text>
    </g>
  </g>


  <!-- ========================================================================= -->
  <!-- 4. TRAINING & MULTI-TASK OPTIMIZATION MODULE (BOTTOM-RIGHT PANEL) -->
  <!-- ========================================================================= -->
  <g filter="url(#card-shadow)">
    <!-- Container Card -->
    <rect x="1100" y="1130" width="530" height="1200" rx="16" fill="#1e293b" stroke="#a855f7" stroke-width="1.5"/>
    
    <!-- Module Header Banner -->
    <rect x="1125" y="1150" width="240" height="36" rx="8" fill="url(#grad-hdr-train)"/>
    <text x="1245" y="1173" font-size="14.5" font-weight="700" fill="#ffffff" text-anchor="middle" letter-spacing="0.5">Training Module</text>

    <!-- Multi-Task Loss Objectives Container -->
    <rect x="1125" y="1210" width="480" height="480" rx="12" fill="#0f172a" stroke="#a855f7" stroke-width="1.5"/>
    <text x="1365" y="1238" font-size="14" font-weight="800" fill="#c084fc" text-anchor="middle">Multi-Task Loss Objective Formulation</text>

    <!-- Loss 1: Detection Loss -->
    <g filter="url(#node-shadow)">
      <rect x="1145" y="1255" width="440" height="75" rx="8" fill="url(#grad-red)" stroke="#ef4444" stroke-width="1.5"/>
      <text x="1365" y="1278" font-size="13" font-weight="700" fill="#991b1b" text-anchor="middle">Detection Loss (L_det)</text>
      <text x="1365" y="1296" font-size="11.5" fill="#b91c1c" text-anchor="middle">Focal Loss (γ=2.0) + Supervised Margin Loss (m=1.5)</text>
      <text x="1365" y="1312" font-size="10.5" fill="#7f1d1d" text-anchor="middle">Enforces clear metric distance between pristine and fake manifolds</text>
    </g>

    <!-- Loss 2: Emotion Head Losses -->
    <g filter="url(#node-shadow)">
      <rect x="1145" y="1340" width="440" height="70" rx="8" fill="url(#grad-emo)" stroke="#eab308" stroke-width="1.5"/>
      <text x="1365" y="1363" font-size="13" font-weight="700" fill="#854d0e" text-anchor="middle">Affect Losses (L_emoA, L_emoB)</text>
      <text x="1365" y="1381" font-size="11.5" fill="#a16207" text-anchor="middle">Categorical Cross-Entropy with Label Smoothing (0.1)</text>
      <text x="1365" y="1396" font-size="10.5" fill="#713f12" text-anchor="middle">Supervises acoustic and facial emotional classifications</text>
    </g>

    <!-- Loss 3: Sarcasm Loss -->
    <g filter="url(#node-shadow)">
      <rect x="1145" y="1420" width="440" height="65" rx="8" fill="url(#grad-sarc)" stroke="#eab308" stroke-width="1.5"/>
      <text x="1365" y="1443" font-size="13" font-weight="700" fill="#854d0e" text-anchor="middle">Sarcasm Loss (L_sarc)</text>
      <text x="1365" y="1461" font-size="11.5" fill="#a16207" text-anchor="middle">Binary Cross-Entropy Loss with Logits</text>
      <text x="1365" y="1475" font-size="10.5" fill="#713f12" text-anchor="middle">Guards against acoustic-lexical sarcasm false positives</text>
    </g>

    <!-- Loss 4: Domain Adversarial Loss -->
    <g filter="url(#node-shadow)">
      <rect x="1145" y="1495" width="440" height="70" rx="8" fill="url(#grad-purple)" stroke="#a855f7" stroke-width="1.5"/>
      <text x="1365" y="1518" font-size="13" font-weight="700" fill="#6b21a8" text-anchor="middle">Domain Invariance Loss (L_domain)</text>
      <text x="1365" y="1536" font-size="11.5" fill="#7e22ce" text-anchor="middle">Gradient Reversal Layer (GRL, α=1.0) → 5-Domain Classifier</text>
      <text x="1365" y="1551" font-size="10.5" fill="#581c87" text-anchor="middle">Prevents shortcut learning across distinct datasets</text>
    </g>

    <!-- Loss 5: Calibration Loss -->
    <g filter="url(#node-shadow)">
      <rect x="1145" y="1575" width="440" height="60" rx="8" fill="#1e293b" stroke="#64748b" stroke-width="1.5"/>
      <text x="1365" y="1598" font-size="12.5" font-weight="700" fill="#f8fafc" text-anchor="middle">Forensic Calibration Loss (Brier + ECE)</text>
      <text x="1365" y="1616" font-size="11" fill="#94a3b8" text-anchor="middle">Minimizes Expected Calibration Error for court-admissible confidence</text>
    </g>

    <!-- Multi-Task Aggregator Formula -->
    <rect x="1145" y="1645" width="440" height="35" rx="6" fill="#581c87" stroke="#a855f7" stroke-width="1.2"/>
    <text x="1365" y="1668" font-size="12" font-weight="700" fill="#ffffff" text-anchor="middle">ℒ_total = λ₁ L_det + λ₂ (L_emoA + L_emoB) + λ₃ L_sarc + λ₄ L_domain</text>


    <!-- Optimization & Training Execution -->
    <path d="M 1365 1690 L 1365 1730" fill="none" stroke="#a855f7" stroke-width="2.5" marker-end="url(#arr-purple)"/>

    <g filter="url(#node-shadow)">
      <rect x="1125" y="1730" width="480" height="175" rx="12" fill="#0f172a" stroke="#a855f7" stroke-width="1.5"/>
      <text x="1365" y="1758" font-size="14" font-weight="800" fill="#c084fc" text-anchor="middle">Backpropagation &amp; Optimization Strategy</text>

      <!-- Phase 1 Card -->
      <rect x="1145" y="1775" width="210" height="110" rx="8" fill="#1e293b" stroke="#64748b" stroke-width="1.2"/>
      <text x="1250" y="1800" font-size="12" font-weight="700" fill="#f8fafc" text-anchor="middle">Phase 1: Feature Training</text>
      <text x="1250" y="1820" font-size="11" font-weight="600" fill="#4ade80" text-anchor="middle">Backbones Frozen</text>
      <text x="1250" y="1838" font-size="10.5" fill="#cbd5e1" text-anchor="middle">Rapid convergence of</text>
      <text x="1250" y="1853" font-size="10.5" fill="#cbd5e1" text-anchor="middle">CBP, Heads &amp; Classifier</text>
      <text x="1250" y="1870" font-size="10" fill="#94a3b8" text-anchor="middle">AdamW (lr=1e-4)</text>

      <!-- Phase 2 Card -->
      <rect x="1375" y="1775" width="210" height="110" rx="8" fill="#1e293b" stroke="#64748b" stroke-width="1.2"/>
      <text x="1480" y="1800" font-size="12" font-weight="700" fill="#f8fafc" text-anchor="middle">Phase 2: Fine-Tuning</text>
      <text x="1480" y="1820" font-size="11" font-weight="600" fill="#f87171" text-anchor="middle">Top-2 Layers Unfrozen</text>
      <text x="1480" y="1838" font-size="10.5" fill="#cbd5e1" text-anchor="middle">End-to-End Alignment of</text>
      <text x="1480" y="1853" font-size="10.5" fill="#cbd5e1" text-anchor="middle">Wav2Vec, BERT &amp; ViT</text>
      <text x="1480" y="1870" font-size="10" fill="#94a3b8" text-anchor="middle">Cosine Annealing (lr=1e-5)</text>
    </g>

    <!-- Training Deliverables Box -->
    <g filter="url(#node-shadow)">
      <rect x="1125" y="1925" width="480" height="120" rx="12" fill="#020617" stroke="#a855f7" stroke-width="1.5"/>
      <text x="1365" y="1955" font-size="14" font-weight="800" fill="#c084fc" text-anchor="middle">Checkpoint Artifacts &amp; Telemetry</text>
      <text x="1365" y="1978" font-size="12" font-weight="600" fill="#e9d5ff" text-anchor="middle">Best Metric Checkpoints (.pt) • TensorBoard Event Logs</text>
      <text x="1365" y="1998" font-size="11.5" fill="#d8b4fe" text-anchor="middle">Cross-Dataset Generalization Validation (AUC-ROC &gt; 92%)</text>
      <text x="1365" y="2018" font-size="11" fill="#a855f7" text-anchor="middle">Real-Time WebSocket Inference Metrics for WebApp HUD</text>
    </g>
  </g>

</svg>'''
    return svg

def main():
    svg_content = create_system_architecture_svg()
    out_path = Path(r"d:\Internship\DeepSentinel\docs\complete_system_architecture.svg")
    out_path.write_text(svg_content, encoding="utf-8")
    print(f"Successfully generated: {out_path}")

if __name__ == "__main__":
    main()
