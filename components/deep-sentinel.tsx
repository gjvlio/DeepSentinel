"use client"

import { useState, useEffect, useCallback } from "react"
import { Upload, Check, FileVideo } from "lucide-react"
import { cn } from "@/lib/utils"

type Screen = "upload" | "processing" | "result"
type ResultType = "real" | "fake"

const PROCESSING_STEPS = [
  "Extracting audio waveform",
  "Running Wav2Vec 2.0 + BERT encoder → Z_at",
  "Extracting video keyframes (Top-8)",
  "Running ViT encoder → Z_v",
  "Computing emotion heads A + B",
  "Computing discrepancy score Δ",
  "Bilinear fusion + classifier → P(fake)",
]

type DeltaItem = {
  emotion: string
  delta: number
  severity: "Low" | "Moderate" | "High"
}

const REAL_DATA = {
  score: 12,
  label: "Likely authentic",
  subtitle: "Emotions are consistent across audio and visual modalities.",
  mismatch: {
    emotion: "Happy",
    delta: 0.08,
    description: "audio says happy, face says happy · low mismatch signal",
    percentage: 8,
  },
  headA: [
    { emotion: "Angry", value: 0.03, highlight: false },
    { emotion: "Happy", value: 0.72, highlight: true },
    { emotion: "Sad", value: 0.06, highlight: false },
    { emotion: "Neutral", value: 0.17, highlight: false },
    { emotion: "Fearful", value: 0.01, highlight: false },
    { emotion: "Disgust", value: 0.01, highlight: false },
  ],
  headB: [
    { emotion: "Angry", value: 0.02, highlight: false },
    { emotion: "Happy", value: 0.89, highlight: true },
    { emotion: "Sad", value: 0.04, highlight: false },
    { emotion: "Neutral", value: 0.03, highlight: false },
    { emotion: "Fearful", value: 0.01, highlight: false },
    { emotion: "Disgust", value: 0.01, highlight: false },
  ],
  deltas: [
    { emotion: "Angry", delta: 0.01, severity: "Low" },
    { emotion: "Happy", delta: 0.08, severity: "Low" },
    { emotion: "Sad", delta: 0.02, severity: "Low" },
    { emotion: "Neutral", delta: 0.04, severity: "Low" },
    { emotion: "Fearful", delta: 0.01, severity: "Low" },
    { emotion: "Disgust", delta: 0.01, severity: "Low" },
  ] as DeltaItem[],
}

const FAKE_DATA = {
  score: 87,
  label: "Likely deepfake",
  subtitle: "Strong emotional mismatch detected across modalities.",
  mismatch: {
    emotion: "Angry",
    delta: 0.65,
    description: "audio says angry, face says happy · high fake signal",
    percentage: 65,
  },
  headA: [
    { emotion: "Angry", value: 0.78, highlight: true },
    { emotion: "Happy", value: 0.10, highlight: false },
    { emotion: "Sad", value: 0.05, highlight: false },
    { emotion: "Neutral", value: 0.04, highlight: false },
    { emotion: "Fearful", value: 0.02, highlight: false },
    { emotion: "Disgust", value: 0.01, highlight: false },
  ],
  headB: [
    { emotion: "Angry", value: 0.05, highlight: false },
    { emotion: "Happy", value: 0.88, highlight: true },
    { emotion: "Sad", value: 0.03, highlight: false },
    { emotion: "Neutral", value: 0.02, highlight: false },
    { emotion: "Fearful", value: 0.01, highlight: false },
    { emotion: "Disgust", value: 0.01, highlight: false },
  ],
  deltas: [
    { emotion: "Angry", delta: 0.12, severity: "Low" },
    { emotion: "Happy", delta: 0.72, severity: "High" },
    { emotion: "Sad", delta: 0.05, severity: "Low" },
    { emotion: "Neutral", delta: 0.41, severity: "Moderate" },
    { emotion: "Fearful", delta: 0.08, severity: "Low" },
    { emotion: "Disgust", delta: 0.15, severity: "Low" },
  ] as DeltaItem[],
}

function GeometricFace() {
  return (
    <svg
      viewBox="0 0 200 240"
      className="w-48 h-56 opacity-[0.08]"
      fill="none"
      stroke="currentColor"
      strokeWidth="0.8"
    >
      {/* Head outline */}
      <path d="M100 10 L140 30 L160 70 L165 120 L155 170 L130 210 L100 225 L70 210 L45 170 L35 120 L40 70 L60 30 Z" />
      
      {/* Forehead facets */}
      <path d="M100 10 L120 25 L100 45 L80 25 Z" />
      <path d="M120 25 L140 30 L135 55 L100 45 Z" />
      <path d="M80 25 L100 45 L65 55 L60 30 Z" />
      
      {/* Upper face */}
      <path d="M135 55 L160 70 L155 95 L130 85 L100 90 Z" />
      <path d="M65 55 L100 90 L70 85 L45 95 L40 70 Z" />
      <path d="M100 45 L135 55 L100 90 L65 55 Z" />
      
      {/* Eyes area */}
      <path d="M75 85 L95 82 L95 100 L75 102 Z" />
      <path d="M105 82 L125 85 L125 102 L105 100 Z" />
      
      {/* Nose */}
      <path d="M95 100 L105 100 L108 130 L100 140 L92 130 Z" />
      
      {/* Cheeks */}
      <path d="M45 95 L70 85 L75 120 L55 140 L35 120 Z" />
      <path d="M155 95 L165 120 L145 140 L125 120 L130 85 Z" />
      
      {/* Mouth area */}
      <path d="M75 140 L92 130 L100 140 L108 130 L125 140 L115 160 L100 165 L85 160 Z" />
      
      {/* Lower face */}
      <path d="M55 140 L75 140 L85 160 L70 180 L45 170 Z" />
      <path d="M125 140 L145 140 L155 170 L130 180 L115 160 Z" />
      
      {/* Chin */}
      <path d="M70 180 L85 160 L100 165 L115 160 L130 180 L130 210 L100 225 L70 210 Z" />
      
      {/* Jaw lines */}
      <path d="M45 170 L70 180 L70 210 Z" />
      <path d="M155 170 L130 210 L130 180 Z" />
      
      {/* Ear left */}
      <path d="M35 90 L25 105 L25 130 L35 140 L45 130 L45 95 Z" />
      
      {/* Ear right */}
      <path d="M165 90 L175 105 L175 130 L165 140 L155 130 L155 95 Z" />
      
      {/* Additional facet details */}
      <line x1="100" y1="90" x2="100" y2="140" />
      <line x1="75" y1="120" x2="125" y2="120" />
      <line x1="85" y1="95" x2="85" y2="105" />
      <line x1="115" y1="95" x2="115" y2="105" />
    </svg>
  )
}

function BackgroundShapes() {
  return (
    <div className="absolute inset-0 overflow-hidden pointer-events-none">
      {/* Floating geometric shapes */}
      <div 
        className="absolute top-20 left-10 w-32 h-32 border border-gray-200/40 rotate-45"
        style={{ borderRadius: "30% 70% 70% 30% / 30% 30% 70% 70%" }}
      />
      <div 
        className="absolute top-40 right-20 w-24 h-24 border border-violet-200/30 rotate-12"
      />
      <div 
        className="absolute bottom-32 left-1/4 w-16 h-16 border border-green-200/30 rotate-[60deg]"
      />
      <div 
        className="absolute top-1/3 right-1/4 w-20 h-20 border border-gray-200/30"
        style={{ clipPath: "polygon(50% 0%, 100% 50%, 50% 100%, 0% 50%)" }}
      />
      <div 
        className="absolute bottom-40 right-10 w-28 h-28 border border-violet-200/20 rotate-[30deg]"
        style={{ borderRadius: "60% 40% 30% 70% / 60% 30% 70% 40%" }}
      />
      {/* Subtle grid lines */}
      <div className="absolute inset-0 opacity-[0.03]" style={{
        backgroundImage: `
          linear-gradient(to right, #000 1px, transparent 1px),
          linear-gradient(to bottom, #000 1px, transparent 1px)
        `,
        backgroundSize: "60px 60px"
      }} />
    </div>
  )
}

function AuroraSpinner() {
  return (
    <div className="relative w-16 h-16">
      <div className="absolute inset-0 rounded-full animate-spin" style={{
        background: "conic-gradient(from 0deg, #4ade80, #86efac, #a78bfa, #7c3aed, #4ade80)",
        animationDuration: "1.5s",
      }} />
      <div className="absolute inset-1 rounded-full bg-white" />
      <div className="absolute inset-2 rounded-full animate-spin" style={{
        background: "conic-gradient(from 180deg, #4ade80, #86efac, #a78bfa, #7c3aed, #4ade80)",
        animationDuration: "2s",
        animationDirection: "reverse",
      }} />
      <div className="absolute inset-3 rounded-full bg-white" />
    </div>
  )
}

function EmotionBar({ emotion, value, highlight, type }: { 
  emotion: string
  value: number
  highlight: boolean
  type: "audio" | "visual"
}) {
  const barColor = highlight 
    ? type === "audio" ? "bg-violet-500" : "bg-green-500"
    : "bg-gray-200"
  
  return (
    <div className="flex items-center gap-3">
      <span className="text-xs text-gray-500 w-14 text-right">{emotion}</span>
      <div className="flex-1 h-2 bg-gray-100 rounded-full overflow-hidden">
        <div 
          className={cn("h-full rounded-full transition-all duration-500", barColor)}
          style={{ width: `${value * 100}%` }}
        />
      </div>
      <span className="text-xs text-gray-400 w-8">{value.toFixed(2)}</span>
    </div>
  )
}

function DeltaBar({ emotion, delta, severity, animate }: { 
  emotion: string
  delta: number
  severity: "Low" | "Moderate" | "High"
  animate: boolean
}) {
  const [width, setWidth] = useState(0)
  
  useEffect(() => {
    if (animate) {
      const timer = setTimeout(() => {
        setWidth(Math.min(delta * 100, 100))
      }, 100)
      return () => clearTimeout(timer)
    }
  }, [animate, delta])

  const severityColors = {
    Low: "bg-teal-500 text-teal-700 bg-teal-50",
    Moderate: "bg-amber-500 text-amber-700 bg-amber-50",
    High: "bg-rose-500 text-rose-700 bg-rose-50",
  }

  const barColor = {
    Low: "from-teal-400 to-green-400",
    Moderate: "from-amber-400 to-orange-400",
    High: "from-rose-400 to-pink-400",
  }
  
  return (
    <div className="flex items-center gap-3">
      <span className="text-xs text-gray-600 w-16 font-medium">{emotion}</span>
      <span className="text-xs text-gray-400 w-10 text-right">{delta.toFixed(2)}</span>
      <div className="flex-1 h-2.5 bg-gray-100 rounded-full overflow-hidden">
        <div 
          className={cn("h-full rounded-full bg-gradient-to-r transition-all duration-[600ms] ease-out", barColor[severity])}
          style={{ width: `${width}%` }}
        />
      </div>
      <span className={cn(
        "text-[10px] font-medium px-2 py-0.5 rounded-full w-16 text-center",
        severity === "Low" && "bg-teal-50 text-teal-700",
        severity === "Moderate" && "bg-amber-50 text-amber-700",
        severity === "High" && "bg-rose-50 text-rose-700",
      )}>
        {severity}
      </span>
    </div>
  )
}

export function DeepSentinel() {
  const [screen, setScreen] = useState<Screen>("upload")
  const [fileName, setFileName] = useState<string | null>(null)
  const [completedSteps, setCompletedSteps] = useState<number>(0)
  const [runCount, setRunCount] = useState(0)
  const [isTransitioning, setIsTransitioning] = useState(false)
  const [animateDeltas, setAnimateDeltas] = useState(false)

  const resultType: ResultType = runCount % 2 === 0 ? "real" : "fake"
  const resultData = resultType === "real" ? REAL_DATA : FAKE_DATA

  const handleFileSelect = useCallback(() => {
    setFileName("sample_video.mp4 · 156 MB")
  }, [])

  const handleRunAnalysis = useCallback(() => {
    setIsTransitioning(true)
    setTimeout(() => {
      setScreen("processing")
      setCompletedSteps(0)
      setAnimateDeltas(false)
      setIsTransitioning(false)
    }, 300)
  }, [])

  const handleAnalyzeAnother = useCallback(() => {
    setIsTransitioning(true)
    setTimeout(() => {
      setScreen("upload")
      setFileName(null)
      setRunCount(prev => prev + 1)
      setAnimateDeltas(false)
      setIsTransitioning(false)
    }, 300)
  }, [])

  useEffect(() => {
    if (screen === "processing") {
      const interval = setInterval(() => {
        setCompletedSteps(prev => {
          if (prev >= PROCESSING_STEPS.length) {
            clearInterval(interval)
            setTimeout(() => {
              setIsTransitioning(true)
              setTimeout(() => {
                setScreen("result")
                setIsTransitioning(false)
                // Trigger delta animations after result screen appears
                setTimeout(() => setAnimateDeltas(true), 100)
              }, 300)
            }, 400)
            return prev
          }
          return prev + 1
        })
      }, 400)
      return () => clearInterval(interval)
    }
  }, [screen])

  return (
    <div className="min-h-screen bg-[#fafafa] px-4 py-8 relative">
      {/* Header */}
      <header className="max-w-4xl mx-auto flex items-center justify-between mb-12 relative z-10">
        <div className="flex items-center gap-2">
          <div 
            className="w-3 h-3 rounded-full"
            style={{
              background: "linear-gradient(135deg, #4ade80 0%, #a78bfa 50%, #7c3aed 100%)",
              boxShadow: "0 0 12px rgba(124, 58, 237, 0.4), 0 0 12px rgba(74, 222, 128, 0.4)",
            }}
          />
          <h1 className="text-xl font-semibold text-gray-900">DeepSentinel</h1>
        </div>
        <span className="px-3 py-1 text-xs font-medium text-gray-500 bg-gray-100 rounded-full">
          Research prototype
        </span>
      </header>

      {/* Main Content */}
      <main className={cn(
        "max-w-4xl mx-auto transition-all duration-300 relative z-10",
        isTransitioning ? "opacity-0 translate-y-2" : "opacity-100 translate-y-0"
      )}>
        {/* Upload Screen */}
        {screen === "upload" && (
          <div className="flex flex-col items-center relative">
            <BackgroundShapes />
            
            <div 
              className="relative p-[2px] rounded-2xl w-full max-w-xl"
              style={{
                background: "linear-gradient(135deg, #4ade80 0%, #86efac 25%, #a78bfa 75%, #7c3aed 100%)",
                boxShadow: "0 0 40px rgba(74, 222, 128, 0.15), 0 0 40px rgba(124, 58, 237, 0.15)",
              }}
            >
              <div 
                className="bg-white/95 backdrop-blur-xl rounded-2xl p-8 cursor-pointer relative overflow-hidden"
                onClick={handleFileSelect}
              >
                {/* Geometric face watermark */}
                <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 text-gray-900">
                  <GeometricFace />
                </div>
                
                {!fileName ? (
                  <div className="flex flex-col items-center py-12 relative z-10">
                    <div className="w-16 h-16 rounded-full bg-gradient-to-br from-green-50 to-violet-50 flex items-center justify-center mb-4">
                      <Upload className="w-7 h-7 text-gray-400" />
                    </div>
                    <p className="text-gray-700 font-medium mb-1">
                      Drop your video here
                    </p>
                    <p className="text-gray-400 text-sm mb-4">
                      or click to browse files
                    </p>
                    <p className="text-xs text-gray-400">
                      Max 10 seconds · MP4, MOV, WEBM
                    </p>
                  </div>
                ) : (
                  <div className="flex flex-col items-center py-12 relative z-10">
                    <div className="w-16 h-16 rounded-full bg-gradient-to-br from-green-100 to-violet-100 flex items-center justify-center mb-4">
                      <FileVideo className="w-7 h-7 text-violet-500" />
                    </div>
                    <p className="text-gray-700 font-medium">
                      {fileName}
                    </p>
                  </div>
                )}
              </div>
            </div>

            {fileName && (
              <button
                onClick={handleRunAnalysis}
                className="mt-6 px-8 py-3 rounded-xl font-medium text-white transition-all hover:scale-105 active:scale-95"
                style={{
                  background: "linear-gradient(135deg, #ec4899 0%, #a78bfa 50%, #7c3aed 100%)",
                  boxShadow: "0 4px 20px rgba(124, 58, 237, 0.3)",
                }}
              >
                Run Analysis
              </button>
            )}
          </div>
        )}

        {/* Processing Screen */}
        {screen === "processing" && (
          <div className="flex flex-col items-center">
            <AuroraSpinner />
            <h2 className="text-xl font-semibold text-gray-900 mt-6 mb-1">
              Analyzing video
            </h2>
            <p className="text-sm text-gray-500 mb-8">{fileName}</p>

            <div className="w-full max-w-md space-y-3">
              {PROCESSING_STEPS.map((step, index) => (
                <div 
                  key={step}
                  className={cn(
                    "flex items-center gap-3 transition-all duration-300",
                    index < completedSteps ? "opacity-100" : "opacity-40"
                  )}
                >
                  <div className={cn(
                    "w-5 h-5 rounded-full flex items-center justify-center transition-colors duration-300",
                    index < completedSteps ? "bg-green-500" : "bg-gray-200"
                  )}>
                    {index < completedSteps && (
                      <Check className="w-3 h-3 text-white" />
                    )}
                  </div>
                  <span className="text-sm text-gray-600">{step}</span>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Result Screen */}
        {screen === "result" && (
          <div className="space-y-6">
            {/* Main Result Card */}
            <div 
              className="relative rounded-2xl p-8 backdrop-blur-xl"
              style={{
                background: resultType === "real" 
                  ? "linear-gradient(135deg, #dcfce7 0%, #f0fdf4 100%)"
                  : "linear-gradient(135deg, #ede9fe 0%, #faf5ff 100%)",
                boxShadow: "0 4px 24px rgba(0, 0, 0, 0.06)",
              }}
            >
              <div className="absolute top-6 right-6">
                <span className={cn(
                  "px-3 py-1 text-xs font-medium rounded-full",
                  resultType === "real" 
                    ? "bg-green-100 text-green-700"
                    : "bg-violet-100 text-violet-700"
                )}>
                  {resultType === "real" ? "Real" : "Fake"}
                </span>
              </div>

              <div className="text-center">
                <p className={cn(
                  "text-7xl font-bold mb-2",
                  resultType === "real" ? "text-green-600" : "text-violet-600"
                )}>
                  {resultData.score}%
                </p>
                <p className={cn(
                  "text-lg font-medium mb-1",
                  resultType === "real" ? "text-green-600" : "text-violet-600"
                )}>
                  {resultData.label}
                </p>
                <p className="text-sm text-gray-500">
                  {resultData.subtitle}
                </p>
              </div>
            </div>

            {/* Mismatch Bar */}
            <div className="bg-white/80 backdrop-blur-xl rounded-xl p-5 shadow-sm border border-gray-100">
              <div className="flex items-center justify-between mb-3">
                <span className="text-sm font-medium text-gray-700">
                  Dominant mismatch · {resultData.mismatch.emotion}
                </span>
                <span className="text-xs text-gray-400">
                  Δ = {resultData.mismatch.delta.toFixed(2)}
                </span>
              </div>
              <div className="h-3 bg-gray-100 rounded-full overflow-hidden mb-2">
                <div 
                  className="h-full rounded-full transition-all duration-700"
                  style={{
                    width: `${resultData.mismatch.percentage}%`,
                    background: resultType === "real"
                      ? "linear-gradient(90deg, #4ade80 0%, #86efac 100%)"
                      : "linear-gradient(90deg, #a78bfa 0%, #f59e0b 100%)",
                  }}
                />
              </div>
              <p className="text-xs text-gray-400">
                {resultData.mismatch.description}
              </p>
            </div>

            {/* Emotion Heads */}
            <div className="grid md:grid-cols-2 gap-4">
              {/* Head A - Audio */}
              <div className="bg-white/80 backdrop-blur-xl rounded-xl p-5 shadow-sm border border-gray-100">
                <div className="mb-4">
                  <h3 className="text-sm font-semibold text-gray-700">Head A — audio</h3>
                  <p className="text-xs text-gray-400">what the voice expresses</p>
                </div>
                <div className="space-y-2.5">
                  {resultData.headA.map(item => (
                    <EmotionBar 
                      key={item.emotion} 
                      emotion={item.emotion} 
                      value={item.value} 
                      highlight={item.highlight}
                      type="audio"
                    />
                  ))}
                </div>
              </div>

              {/* Head B - Visual */}
              <div className="bg-white/80 backdrop-blur-xl rounded-xl p-5 shadow-sm border border-gray-100">
                <div className="mb-4">
                  <h3 className="text-sm font-semibold text-gray-700">Head B — visual</h3>
                  <p className="text-xs text-gray-400">what the face shows</p>
                </div>
                <div className="space-y-2.5">
                  {resultData.headB.map(item => (
                    <EmotionBar 
                      key={item.emotion} 
                      emotion={item.emotion} 
                      value={item.value} 
                      highlight={item.highlight}
                      type="visual"
                    />
                  ))}
                </div>
              </div>
            </div>

            {/* Delta per emotion class */}
            <div className="bg-white/80 backdrop-blur-xl rounded-xl p-5 shadow-sm border border-gray-100">
              <div className="mb-4">
                <h3 className="text-sm font-semibold text-gray-700">
                  <span className="text-violet-500">Δ</span> per emotion class
                </h3>
              </div>
              <div className="space-y-3">
                {resultData.deltas.map(item => (
                  <DeltaBar 
                    key={item.emotion}
                    emotion={item.emotion}
                    delta={item.delta}
                    severity={item.severity}
                    animate={animateDeltas}
                  />
                ))}
              </div>
            </div>

            {/* Analyze Another Button */}
            <div className="flex justify-center pt-4">
              <button
                onClick={handleAnalyzeAnother}
                className="px-6 py-2.5 rounded-xl font-medium text-gray-600 border border-gray-200 bg-white hover:bg-gray-50 transition-all hover:scale-105 active:scale-95"
              >
                Analyze another
              </button>
            </div>

            {/* Disclaimer Footer */}
            <div className="bg-gray-100/80 rounded-xl py-3 px-4 text-center">
              <p className="text-xs text-gray-400">
                Research prototype. Do not use as sole evidence in legal or journalistic contexts.
              </p>
            </div>
          </div>
        )}
      </main>
    </div>
  )
}
