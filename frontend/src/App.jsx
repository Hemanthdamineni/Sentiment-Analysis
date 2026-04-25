import React, { useState, useEffect } from 'react'

function App() {
  const [showLanding, setShowLanding] = useState(true)
  const [text, setText] = useState('')
  const [modelType, setModelType] = useState('transformer')
  const [algorithm, setAlgorithm] = useState('')
  const [availableAlgorithms, setAvailableAlgorithms] = useState([])
  const [prediction, setPrediction] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [comparison, setComparison] = useState(null)
  const [showComparison, setShowComparison] = useState(false)
  const [wordclouds, setWordclouds] = useState([])
  const [showWordclouds, setShowWordclouds] = useState(false)

  const STOP_WORDS = new Set([
    'a','an','the','and','or','but','if','while','with','without','to','from','of','in','on','for','by','as','at','into','about','than','then','so','too','very','can','could','should','would','may','might','must','do','does','did','doing','done','is','am','are','was','were','be','been','being','have','has','had','having','will','shall','it','its','itself','this','that','these','those','there','here','not','no','nor','only','own','same','such','also','just','both','all','any','each','few','more','most','other','some','much','many','you','your','yours','yourself','yourselves','i','me','my','myself','we','our','ours','ourselves','he','him','his','himself','she','her','hers','herself','they','them','their','theirs','themselves'
  ])

  const cleanText = (input) => {
    const tokens = (input.toLowerCase().match(/[a-z]+/g) || [])
    const filtered = tokens.filter(t => t.length > 2 && !STOP_WORDS.has(t))
    return filtered.join(' ')
  }

  const API_BASE = 'http://localhost:8000'

  useEffect(() => {
    if (!showLanding) {
      fetchAvailableModels()
      fetchComparison()
      fetchWordclouds()
    }
  }, [showLanding])

  const fetchAvailableModels = async () => {
    try {
      const res = await fetch(`${API_BASE}/models`)
      const data = await res.json()
      setAvailableAlgorithms(data.classical_algorithms)
      if (data.classical_algorithms.length > 0) {
        setAlgorithm(data.classical_algorithms[0])
      }
    } catch (err) {
      console.error('Failed to fetch models:', err)
    }
  }

  const fetchComparison = async () => {
    try {
      const res = await fetch(`${API_BASE}/comparison`)
      const data = await res.json()
      setComparison(data)
    } catch (err) {
      console.error('Failed to fetch comparison:', err)
    }
  }

  const fetchWordclouds = async () => {
    try {
      const res = await fetch(`${API_BASE}/wordclouds`)
      const data = await res.json()
      setWordclouds(data.wordclouds || [])
    } catch (err) {
      console.error('Failed to fetch wordclouds:', err)
    }
  }

  const handlePredict = async () => {
    if (!text.trim()) {
      setError('Please enter some text')
      return
    }

    const cleaned = cleanText(text)
    if (!cleaned.trim()) {
      setError('Please enter valid review to analyze.')
      return
    }

    setLoading(true)
    setError(null)
    setPrediction(null)

    try {
      const res = await fetch(`${API_BASE}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text,
          model_type: modelType,
          algorithm: modelType === 'classical' ? algorithm : null
        })
      })

      if (!res.ok) {
        const errData = await res.json()
        throw new Error(errData.detail || 'Prediction failed')
      }

      const data = await res.json()
      setPrediction(data)
      
      if (!data.valid) {
        setError(data.message || 'Invalid input')
      }
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const getSentimentColor = (sentiment) => {
    const s = sentiment?.toLowerCase()
    if (s === 'positive') return 'text-emerald-600'
    if (s === 'negative') return 'text-rose-600'
    if (s === 'neutral') return 'text-amber-600'
    return 'text-gray-600'
  }

  const getSentimentBg = (sentiment) => {
    const s = sentiment?.toLowerCase()
    if (s === 'positive') return 'bg-emerald-50 border-emerald-300'
    if (s === 'negative') return 'bg-rose-50 border-rose-300'
    if (s === 'neutral') return 'bg-amber-50 border-amber-300'
    return 'bg-gray-50 border-gray-200'
  }

  

  const formatAlgorithmName = (name) => {
    return name.replace(/_/g, ' ').split(' ').map(word => 
      word.charAt(0).toUpperCase() + word.slice(1)
    ).join(' ')
  }

  if (showLanding) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 relative overflow-hidden">
        {/* Animated background elements */}
        <div className="absolute inset-0 overflow-hidden">
          <div className="absolute w-96 h-96 bg-purple-500 rounded-full mix-blend-multiply filter blur-3xl opacity-20 animate-pulse top-0 -left-48"></div>
          <div className="absolute w-96 h-96 bg-blue-500 rounded-full mix-blend-multiply filter blur-3xl opacity-20 animate-pulse bottom-0 -right-48 animation-delay-2000"></div>
          <div className="absolute w-96 h-96 bg-pink-500 rounded-full mix-blend-multiply filter blur-3xl opacity-20 animate-pulse top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 animation-delay-4000"></div>
        </div>

        <div className="relative z-10 flex flex-col items-center justify-center min-h-screen px-4 py-12">
          

          {/* Main heading */}
          <h1 className="text-5xl md:text-7xl font-bold text-white text-center mb-6 bg-clip-text text-transparent bg-gradient-to-r from-purple-400 via-pink-400 to-blue-400">
            Smartwatch Sentiment Analyzer
          </h1>

          {/* Subtitle */}
          <p className="text-xl md:text-2xl text-gray-300 text-center mb-4 max-w-2xl">
            Advanced AI-Powered Review Analysis
          </p>
          
          <p className="text-md text-gray-400 text-center mb-12 max-w-xl">
            Compare cutting-edge Transformer models with classical ML algorithms for accurate sentiment detection
          </p>

          {/* Feature cards */}
          <div className="grid md:grid-cols-3 gap-6 mb-12 max-w-4xl w-full">
            <div className="bg-white/10 backdrop-blur-lg rounded-xl p-6 border border-white/20 hover:bg-white/15 transition-all transform hover:scale-105">
              <h3 className="text-white font-semibold text-lg mb-2">🤖 DISTILBERT TRANSFORMER</h3>
              <p className="text-gray-300 text-sm">State-of-the-art deep learning for nuanced sentiment analysis</p>
            </div>
            
            <div className="bg-white/10 backdrop-blur-lg rounded-xl p-6 border border-white/20 hover:bg-white/15 transition-all transform hover:scale-105">
              <h3 className="text-white font-semibold text-lg mb-2">📊 CLASSICAL MODELS</h3>
              <p className="text-gray-300 text-sm">Compare performance across traditional ML algorithms</p>
            </div>
            
            <div className="bg-white/10 backdrop-blur-lg rounded-xl p-6 border border-white/20 hover:bg-white/15 transition-all transform hover:scale-105">
              <h3 className="text-white font-semibold text-lg mb-2">⚡ REAL-TIME ANALYSIS</h3>
              <p className="text-gray-300 text-sm">Instant predictions with confidence scores and insights</p>
            </div>
          </div>

          {/* CTA Button */}
          <button
            onClick={() => setShowLanding(false)}
            className="group relative px-8 py-4 bg-gradient-to-r from-purple-600 to-blue-600 text-white text-lg font-semibold rounded-full shadow-2xl hover:shadow-purple-500/50 transition-all transform hover:scale-105 active:scale-95"
          >
            <span className="relative z-10 flex items-center gap-2">
              Get Started
              <svg className="w-5 h-5 group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7l5 5m0 0l-5 5m5-5H6" />
              </svg>
            </span>
          </button>

          {/* Footer */}
          <div className="mt-16 text-center">
            <p className="text-gray-400 text-sm">Built with FastAPI, React, and Tailwind CSS</p>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-50">
      {/* Header with back button */}
      <div className="bg-white/80 backdrop-blur-md shadow-sm border-b border-gray-200 sticky top-0 z-50">
        <div className="container mx-auto px-4 py-4 max-w-7xl flex items-center justify-between">
          <button
            onClick={() => setShowLanding(true)}
            className="flex items-center gap-2 text-gray-600 hover:text-gray-900 transition-colors"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 19l-7-7m0 0l7-7m-7 7h18" />
            </svg>
            <span className="font-medium">Back</span>
          </button>
          <div className="flex items-center gap-3">
            <div>
              <h1 className="text-xl font-bold text-gray-800">Sentiment Analyzer</h1>
              <p className="text-xs text-gray-500">AI-Powered Analysis</p>
            </div>
          </div>
          <div className="w-20"></div>
        </div>
      </div>

      <div className="container mx-auto px-4 py-8 max-w-7xl">
        {/* Main Content */}
        <div className="grid lg:grid-cols-3 gap-6">
          {/* Left Panel - Input (2 columns) */}
          <div className="lg:col-span-2 space-y-6">
            <div className="bg-white rounded-2xl shadow-xl p-8 border border-gray-100">
              <div className="flex items-center gap-3 mb-6">
                <div className="w-12 h-12 bg-gradient-to-br from-blue-500 to-indigo-600 rounded-xl flex items-center justify-center">
                  <span className="text-2xl">🔍</span>
                </div>
                <div>
                  <h2 className="text-2xl font-bold text-gray-800">Analyze Review</h2>
                  <p className="text-sm text-gray-500">Choose a model and enter text to analyze</p>
                </div>
              </div>

              {/* Model Selection */}
              <div className="mb-6">
                <label className="block text-sm font-semibold text-gray-700 mb-3">
                  Select Model Type
                </label>
                <div className="grid grid-cols-2 gap-4">
                  <button
                    onClick={() => setModelType('transformer')}
                    className={`p-5 rounded-xl border-2 transition-all ${
                      modelType === 'transformer'
                        ? 'border-blue-500 bg-gradient-to-br from-blue-50 to-indigo-50 shadow-lg scale-105'
                        : 'border-gray-200 hover:border-gray-300 hover:shadow-md'
                    }`}
                  >
                    <div className="font-bold text-gray-800">Transformer</div>
                    <div className="text-xs text-gray-500 mt-1">DistilBERT Model</div>
                  </button>
                  <button
                    onClick={() => setModelType('classical')}
                    className={`p-5 rounded-xl border-2 transition-all ${
                      modelType === 'classical'
                        ? 'border-blue-500 bg-gradient-to-br from-blue-50 to-indigo-50 shadow-lg scale-105'
                        : 'border-gray-200 hover:border-gray-300 hover:shadow-md'
                    }`}
                  >
                    <div className="font-bold text-gray-800">Classical ML</div>
                    <div className="text-xs text-gray-500 mt-1">Traditional Algorithms</div>
                  </button>
                </div>
              </div>

              {/* Algorithm Selection (for classical) */}
              {modelType === 'classical' && (
                <div className="mb-6">
                  <label className="block text-sm font-semibold text-gray-700 mb-3">
                    Choose Algorithm
                  </label>
                  <select
                    value={algorithm}
                    onChange={(e) => setAlgorithm(e.target.value)}
                    className="w-full px-4 py-3 border-2 border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all bg-white"
                  >
                    {availableAlgorithms.map((alg) => (
                      <option key={alg} value={alg}>
                        {formatAlgorithmName(alg)}
                      </option>
                    ))}
                  </select>
                </div>
              )}

              {/* Text Input */}
              <div className="mb-6">
                <label className="block text-sm font-semibold text-gray-700 mb-3">
                  Review Text
                </label>
                <textarea
                  value={text}
                  onChange={(e) => setText(e.target.value)}
                  placeholder="Enter a product review here... e.g., 'The battery life is excellent and lasts all day!'"
                  rows="8"
                  className="w-full px-4 py-3 border-2 border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all resize-none bg-gray-50"
                />
                <div className="flex justify-between items-center mt-2">
                  <div className="text-xs text-gray-500">
                    {text.length} characters
                  </div>
                </div>
              </div>

              {/* Analyze Button */}
              <button
                onClick={handlePredict}
                disabled={loading || !text.trim()}
                className="w-full bg-gradient-to-r from-blue-600 via-indigo-600 to-purple-600 hover:from-blue-700 hover:via-indigo-700 hover:to-purple-700 disabled:from-gray-300 disabled:to-gray-400 text-white font-bold py-4 px-6 rounded-xl transition-all shadow-lg hover:shadow-xl disabled:cursor-not-allowed transform hover:scale-[1.02] active:scale-[0.98]"
              >
                {loading ? (
                  <span className="flex items-center justify-center gap-2">
                    <svg className="animate-spin h-5 w-5" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                    </svg>
                    Analyzing...
                  </span>
                ) : (
                  <span className="flex items-center justify-center gap-2">
                    ANALYZE SENTIMENT
                  </span>
                )}
              </button>

              {/* Error Message */}
              {error && (
                <div className="mt-6 p-4 bg-rose-50 border-2 border-rose-200 rounded-xl flex items-start gap-3">
                  
                  <div>
                    <p className="text-rose-700 font-semibold">Error</p>
                    <p className="text-rose-600 text-sm">{error}</p>
                  </div>
                </div>
              )}

              {/* Prediction Result */}
              {prediction && prediction.valid && (
                <div className={`mt-6 p-6 border-2 rounded-xl ${getSentimentBg(prediction.sentiment)} shadow-lg`}>
                  <div className="flex items-center justify-between mb-4">
                    <div className="flex-1">
                      <p className="text-sm text-gray-600 mb-2 font-semibold uppercase tracking-wide">Predicted Sentiment</p>
                      <div className="flex items-center gap-4">
                        <p className={`text-4xl font-bold ${getSentimentColor(prediction.sentiment)}`}>
                          {prediction.sentiment?.toUpperCase()}
                        </p>
                      </div>
                    </div>
                    {prediction.confidence && (
                      <div className="text-right bg-white rounded-xl p-4 shadow-md">
                        <p className="text-xs text-gray-600 mb-1 uppercase tracking-wide font-semibold">Confidence</p>
                        <p className="text-3xl font-bold text-gray-800">
                          {prediction.confidence.toFixed(1)}%
                        </p>
                        <div className="w-24 bg-gray-200 rounded-full h-3 mt-3">
                          <div 
                            className="bg-gradient-to-r from-blue-600 to-indigo-600 h-3 rounded-full transition-all"
                            style={{ width: `${prediction.confidence}%` }}
                          />
                        </div>
                      </div>
                    )}
                  </div>
                  <div className="pt-4 border-t-2 border-white/50">
                    <p className="text-xs text-gray-600 font-medium">Model Used: {prediction.model_used}</p>
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Right Panel - Info */}
          <div className="space-y-6">
            <div className="bg-white rounded-2xl shadow-xl p-6 border border-gray-100">
              <h3 className="text-lg font-bold text-gray-800 mb-4">QUICK SAMPLES</h3>
              <div className="space-y-3">
                <button
                  onClick={() => setText("The battery life is excellent and lasts all day! I'm very impressed with the performance.")}
                  className="w-full text-left text-sm p-4 bg-gradient-to-br from-emerald-50 to-green-50 hover:from-emerald-100 hover:to-green-100 rounded-xl transition-all border-2 border-emerald-200 hover:shadow-md transform hover:scale-105"
                >
                  <div className="flex items-center gap-3">
                    <div>
                      <div className="font-bold text-emerald-800">POSITIVE REVIEW</div>
                      <div className="text-xs text-emerald-600 mt-1">Battery life & performance</div>
                    </div>
                  </div>
                </button>
                <button
                  onClick={() => setText("The screen is too small and very hard to read. Disappointed with the quality.")}
                  className="w-full text-left text-sm p-4 bg-gradient-to-br from-rose-50 to-red-50 hover:from-rose-100 hover:to-red-100 rounded-xl transition-all border-2 border-rose-200 hover:shadow-md transform hover:scale-105"
                >
                  <div className="flex items-center gap-3">
                    <div>
                      <div className="font-bold text-rose-800">NEGATIVE REVIEW</div>
                      <div className="text-xs text-rose-600 mt-1">Screen size issues</div>
                    </div>
                  </div>
                </button>
                <button
                  onClick={() => setText("It's okay, nothing special but works as expected. Average product for the price.")}
                  className="w-full text-left text-sm p-4 bg-gradient-to-br from-amber-50 to-yellow-50 hover:from-amber-100 hover:to-yellow-100 rounded-xl transition-all border-2 border-amber-200 hover:shadow-md transform hover:scale-105"
                >
                  <div className="flex items-center gap-3">
                    <div>
                      <div className="font-bold text-amber-800">NEUTRAL REVIEW</div>
                      <div className="text-xs text-amber-600 mt-1">Average experience</div>
                    </div>
                  </div>
                </button>
              </div>
            </div>

            <div className="bg-gradient-to-br from-indigo-50 via-purple-50 to-pink-50 rounded-2xl shadow-xl p-6 border border-indigo-200">
              <h3 className="text-lg font-bold text-gray-800 mb-3">ABOUT THIS PROJECT</h3>
              <p className="text-sm text-gray-700 leading-relaxed mb-4">
                Compare classical machine learning models with transformer-based deep learning for sentiment analysis on product reviews.
              </p>
              <div className="space-y-2 text-xs text-gray-700">
                <div className="flex items-center gap-2 bg-white/60 p-2 rounded-lg">
                  <span className="text-blue-500">✓</span>
                  <span className="font-medium">DistilBERT Transformer</span>
                </div>
                <div className="flex items-center gap-2 bg-white/60 p-2 rounded-lg">
                  <span className="text-blue-500">✓</span>
                  <span className="font-medium">8 Classical ML Algorithms</span>
                </div>
                <div className="flex items-center gap-2 bg-white/60 p-2 rounded-lg">
                  <span className="text-blue-500">✓</span>
                  <span className="font-medium">Real-time Predictions</span>
                </div>
              </div>
              <button
                onClick={() => setShowComparison(!showComparison)}
                className="mt-4 w-full text-sm bg-white hover:bg-indigo-50 text-indigo-700 font-semibold py-3 px-4 rounded-xl border-2 border-indigo-200 transition-all hover:shadow-md"
              >
                {showComparison ? '▲ Hide' : '▼ Show'} Model Comparison
              </button>
              {wordclouds.length > 0 && (
                <button
                  onClick={() => setShowWordclouds(!showWordclouds)}
                  className="mt-3 w-full text-sm bg-white hover:bg-purple-50 text-purple-700 font-semibold py-3 px-4 rounded-xl border-2 border-purple-200 transition-all hover:shadow-md"
                >
                  {showWordclouds ? '▲ Hide' : '▼ Show'} Word Clouds
                </button>
              )}
            </div>
          </div>
        </div>

        {/* Word Clouds Section */}
        {showWordclouds && wordclouds.length > 0 && (
  <div className="mt-6 bg-white rounded-2xl shadow-xl p-8 border border-gray-100">
    
    <h2 className="text-2xl font-bold text-gray-800 mb-4">
      WORD CLOUDS BY SENTIMENT
    </h2>

    <p className="text-sm text-gray-600 mb-6">
      Visual representation of most common words in each sentiment category
    </p>

    <div className="grid md:grid-cols-3 gap-6">

      {wordclouds.map((wc) => {
        const url = typeof wc === "string"
          ? `${API_BASE}${wc}`               // wc is "/static/results/wordcloud_xxx.png"
          : `${API_BASE}${wc.filename}`;     // in case API returns an object

        return (
          <div key={url} className="border-2 rounded-xl p-5 bg-gray-50 hover:shadow-lg transition-all">
            <img
              src={url}
              alt="Word Cloud"
              className="w-full rounded-lg shadow-sm"
              onError={(e) => (e.target.style.display = "none")}
            />
          </div>
        );
      })}

    </div>

  </div>
)}


        {showWordclouds && wordclouds.length === 0 && (
          <div className="mt-6 bg-white rounded-2xl shadow-xl p-8 border border-gray-100">
            <h2 className="text-2xl font-bold text-gray-800 mb-4">WORD CLOUDS BY SENTIMENT</h2>
            <p className="text-sm text-gray-600 mb-6">Default word clouds from results</p>
            <div className="grid md:grid-cols-3 gap-6">
              {[
                { sentiment: 'positive', file: 'wordcloud_positive.png' },
                { sentiment: 'negative', file: 'wordcloud_negative.png' },
                { sentiment: 'neutral', file: 'wordcloud_neutral.png' }
              ].map(({ sentiment, file }) => (
                <div key={file} className="border-2 rounded-xl p-5 bg-gray-50 hover:shadow-lg transition-all">
                  <h4 className="text-sm font-bold text-gray-700 mb-3">{sentiment.toUpperCase()}</h4>
                  <img
                    src={`${API_BASE}/static/results/${file}`}
                    alt={`${sentiment} Word Cloud`}
                    className="w-full rounded-lg shadow-sm"
                    onError={(e) => e.target.style.display = 'none'}
                  />
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Comparison Section */}
        {showComparison && comparison && (
          <div className="mt-6 bg-white rounded-2xl shadow-xl p-8 border border-gray-100">
            <h2 className="text-2xl font-bold text-gray-800 mb-6">MODEL PERFORMANCE COMPARISON</h2>

            <div className="grid lg:grid-cols-2 gap-6 mb-6">
              {/* Transformer Metrics */}
              <div className="border-2 border-blue-300 rounded-xl p-6 bg-gradient-to-br from-blue-50 to-indigo-50 shadow-lg">
                <div className="flex items-center gap-3 mb-5">
                  <h3 className="text-xl font-bold text-blue-900">TRANSFORMER (DISTILBERT)</h3>
                </div>
                <div className="space-y-3">
                  <div className="flex justify-between items-center bg-white p-4 rounded-xl shadow-sm">
                    <span className="text-gray-700 font-semibold">Accuracy</span>
                    <span className="font-bold text-blue-600 text-lg">
                      {(comparison.transformer.accuracy * 100).toFixed(2)}%
                    </span>
                  </div>
                  {comparison.transformer.report && (
                    <>
                      <div className="flex justify-between items-center bg-white p-4 rounded-xl shadow-sm">
                        <span className="text-gray-700 font-semibold">Precision</span>
                        <span className="font-bold text-blue-600 text-lg">
                          {(comparison.transformer.report['weighted avg'].precision * 100).toFixed(2)}%
                        </span>
                      </div>
                      <div className="flex justify-between items-center bg-white p-4 rounded-xl shadow-sm">
                        <span className="text-gray-700 font-semibold">Recall</span>
                        <span className="font-bold text-blue-600 text-lg">
                          {(comparison.transformer.report['weighted avg'].recall * 100).toFixed(2)}%
                        </span>
                      </div>
                      <div className="flex justify-between items-center bg-white p-4 rounded-xl shadow-sm">
                        <span className="text-gray-700 font-semibold">F1-Score</span>
                        <span className="font-bold text-blue-600 text-lg">
                          {(comparison.transformer.report['weighted avg']['f1-score'] * 100).toFixed(2)}%
                        </span>
                      </div>
                    </>
                  )}
                </div>
              </div>

              {/* Classical Models */}
              <div className="border-2 border-emerald-300 rounded-xl p-6 bg-gradient-to-br from-emerald-50 to-green-50 shadow-lg">
                <div className="flex items-center gap-3 mb-5">
                  <h3 className="text-xl font-bold text-emerald-900">TOP CLASSICAL MODELS</h3>
                </div>
                <div className="space-y-3 max-h-80 overflow-y-auto pr-2">
                  {Object.entries(comparison.classical)
                    .sort((a, b) => b[1].accuracy - a[1].accuracy)
                    .map(([name, metrics]) => (
                      <div key={name} className="bg-white p-4 rounded-xl hover:shadow-md transition-shadow border border-emerald-100">
                        <p className="text-sm font-bold text-gray-800 mb-2">
                          {formatAlgorithmName(name)}
                        </p>
                        <div className="grid grid-cols-2 gap-3 text-xs">
                          <div className="bg-emerald-50 p-2 rounded-lg">
                            <span className="text-gray-600 font-medium">Accuracy:</span>
                            <span className="font-bold text-emerald-600 ml-1">
                              {(metrics.accuracy * 100).toFixed(2)}%
                            </span>
                          </div>
                          <div className="bg-emerald-50 p-2 rounded-lg">
                            <span className="text-gray-600 font-medium">F1:</span>
                            <span className="font-bold text-emerald-600 ml-1">
                              {(metrics.f1 * 100).toFixed(2)}%
                            </span>
                          </div>
                        </div>
                      </div>
                    ))}
                </div>
              </div>
            </div>

            {/* Visualizations */}
            <div className="mt-8">
              <h3 className="text-xl font-bold text-gray-800 mb-5">Visualizations</h3>
              <div className="grid md:grid-cols-2 gap-6">
                <div className="border-2 rounded-xl p-5 bg-gray-50 hover:shadow-lg transition-all">
                  <h4 className="text-sm font-bold text-gray-700 mb-4">CONFUSION MATRIX (TRANSFORMER)</h4>
                  <img
                    src={`${API_BASE}/static/results/confusion_matrix.png`}
                    alt="Confusion Matrix"
                    className="w-full rounded-lg shadow-sm"
                    onError={(e) => {
                      e.target.style.display = 'none'
                      e.target.nextSibling.style.display = 'block'
                    }}
                  />
                  <p className="hidden text-xs text-gray-500 mt-3 text-center">
                    Run evaluation to generate this visualization
                  </p>
                </div>
                <div className="border-2 rounded-xl p-5 bg-gray-50 hover:shadow-lg transition-all">
                  <h4 className="text-sm font-bold text-gray-700 mb-4">MODEL ACCURACY COMPARISON</h4>
                  <img
                    src={`${API_BASE}/static/results/model_comparison_accuracy.png`}
                    alt="Accuracy Comparison"
                    className="w-full rounded-lg shadow-sm"
                    onError={(e) => {
                      e.target.style.display = 'none'
                      e.target.nextSibling.style.display = 'block'
                    }}
                  />
                  <p className="hidden text-xs text-gray-500 mt-3 text-center">
                    Run evaluation to generate this visualization
                  </p>
                </div>
              </div>
              
              <div className="mt-6 border-2 rounded-xl p-5 bg-gray-50 hover:shadow-lg transition-all">
                <h4 className="text-sm font-bold text-gray-700 mb-3">TOP TF-IDF FEATURES (LOGISTIC REGRESSION)</h4>
                <img
                  src={`${API_BASE}/static/results/top_tfidf_features_logreg.png`}
                  alt="TF-IDF Features"
                  className="w-full rounded-lg shadow-sm"
                  onError={(e) => {
                    e.target.style.display = 'none'
                    e.target.nextSibling.style.display = 'block'
                  }}
                />
                <p className="hidden text-xs text-gray-500 mt-3 text-center">
                  Train Logistic Regression to generate this visualization
                </p>
              </div>
            </div>
          </div>
        )}

        {/* Footer */}
        <div className="mt-12 text-center text-gray-500 text-sm border-t pt-8">
          <p className="mb-2 font-medium">SENTIMENT ANALYSIS: TRANSFORMER VS CLASSICAL ML MODELS</p>
          <p className="text-xs">Built with FastAPI, React, and Tailwind CSS</p>
        </div>
      </div>
    </div>
  )
}

export default App
