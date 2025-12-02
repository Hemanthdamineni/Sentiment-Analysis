import React, { useState, useEffect } from 'react'

function App() {
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

  const API_BASE = 'http://localhost:8000'

  useEffect(() => {
    fetchAvailableModels()
    fetchComparison()
    fetchWordclouds()
  }, [])

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
    if (s === 'positive') return 'text-green-600'
    if (s === 'negative') return 'text-red-600'
    if (s === 'neutral') return 'text-yellow-600'
    return 'text-gray-600'
  }

  const getSentimentBg = (sentiment) => {
    const s = sentiment?.toLowerCase()
    if (s === 'positive') return 'bg-green-50 border-green-200'
    if (s === 'negative') return 'bg-red-50 border-red-200'
    if (s === 'neutral') return 'bg-yellow-50 border-yellow-200'
    return 'bg-gray-50 border-gray-200'
  }

  const getSentimentEmoji = (sentiment) => {
    const s = sentiment?.toLowerCase()
    if (s === 'positive') return '😊'
    if (s === 'negative') return '😞'
    if (s === 'neutral') return '😐'
    return '❓'
  }

  const formatAlgorithmName = (name) => {
    return name.replace(/_/g, ' ').split(' ').map(word => 
      word.charAt(0).toUpperCase() + word.slice(1)
    ).join(' ')
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-indigo-50 to-purple-50">
      <div className="container mx-auto px-4 py-8 max-w-7xl">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-5xl font-bold text-gray-800 mb-3 flex items-center justify-center gap-3">
            <span>📱</span>
            Smartwatch Sentiment Analyzer
          </h1>
          <p className="text-gray-600 text-lg">Classical ML vs Transformer Comparison</p>
          <p className="text-sm text-gray-500 mt-2">Analyzing smartwatch reviews with state-of-the-art NLP models</p>
        </div>

        {/* Main Content */}
        <div className="grid lg:grid-cols-3 gap-6">
          {/* Left Panel - Input (2 columns) */}
          <div className="lg:col-span-2 space-y-4">
            <div className="bg-white rounded-xl shadow-lg p-6 border border-gray-100">
              <h2 className="text-2xl font-semibold text-gray-800 mb-5 flex items-center gap-2">
                <span>🔍</span>
                Analyze Review
              </h2>

              {/* Model Selection */}
              <div className="mb-5">
                <label className="block text-sm font-medium text-gray-700 mb-3">
                  Select Model Type
                </label>
                <div className="grid grid-cols-2 gap-3">
                  <button
                    onClick={() => setModelType('transformer')}
                    className={`p-4 rounded-lg border-2 transition-all ${
                      modelType === 'transformer'
                        ? 'border-blue-500 bg-blue-50 shadow-md'
                        : 'border-gray-200 hover:border-gray-300'
                    }`}
                  >
                    <div className="text-2xl mb-1">🤖</div>
                    <div className="font-semibold text-sm">Transformer</div>
                    <div className="text-xs text-gray-500">DistilBERT</div>
                  </button>
                  <button
                    onClick={() => setModelType('classical')}
                    className={`p-4 rounded-lg border-2 transition-all ${
                      modelType === 'classical'
                        ? 'border-blue-500 bg-blue-50 shadow-md'
                        : 'border-gray-200 hover:border-gray-300'
                    }`}
                  >
                    <div className="text-2xl mb-1">📊</div>
                    <div className="font-semibold text-sm">Classical ML</div>
                    <div className="text-xs text-gray-500">Multiple algorithms</div>
                  </button>
                </div>
              </div>

              {/* Algorithm Selection (for classical) */}
              {modelType === 'classical' && (
                <div className="mb-5">
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Choose Algorithm
                  </label>
                  <select
                    value={algorithm}
                    onChange={(e) => setAlgorithm(e.target.value)}
                    className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all"
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
              <div className="mb-5">
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Review Text
                </label>
                <textarea
                  value={text}
                  onChange={(e) => setText(e.target.value)}
                  placeholder="Enter a smartwatch review here... e.g., 'The battery life is excellent and lasts all day!'"
                  rows="7"
                  className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all resize-none"
                />
                <div className="text-xs text-gray-500 mt-1 text-right">
                  {text.length} characters
                </div>
              </div>

              {/* Analyze Button */}
              <button
                onClick={handlePredict}
                disabled={loading || !text.trim()}
                className="w-full bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 disabled:from-gray-300 disabled:to-gray-400 text-white font-semibold py-4 px-4 rounded-lg transition-all shadow-md hover:shadow-lg disabled:cursor-not-allowed transform hover:scale-[1.02] active:scale-[0.98]"
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
                  '🚀 Analyze Sentiment'
                )}
              </button>

              {/* Error Message */}
              {error && (
                <div className="mt-4 p-4 bg-red-50 border border-red-200 rounded-lg flex items-start gap-3">
                  <span className="text-red-500 text-xl">⚠️</span>
                  <div>
                    <p className="text-red-600 text-sm font-medium">Error</p>
                    <p className="text-red-600 text-sm">{error}</p>
                  </div>
                </div>
              )}

              {/* Prediction Result */}
              {prediction && prediction.valid && (
                <div className={`mt-4 p-5 border-2 rounded-lg ${getSentimentBg(prediction.sentiment)} shadow-md`}>
                  <div className="flex items-center justify-between mb-3">
                    <div className="flex-1">
                      <p className="text-sm text-gray-600 mb-1 font-medium">Predicted Sentiment</p>
                      <div className="flex items-center gap-3">
                        <span className="text-4xl">{getSentimentEmoji(prediction.sentiment)}</span>
                        <p className={`text-3xl font-bold ${getSentimentColor(prediction.sentiment)}`}>
                          {prediction.sentiment}
                        </p>
                      </div>
                    </div>
                    {prediction.confidence && (
                      <div className="text-right bg-white rounded-lg p-3 shadow-sm">
                        <p className="text-xs text-gray-600 mb-1">Confidence</p>
                        <p className="text-2xl font-bold text-gray-800">
                          {prediction.confidence.toFixed(1)}%
                        </p>
                        <div className="w-20 bg-gray-200 rounded-full h-2 mt-2">
                          <div 
                            className="bg-blue-600 h-2 rounded-full transition-all"
                            style={{ width: `${prediction.confidence}%` }}
                          />
                        </div>
                      </div>
                    )}
                  </div>
                  <div className="pt-3 border-t border-gray-200">
                    <p className="text-xs text-gray-500 flex items-center gap-2">
                      <span>🤖</span>
                      <span>Model: {prediction.model_used}</span>
                    </p>
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Right Panel - Info */}
          <div className="space-y-4">
            <div className="bg-white rounded-xl shadow-lg p-6 border border-gray-100">
              <h3 className="text-lg font-semibold text-gray-800 mb-4 flex items-center gap-2">
                <span>💡</span>
                Quick Samples
              </h3>
              <div className="space-y-2">
                <button
                  onClick={() => setText("The battery life is excellent and lasts all day! I'm very impressed with the performance.")}
                  className="w-full text-left text-sm p-3 bg-green-50 hover:bg-green-100 rounded-lg transition-all border border-green-200 hover:shadow-md"
                >
                  <div className="flex items-center gap-2">
                    <span className="text-lg">😊</span>
                    <div>
                      <div className="font-medium text-green-800">Positive Review</div>
                      <div className="text-xs text-green-600">Battery life & performance</div>
                    </div>
                  </div>
                </button>
                <button
                  onClick={() => setText("The screen is too small and very hard to read. Disappointed with the quality.")}
                  className="w-full text-left text-sm p-3 bg-red-50 hover:bg-red-100 rounded-lg transition-all border border-red-200 hover:shadow-md"
                >
                  <div className="flex items-center gap-2">
                    <span className="text-lg">😞</span>
                    <div>
                      <div className="font-medium text-red-800">Negative Review</div>
                      <div className="text-xs text-red-600">Screen size issues</div>
                    </div>
                  </div>
                </button>
                <button
                  onClick={() => setText("It's okay, nothing special but works as expected. Average product for the price.")}
                  className="w-full text-left text-sm p-3 bg-yellow-50 hover:bg-yellow-100 rounded-lg transition-all border border-yellow-200 hover:shadow-md"
                >
                  <div className="flex items-center gap-2">
                    <span className="text-lg">😐</span>
                    <div>
                      <div className="font-medium text-yellow-800">Neutral Review</div>
                      <div className="text-xs text-yellow-600">Average experience</div>
                    </div>
                  </div>
                </button>
              </div>
            </div>

            <div className="bg-gradient-to-br from-indigo-50 to-purple-50 rounded-xl shadow-lg p-6 border border-indigo-100">
              <h3 className="text-lg font-semibold text-gray-800 mb-3 flex items-center gap-2">
                <span>📚</span>
                About This Project
              </h3>
              <p className="text-sm text-gray-600 leading-relaxed mb-4">
                Compare classical machine learning models with transformer-based deep learning for sentiment analysis on smartwatch reviews.
              </p>
              <div className="space-y-2 text-xs text-gray-600">
                <div className="flex items-center gap-2">
                  <span className="text-blue-500">✓</span>
                  <span>DistilBERT Transformer</span>
                </div>
                <div className="flex items-center gap-2">
                  <span className="text-blue-500">✓</span>
                  <span>8 Classical ML Algorithms</span>
                </div>
                <div className="flex items-center gap-2">
                  <span className="text-blue-500">✓</span>
                  <span>Real-time Predictions</span>
                </div>
              </div>
              <button
                onClick={() => setShowComparison(!showComparison)}
                className="mt-4 w-full text-sm bg-white text-indigo-600 hover:bg-indigo-50 font-medium py-2 px-4 rounded-lg border border-indigo-200 transition-all"
              >
                {showComparison ? '▲ Hide' : '▼ Show'} Model Comparison
              </button>
              {wordclouds.length > 0 && (
                <button
                  onClick={() => setShowWordclouds(!showWordclouds)}
                  className="mt-2 w-full text-sm bg-white text-purple-600 hover:bg-purple-50 font-medium py-2 px-4 rounded-lg border border-purple-200 transition-all"
                >
                  {showWordclouds ? '▲ Hide' : '▼ Show'} Word Clouds
                </button>
              )}
            </div>
          </div>
        </div>

        {/* Word Clouds Section */}
        {showWordclouds && wordclouds.length > 0 && (
          <div className="mt-6 bg-white rounded-xl shadow-lg p-6 border border-gray-100">
            <h2 className="text-2xl font-semibold text-gray-800 mb-4 flex items-center gap-2">
              <span>☁️</span>
              Word Clouds by Sentiment
            </h2>
            <p className="text-sm text-gray-600 mb-4">
              Visual representation of most common words in each sentiment category
            </p>
            <div className="grid md:grid-cols-3 gap-4">
              {wordclouds.map((wc) => {
                const isString = typeof wc === 'string'
                const imgSrc = isString ? wc : `${API_BASE}/images/${wc.filename}`
                const sentiment = isString ? null : wc.sentiment
                return (
                  <div key={imgSrc} className="border rounded-lg p-4 bg-gray-50">
                    {sentiment && (
                      <h4 className="text-sm font-semibold text-gray-700 mb-2 flex items-center gap-2">
                        <span>{getSentimentEmoji(sentiment)}</span>
                        {sentiment}
                      </h4>
                    )}
                    <img
                      src={imgSrc}
                      alt={sentiment ? `${sentiment} Word Cloud` : 'Word Cloud'}
                      className="w-full rounded-lg shadow-sm"
                      onError={(e) => e.target.style.display = 'none'}
                    />
                  </div>
                )
              })}
            </div>
          </div>
        )}

        {/* Comparison Section */}
        {showComparison && comparison && (
          <div className="mt-6 bg-white rounded-xl shadow-lg p-6 border border-gray-100">
            <h2 className="text-2xl font-semibold text-gray-800 mb-5 flex items-center gap-2">
              <span>📊</span>
              Model Performance Comparison
            </h2>

            <div className="grid lg:grid-cols-2 gap-6 mb-6">
              {/* Transformer Metrics */}
              <div className="border-2 border-blue-200 rounded-xl p-5 bg-gradient-to-br from-blue-50 to-indigo-50">
                <div className="flex items-center gap-2 mb-4">
                  <span className="text-2xl">🤖</span>
                  <h3 className="text-lg font-semibold text-blue-900">
                    Transformer (DistilBERT)
                  </h3>
                </div>
                <div className="space-y-3">
                  <div className="flex justify-between items-center bg-white p-3 rounded-lg">
                    <span className="text-gray-700 font-medium">Accuracy</span>
                    <span className="font-bold text-blue-600">
                      {(comparison.transformer.accuracy * 100).toFixed(2)}%
                    </span>
                  </div>
                  {comparison.transformer.report && (
                    <>
                      <div className="flex justify-between items-center bg-white p-3 rounded-lg">
                        <span className="text-gray-700 font-medium">Precision</span>
                        <span className="font-bold text-blue-600">
                          {(comparison.transformer.report['weighted avg'].precision * 100).toFixed(2)}%
                        </span>
                      </div>
                      <div className="flex justify-between items-center bg-white p-3 rounded-lg">
                        <span className="text-gray-700 font-medium">Recall</span>
                        <span className="font-bold text-blue-600">
                          {(comparison.transformer.report['weighted avg'].recall * 100).toFixed(2)}%
                        </span>
                      </div>
                      <div className="flex justify-between items-center bg-white p-3 rounded-lg">
                        <span className="text-gray-700 font-medium">F1-Score</span>
                        <span className="font-bold text-blue-600">
                          {(comparison.transformer.report['weighted avg']['f1-score'] * 100).toFixed(2)}%
                        </span>
                      </div>
                    </>
                  )}
                </div>
              </div>

              {/* Classical Models */}
              <div className="border-2 border-green-200 rounded-xl p-5 bg-gradient-to-br from-green-50 to-emerald-50">
                <div className="flex items-center gap-2 mb-4">
                  <span className="text-2xl">📊</span>
                  <h3 className="text-lg font-semibold text-green-900">
                    Top Classical Models
                  </h3>
                </div>
                <div className="space-y-2 max-h-80 overflow-y-auto pr-2">
                  {Object.entries(comparison.classical)
                    .sort((a, b) => b[1].accuracy - a[1].accuracy)
                    .map(([name, metrics]) => (
                      <div key={name} className="bg-white p-3 rounded-lg hover:shadow-md transition-shadow">
                        <p className="text-sm font-semibold text-gray-800 mb-2">
                          {formatAlgorithmName(name)}
                        </p>
                        <div className="grid grid-cols-2 gap-2 text-xs">
                          <div>
                            <span className="text-gray-600">Accuracy:</span>
                            <span className="font-bold text-green-600 ml-1">
                              {(metrics.accuracy * 100).toFixed(2)}%
                            </span>
                          </div>
                          <div>
                            <span className="text-gray-600">F1:</span>
                            <span className="font-bold text-green-600 ml-1">
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
            <div className="mt-6">
              <h3 className="text-lg font-semibold text-gray-800 mb-4">Visualizations</h3>
              <div className="grid md:grid-cols-2 gap-6">
                <div className="border rounded-lg p-4 bg-gray-50">
                  <h4 className="text-sm font-semibold text-gray-700 mb-3 flex items-center gap-2">
                    <span>📈</span>
                    Confusion Matrix (Transformer)
                  </h4>
                  <img
                    src={`${API_BASE}/images/confusion_matrix.png`}
                    alt="Confusion Matrix"
                    className="w-full rounded-lg shadow-sm"
                    onError={(e) => {
                      e.target.style.display = 'none'
                      e.target.nextSibling.style.display = 'block'
                    }}
                  />
                  <p className="hidden text-xs text-gray-500 mt-2 text-center">
                    Run evaluation to generate this visualization
                  </p>
                </div>
                <div className="border rounded-lg p-4 bg-gray-50">
                  <h4 className="text-sm font-semibold text-gray-700 mb-3 flex items-center gap-2">
                    <span>📊</span>
                    Model Accuracy Comparison
                  </h4>
                  <img
                    src={`${API_BASE}/images/model_comparison_accuracy.png`}
                    alt="Accuracy Comparison"
                    className="w-full rounded-lg shadow-sm"
                    onError={(e) => {
                      e.target.style.display = 'none'
                      e.target.nextSibling.style.display = 'block'
                    }}
                  />
                  <p className="hidden text-xs text-gray-500 mt-2 text-center">
                    Run evaluation to generate this visualization
                  </p>
                </div>
              </div>
              
              <div className="mt-6 border rounded-lg p-4 bg-gray-50">
                <h4 className="text-sm font-semibold text-gray-700 mb-3 flex items-center gap-2">
                  <span>🔤</span>
                  Top TF-IDF Features (Logistic Regression)
                </h4>
                <img
                  src={`${API_BASE}/images/top_tfidf_features_logreg.png`}
                  alt="TF-IDF Features"
                  className="w-full rounded-lg shadow-sm"
                  onError={(e) => {
                    e.target.style.display = 'none'
                    e.target.nextSibling.style.display = 'block'
                  }}
                />
                <p className="hidden text-xs text-gray-500 mt-2 text-center">
                  Train Logistic Regression to generate this visualization
                </p>
              </div>
            </div>
          </div>
        )}

        {/* Footer */}
        <div className="mt-8 text-center text-gray-500 text-sm border-t pt-6">
          <p className="mb-2">🎓 Sentiment Analysis: Transformer vs Classical ML Models</p>
          <p className="text-xs">Built with FastAPI, React, and Tailwind CSS</p>
        </div>
      </div>
    </div>
  )
}

export default App
