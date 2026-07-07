import { useEffect, useState } from 'react'
import Predict from './components/Predict.jsx'
import Dashboard from './components/Dashboard.jsx'
import Batch from './components/Batch.jsx'
import { getSummary } from './api.js'

const VARIANT_LABELS = { mean: 'Median imputation', knn: 'KNN imputation' }

export default function App() {
  const [tab, setTab] = useState('predict')
  const [variant, setVariant] = useState('mean')
  const [summary, setSummary] = useState(null)
  const [error, setError] = useState(null)

  useEffect(() => {
    getSummary().then(setSummary).catch((e) => setError(e.message))
  }, [])

  return (
    <div className="shell">
      <header className="topbar">
        <div className="brand">
          <h1>RiskDesk</h1>
          <span className="tag">bankruptcy prediction</span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 14 }}>
          <span className="dataset-note">Taiwan dataset · 6,819 firms · 220 bankrupt</span>
          <div className="seg" role="group" aria-label="Imputation variant">
            {Object.entries(VARIANT_LABELS).map(([key, label]) => (
              <button key={key} className={variant === key ? 'on' : ''} onClick={() => setVariant(key)}>
                {label}
              </button>
            ))}
          </div>
        </div>
      </header>

      <nav className="tabs" aria-label="Sections">
        <button className={`tab ${tab === 'predict' ? 'active' : ''}`} onClick={() => setTab('predict')}>Predict</button>
        <button className={`tab ${tab === 'batch' ? 'active' : ''}`} onClick={() => setTab('batch')}>Batch scoring</button>
        <button className={`tab ${tab === 'dashboard' ? 'active' : ''}`} onClick={() => setTab('dashboard')}>Model analytics</button>
      </nav>

      {error && (
        <div className="state-note err">
          <span className="blink">● offline</span> — can't reach the API on port 8000.
          <div className="path">Start it with: <code>cd backend &amp;&amp; uvicorn api.main:app --reload --port 8000</code></div>
        </div>
      )}

      {/* All tabs stay mounted; inactive ones are hidden so their state
          (your prediction, form inputs, uploaded batch) survives switching. */}
      {!error && (
        <>
          <div style={{ display: tab === 'predict' ? 'block' : 'none' }}>
            <Predict variant={variant} summary={summary} />
          </div>
          <div style={{ display: tab === 'batch' ? 'block' : 'none' }}>
            <Batch variant={variant} summary={summary} />
          </div>
          <div style={{ display: tab === 'dashboard' ? 'block' : 'none' }}>
            <Dashboard variant={variant} summary={summary} />
          </div>
        </>
      )}
    </div>
  )
}
