import { useEffect, useState } from 'react'
import Predict from './components/Predict.jsx'
import Dashboard from './components/Dashboard.jsx'
import { getSummary } from './api.js'

const VARIANT_LABELS = { original: 'Original features', knn30: 'KNN-30 imputed' }

export default function App() {
  const [tab, setTab] = useState('predict')
  const [variant, setVariant] = useState('original')
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
          <span className="dataset-note">
            Taiwan dataset · 6,819 firms · 220 bankrupt
          </span>
          <div className="seg" role="group" aria-label="Dataset variant">
            {Object.entries(VARIANT_LABELS).map(([key, label]) => (
              <button
                key={key}
                className={variant === key ? 'on' : ''}
                onClick={() => setVariant(key)}
              >
                {label}
              </button>
            ))}
          </div>
        </div>
      </header>

      <nav className="tabs" aria-label="Sections">
        <button
          className={`tab ${tab === 'predict' ? 'active' : ''}`}
          onClick={() => setTab('predict')}
        >
          Predict
        </button>
        <button
          className={`tab ${tab === 'dashboard' ? 'active' : ''}`}
          onClick={() => setTab('dashboard')}
        >
          Model analytics
        </button>
      </nav>

      {error && (
        <div className="panel">
          <div className="error-note">
            Can't reach the API — is the backend running on port 8000? ({error})
          </div>
        </div>
      )}

      {tab === 'predict' ? (
        <Predict variant={variant} summary={summary} />
      ) : (
        <Dashboard variant={variant} summary={summary} />
      )}
    </div>
  )
}
