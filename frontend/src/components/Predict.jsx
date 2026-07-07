import { useEffect, useMemo, useState } from 'react'
import Gauge from './Gauge.jsx'
import { getFeatures, postPredict } from '../api.js'

const TIER_TEXT = {
  high: 'High risk',
  elevated: 'Elevated risk',
  moderate: 'Moderate risk',
  low: 'Low risk',
}
const TIER_CLASS = {
  high: 'tier-high',
  elevated: 'tier-elevated',
  moderate: 'tier-moderate',
  low: 'tier-low',
}

// Short display names for long dataset column headers
function shortLabel(name) {
  return name.trim().replace(' (Yuan ??)', ' (Yuan)')
}

export default function Predict({ variant }) {
  const [featureMeta, setFeatureMeta] = useState(null)
  const [values, setValues] = useState({})
  const [thresholdMode, setThresholdMode] = useState('tuned')
  const [result, setResult] = useState(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)

  useEffect(() => {
    setResult(null)
    setError(null)
    getFeatures(variant)
      .then((meta) => {
        setFeatureMeta(meta)
        setValues(Object.fromEntries(meta.names.map((n) => [n, ''])))
      })
      .catch((e) => setError(e.message))
  }, [variant])

  const allFilled = useMemo(
    () =>
      featureMeta &&
      featureMeta.names.every(
        (n) => values[n] !== '' && !Number.isNaN(parseFloat(values[n])),
      ),
    [featureMeta, values],
  )

  function loadExample(kind) {
    if (!featureMeta) return
    const source =
      kind === 'median'
        ? Object.fromEntries(
            featureMeta.names.map((n) => [n, featureMeta.stats[n].median]),
          )
        : featureMeta.examples[kind]
    setValues(Object.fromEntries(featureMeta.names.map((n) => [n, String(source[n])])))
    setResult(null)
  }

  async function runPredict() {
    setBusy(true)
    setError(null)
    try {
      const features = Object.fromEntries(
        featureMeta.names.map((n) => [n, parseFloat(values[n])]),
      )
      const res = await postPredict({ variant, features, threshold_mode: thresholdMode })
      setResult(res)
    } catch (e) {
      setError(e.message)
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="grid-2">
      <section className="panel" aria-label="Company financials">
        <div className="panel-title">Company financials — {featureMeta ? featureMeta.names.length : '…'} ratios</div>

        <div className="btn-row">
          <button className="btn ghost" onClick={() => loadExample('healthy')}>
            Load healthy example
          </button>
          <button className="btn ghost" onClick={() => loadExample('distressed')}>
            Load distressed example
          </button>
          <button className="btn ghost" onClick={() => loadExample('median')}>
            Load dataset median
          </button>
        </div>

        {featureMeta ? (
          featureMeta.names.map((name) => (
            <div className="field" key={name}>
              <label htmlFor={name}>{shortLabel(name)}</label>
              <input
                id={name}
                type="number"
                step="any"
                value={values[name]}
                onChange={(e) =>
                  setValues((v) => ({ ...v, [name]: e.target.value }))
                }
              />
              <div className="hint num">
                range {featureMeta.stats[name].min} – {featureMeta.stats[name].max}
              </div>
            </div>
          ))
        ) : (
          <div className="empty-note">Loading feature definitions…</div>
        )}

        <div style={{ display: 'flex', gap: 12, alignItems: 'center', marginTop: 6 }}>
          <button
            className="btn primary"
            onClick={runPredict}
            disabled={!allFilled || busy}
          >
            {busy ? 'Scoring…' : 'Score this company'}
          </button>
          <div className="seg" role="group" aria-label="Decision threshold">
            <button
              className={thresholdMode === 'tuned' ? 'on' : ''}
              onClick={() => setThresholdMode('tuned')}
              title="Each model uses its F1-optimized threshold"
            >
              tuned threshold
            </button>
            <button
              className={thresholdMode === 'default' ? 'on' : ''}
              onClick={() => setThresholdMode('default')}
              title="All models use 0.5"
            >
              0.5
            </button>
          </div>
        </div>
        {error && <div className="error-note">{error}</div>}
      </section>

      <section className="panel" aria-label="Verdict">
        <div className="panel-title">Verdict</div>
        {result ? <Verdict result={result} /> : (
          <div className="empty-note">
            Fill in the ratios (or load an example) and score the company.
          </div>
        )}
      </section>
    </div>
  )
}

function Verdict({ result }) {
  return (
    <>
      <div className="verdict-head">
        <Gauge value={result.average_probability} tier={result.risk_tier} />
        <div className="verdict-meta">
          <div className={`verdict-tier ${TIER_CLASS[result.risk_tier]}`}>
            {TIER_TEXT[result.risk_tier]}
          </div>
          <div className="verdict-sub">
            {result.votes_bankrupt} of {result.n_models} models flag this company
            {result.consensus === 'split' ? ' — split decision' : ''}
          </div>
        </div>
      </div>

      <div className="vote-strip" aria-label="Model votes">
        {result.models.map((m) => (
          <div
            key={m.model_key}
            className="vote-cell"
            title={`${m.model_name}: ${(m.probability * 100).toFixed(1)}%`}
            style={{
              background:
                m.prediction === 'bankrupt'
                  ? 'var(--risk-high)'
                  : 'var(--risk-low)',
            }}
          >
            {m.model_key.toUpperCase()}
          </div>
        ))}
      </div>
      <div className="vote-legend">
        red = flagged bankrupt at that model's threshold · green = clear
      </div>

      <div style={{ marginTop: 18 }}>
        {result.models.map((m) => (
          <div className="model-row" key={m.model_key}>
            <div className="model-name">{m.model_name}</div>
            <div className="prob-track">
              <div
                className="prob-fill"
                style={{
                  width: `${m.probability * 100}%`,
                  background:
                    m.prediction === 'bankrupt'
                      ? 'var(--risk-high)'
                      : 'var(--risk-low)',
                }}
              />
              <div
                className="prob-thresh"
                style={{ left: `${m.threshold * 100}%` }}
                title={`threshold ${m.threshold}`}
              />
            </div>
            <div className="prob-val num">{(m.probability * 100).toFixed(1)}%</div>
          </div>
        ))}
      </div>
      <div className="footnote">
        Each bar is that model's predicted probability of bankruptcy; the tick
        mark is its decision threshold. Thresholds were tuned to maximize
        bankrupt-class F1 on cross-validated training predictions.
      </div>
    </>
  )
}
