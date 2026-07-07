import { useEffect, useMemo, useRef, useState } from 'react'
import Gauge from './Gauge.jsx'
import Info from './Info.jsx'
import ShapWaterfall from './ShapWaterfall.jsx'
import { getFeatures, postPredict, postExplain } from '../api.js'

const TIER_TEXT = { high: 'High risk', elevated: 'Elevated risk', moderate: 'Moderate risk', low: 'Low risk' }
const TIER_CLASS = { high: 'tier-high', elevated: 'tier-elevated', moderate: 'tier-moderate', low: 'tier-low' }

function shortLabel(name) { return name.trim().replace(' (Yuan ??)', ' (Yuan)') }

export default function Predict({ variant }) {
  const [meta, setMeta] = useState(null)
  const [values, setValues] = useState({})
  const [inputMode, setInputMode] = useState('form') // 'form' | 'sliders'
  const [thresholdMode, setThresholdMode] = useState('tuned')
  const [explainModel, setExplainModel] = useState('rf')
  const [result, setResult] = useState(null)
  const [explanation, setExplanation] = useState(null)
  const [busy, setBusy] = useState(false)
  const [explaining, setExplaining] = useState(false)
  const [error, setError] = useState(null)
  const liveTimer = useRef(null)

  useEffect(() => {
    setResult(null); setExplanation(null); setError(null)
    getFeatures(variant)
      .then((m) => { setMeta(m); setValues(Object.fromEntries(m.names.map((n) => [n, '']))) })
      .catch((e) => setError(e.message))
  }, [variant])

  const allFilled = useMemo(
    () => meta && meta.names.every((n) => values[n] !== '' && !Number.isNaN(parseFloat(values[n]))),
    [meta, values],
  )

  function loadExample(kind) {
    if (!meta) return
    const src = kind === 'median'
      ? Object.fromEntries(meta.names.map((n) => [n, meta.stats[n].median]))
      : meta.examples[kind]
    const next = Object.fromEntries(meta.names.map((n) => [n, String(src[n])]))
    setValues(next)
    setResult(null); setExplanation(null)
    if (inputMode === 'sliders') scoreLive(next)
  }

  function buildFeatures(v = values) {
    return Object.fromEntries(meta.names.map((n) => [n, parseFloat(v[n])]))
  }

  async function runPredict() {
    setBusy(true); setError(null); setExplanation(null)
    try {
      const res = await postPredict({ variant, features: buildFeatures(), threshold_mode: thresholdMode })
      setResult(res)
      runExplain(buildFeatures())
    } catch (e) { setError(e.message) } finally { setBusy(false) }
  }

  async function runExplain(featuresArg) {
    if (!meta) return
    setExplaining(true)
    try {
      const ex = await postExplain({
        variant, model_key: explainModel,
        features: featuresArg || buildFeatures(),
        threshold_mode: thresholdMode, want_narrative: true,
      })
      setExplanation(ex)
    } catch (e) { setError(e.message) } finally { setExplaining(false) }
  }

  // Debounced live scoring for slider mode
  function scoreLive(v) {
    if (!meta) return
    const feats = buildFeatures(v)
    if (Object.values(feats).some((x) => Number.isNaN(x))) return
    clearTimeout(liveTimer.current)
    liveTimer.current = setTimeout(async () => {
      try {
        const res = await postPredict({ variant, features: feats, threshold_mode: thresholdMode })
        setResult(res)
        const ex = await postExplain({
          variant, model_key: explainModel, features: feats,
          threshold_mode: thresholdMode, want_narrative: false, // skip LLM on every drag
        })
        setExplanation(ex)
      } catch (e) { /* silent during live drag */ }
    }, 180)
  }

  function onSlider(name, val) {
    const next = { ...values, [name]: String(val) }
    setValues(next)
    scoreLive(next)
  }

  useEffect(() => {
    if (result) runExplain()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [explainModel])

  // When entering slider mode with empty inputs, seed with median for a live start
  function enterSliders() {
    setInputMode('sliders')
    if (meta && !allFilled) {
      const seed = Object.fromEntries(meta.names.map((n) => [n, String(meta.stats[n].median)]))
      setValues(seed); scoreLive(seed)
    } else if (allFilled) {
      scoreLive(values)
    }
  }

  return (
    <div className="grid-2">
      <section className="panel" aria-label="Company financials">
        <div className="panel-title">Company financials — {meta ? meta.names.length : '…'} ratios</div>

        <div className="seg mode-toggle" role="group" aria-label="Input mode">
          <button className={inputMode === 'form' ? 'on' : ''} onClick={() => setInputMode('form')}>type values</button>
          <button className={inputMode === 'sliders' ? 'on' : ''} onClick={enterSliders}>what-if sliders</button>
        </div>

        <div className="btn-row">
          <button className="btn ghost" onClick={() => loadExample('healthy')}>Load healthy example</button>
          <button className="btn ghost" onClick={() => loadExample('distressed')}>Load distressed example</button>
          <button className="btn ghost" onClick={() => loadExample('median')}>Load dataset median</button>
        </div>

        {!meta ? (
          [...Array(6)].map((_, i) => <div key={i} className="skeleton skeleton-row" />)
        ) : inputMode === 'form' ? (
          meta.names.map((name) => (
            <div className="field" key={name}>
              <label htmlFor={name}>{shortLabel(name)}<Info text={meta.descriptions?.[name]} /></label>
              <input id={name} type="number" step="any" value={values[name]}
                     onChange={(e) => setValues((v) => ({ ...v, [name]: e.target.value }))} />
              <div className="hint num">range {meta.stats[name].min} – {meta.stats[name].max}</div>
            </div>
          ))
        ) : (
          <>
            <div className="live-badge" style={{ marginBottom: 12 }}>
              <span className="pulse" /> live — drag to see risk update in real time
            </div>
            {meta.names.map((name) => {
              const st = meta.stats[name]
              const val = values[name] === '' ? st.median : parseFloat(values[name])
              const step = (st.max - st.min) / 200 || 0.001
              return (
                <div className="slider-field" key={name}>
                  <div className="slider-head">
                    <label>{shortLabel(name)}<Info text={meta.descriptions?.[name]} /></label>
                    <span className="slider-val num">{Number(val).toFixed(4)}</span>
                  </div>
                  <input className="rng" type="range" min={st.min} max={st.max} step={step}
                         value={val} onChange={(e) => onSlider(name, e.target.value)} />
                </div>
              )
            })}
          </>
        )}

        {inputMode === 'form' && (
          <div style={{ display: 'flex', gap: 12, alignItems: 'center', marginTop: 6, flexWrap: 'wrap' }}>
            <button className="btn primary" onClick={runPredict} disabled={!allFilled || busy}>
              {busy ? <><span className="spinner" /> Scoring…</> : 'Score this company'}
            </button>
            <div className="seg" role="group" aria-label="Decision threshold">
              <button className={thresholdMode === 'tuned' ? 'on' : ''} onClick={() => setThresholdMode('tuned')}>tuned threshold</button>
              <button className={thresholdMode === 'default' ? 'on' : ''} onClick={() => setThresholdMode('default')}>0.5</button>
            </div>
          </div>
        )}
        {error && <div className="error-note">{error}</div>}
      </section>

      <section className="panel" aria-label="Verdict">
        <div className="panel-title">Verdict</div>
        {busy ? (
          <div className="skeleton skeleton-block" />
        ) : result ? (
          <Verdict result={result} explanation={explanation} explaining={explaining}
                   explainModel={explainModel} setExplainModel={setExplainModel} liveMode={inputMode === 'sliders'} />
        ) : (
          <div className="empty-note">
            {inputMode === 'sliders'
              ? 'Drag any slider to score in real time.'
              : 'Fill in the ratios (or load an example) and score the company.'}
          </div>
        )}
      </section>
    </div>
  )
}

function Verdict({ result, explanation, explaining, explainModel, setExplainModel, liveMode }) {
  const conf = result.conformal
  const confColor = conf.ambiguous ? 'var(--risk-elevated)' : (conf.prediction_set[0] === 'bankrupt' ? 'var(--risk-high)' : 'var(--risk-low)')

  return (
    <>
      <div className="verdict-head">
        <Gauge value={result.average_probability} tier={result.risk_tier} />
        <div className="verdict-meta">
          <div className={`verdict-tier ${TIER_CLASS[result.risk_tier]}`}>{TIER_TEXT[result.risk_tier]}</div>
          <div className="verdict-sub">
            {result.votes_bankrupt} of {result.n_models} models flag this company
            {result.consensus === 'split' ? ' — split decision' : ''}
          </div>
          <div className="conf-chip" title={`90% conformal prediction set (${conf.model})`}>
            <span className="dot" style={{ background: confColor }} />
            {conf.ambiguous ? `Uncertain — 90% set includes both outcomes`
              : `90% confident: ${conf.prediction_set[0] === 'bankrupt' ? 'bankrupt' : 'not bankrupt'}`}
          </div>
        </div>
      </div>

      <div className="vote-strip" aria-label="Model votes">
        {result.models.map((m) => (
          <div key={m.model_key} className="vote-cell" title={`${m.model_name}: ${(m.probability * 100).toFixed(1)}%`}
               style={{ background: m.prediction === 'bankrupt' ? 'var(--risk-high)' : 'var(--risk-low)' }}>
            {m.model_key.toUpperCase()}
          </div>
        ))}
      </div>
      <div className="vote-legend">red = flagged bankrupt at that model's threshold · green = clear</div>

      <div style={{ marginTop: 18 }}>
        {result.models.map((m) => (
          <div className="model-row" key={m.model_key}>
            <div className="model-name">{m.model_name}</div>
            <div className="prob-track">
              <div className="prob-fill" style={{ width: `${m.probability * 100}%`,
                background: m.prediction === 'bankrupt' ? 'var(--risk-high)' : 'var(--risk-low)' }} />
              <div className="prob-thresh" style={{ left: `${m.threshold * 100}%` }} title={`threshold ${m.threshold}`} />
            </div>
            <div className="prob-val num">{(m.probability * 100).toFixed(1)}%</div>
          </div>
        ))}
      </div>

      <div className="cf-box">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6 }}>
          <div className="panel-title" style={{ marginBottom: 0 }}>Why — SHAP contributions</div>
          <select className="select" value={explainModel} onChange={(e) => setExplainModel(e.target.value)} aria-label="Explain model">
            {result.models.map((m) => <option key={m.model_key} value={m.model_key}>{m.model_name}</option>)}
          </select>
        </div>
        {explaining && !explanation ? (
          <div className="skeleton skeleton-block" style={{ height: 160 }} />
        ) : explanation ? (
          <>
            <ShapWaterfall shap={explanation.shap} />
            {explanation.narrative && !liveMode && <Narrative narrative={explanation.narrative} />}
            <Counterfactual cf={explanation.counterfactual} />
          </>
        ) : (
          <div className="empty-note">Explanation loads with the prediction.</div>
        )}
      </div>
    </>
  )
}

function Narrative({ narrative }) {
  return (
    <div className="narrative">
      <div className="panel-title">Plain-English assessment</div>
      <div className="narrative-text">{narrative.text}</div>
      <div className="narrative-src">
        {narrative.source === 'llm'
          ? <><span className="live">● local LLM</span> · generated by Ollama</>
          : 'generated from SHAP factors · install Ollama for richer narratives'}
      </div>
    </div>
  )
}

function Counterfactual({ cf }) {
  if (!cf || !cf.applicable) return null
  return (
    <div className="cf-box">
      <div className="panel-title">What would lower the risk?</div>
      {cf.flipped ? (
        <div className="verdict-sub" style={{ marginBottom: 8 }}>
          These changes would move it below the threshold
          (<span className="num">{(cf.base_probability * 100).toFixed(1)}%</span> →
          <span className="num" style={{ color: 'var(--risk-low)' }}> {(cf.final_probability * 100).toFixed(1)}%</span>):
        </div>
      ) : (
        <div className="verdict-sub" style={{ marginBottom: 8 }}>
          No single-feature changes within observed ranges fully clear the flag; closest reached{' '}
          <span className="num">{(cf.final_probability * 100).toFixed(1)}%</span>.
        </div>
      )}
      {cf.changes.map((ch) => (
        <div className="cf-line" key={ch.feature}>
          <span className="cf-verb">{ch.direction}</span>
          <strong>{ch.feature.trim().replace(' (Yuan ??)', '')}</strong>
          <span className="cf-from">{ch.from}</span>
          <span className="cf-arrow">→</span>
          <span className="cf-to">{ch.to}</span>
        </div>
      ))}
    </div>
  )
}
