import { useMemo, useRef, useState } from 'react'
import { postBatch } from '../api.js'

// Minimal CSV parser (handles quoted fields + commas). For portfolio scope
// this is fine; swap for papaparse if you need edge-case robustness.
function parseCSV(text) {
  const rows = []
  let field = '', row = [], inQuotes = false
  for (let i = 0; i < text.length; i++) {
    const ch = text[i]
    if (inQuotes) {
      if (ch === '"' && text[i + 1] === '"') { field += '"'; i++ }
      else if (ch === '"') inQuotes = false
      else field += ch
    } else {
      if (ch === '"') inQuotes = true
      else if (ch === ',') { row.push(field); field = '' }
      else if (ch === '\n' || ch === '\r') {
        if (field !== '' || row.length) { row.push(field); rows.push(row); row = []; field = '' }
        if (ch === '\r' && text[i + 1] === '\n') i++
      } else field += ch
    }
  }
  if (field !== '' || row.length) { row.push(field); rows.push(row) }
  return rows
}

const TIER_ORDER = { high: 0, elevated: 1, moderate: 2, low: 3 }

export default function Batch({ variant, summary }) {
  const [modelKey, setModelKey] = useState('rf')
  const [thresholdMode, setThresholdMode] = useState('tuned')
  const [result, setResult] = useState(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)
  const [over, setOver] = useState(false)
  const [sortKey, setSortKey] = useState('probability')
  const [sortDir, setSortDir] = useState('desc')
  const inputRef = useRef()

  const models = summary ? summary[variant] : null

  async function handleFile(file) {
    setError(null); setResult(null); setBusy(true)
    try {
      const text = await file.text()
      const parsed = parseCSV(text).filter((r) => r.some((c) => c.trim() !== ''))
      if (parsed.length < 2) throw new Error('CSV needs a header row and at least one data row.')
      const header = parsed[0].map((h) => h.trim())
      const rows = parsed.slice(1).map((cells) => {
        const obj = {}
        header.forEach((h, i) => {
          const v = parseFloat(cells[i])
          if (!Number.isNaN(v)) obj[h] = v
        })
        return obj
      })
      const res = await postBatch({ variant, model_key: modelKey, rows, threshold_mode: thresholdMode })
      setResult(res)
    } catch (e) { setError(e.message) } finally { setBusy(false) }
  }

  const sorted = useMemo(() => {
    if (!result) return []
    const arr = [...result.results]
    arr.sort((a, b) => {
      let av = a[sortKey], bv = b[sortKey]
      if (sortKey === 'risk_tier') { av = TIER_ORDER[a.risk_tier]; bv = TIER_ORDER[b.risk_tier] }
      return sortDir === 'desc' ? bv - av : av - bv
    })
    return arr
  }, [result, sortKey, sortDir])

  function toggleSort(key) {
    if (sortKey === key) setSortDir((d) => (d === 'desc' ? 'asc' : 'desc'))
    else { setSortKey(key); setSortDir('desc') }
  }

  function downloadCSV() {
    if (!result) return
    const lines = ['row,probability,prediction,risk_tier']
    result.results.forEach((r) => lines.push(`${r.row},${r.probability},${r.prediction},${r.risk_tier}`))
    const blob = new Blob([lines.join('\n')], { type: 'text/csv' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url; a.download = 'bankruptcy_scores.csv'; a.click()
    URL.revokeObjectURL(url)
  }

  return (
    <section className="panel">
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12, flexWrap: 'wrap', gap: 10 }}>
        <div className="panel-title" style={{ marginBottom: 0 }}>Batch scoring — upload a CSV of companies</div>
        <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
          <select className="select" value={modelKey} onChange={(e) => setModelKey(e.target.value)} aria-label="Model">
            {models && Object.entries(models).map(([k, m]) => <option key={k} value={k}>{m.name}</option>)}
          </select>
          <div className="seg" role="group" aria-label="Threshold">
            <button className={thresholdMode === 'tuned' ? 'on' : ''} onClick={() => setThresholdMode('tuned')}>tuned</button>
            <button className={thresholdMode === 'default' ? 'on' : ''} onClick={() => setThresholdMode('default')}>0.5</button>
          </div>
        </div>
      </div>

      <div
        className={`drop ${over ? 'over' : ''}`}
        onClick={() => inputRef.current.click()}
        onDragOver={(e) => { e.preventDefault(); setOver(true) }}
        onDragLeave={() => setOver(false)}
        onDrop={(e) => { e.preventDefault(); setOver(false); if (e.dataTransfer.files[0]) handleFile(e.dataTransfer.files[0]) }}
      >
        <input ref={inputRef} type="file" accept=".csv" onChange={(e) => e.target.files[0] && handleFile(e.target.files[0])} />
        {busy ? <><span className="spinner" /> Scoring…</> : 'Drop a CSV here or click to browse'}
        <div className="hint num" style={{ marginTop: 8 }}>
          columns must match the selected model's required feature names
        </div>
      </div>

      {error && <div className="error-note">{error}</div>}

      {result && (
        <>
          <div className="batch-summary" style={{ marginTop: 16 }}>
            <span>scored <b>{result.n_scored}</b></span>
            <span>flagged <b>{result.n_flagged}</b></span>
            {result.n_skipped > 0 && <span>skipped <b>{result.n_skipped}</b> (missing columns)</span>}
            <span style={{ marginLeft: 'auto' }}>
              <button className="btn ghost" onClick={downloadCSV}>Download results CSV</button>
            </span>
          </div>
          <table className="batch-table">
            <thead>
              <tr>
                <th onClick={() => toggleSort('row')}>Row</th>
                <th onClick={() => toggleSort('probability')}>P(bankrupt) {sortKey === 'probability' ? (sortDir === 'desc' ? '↓' : '↑') : ''}</th>
                <th onClick={() => toggleSort('risk_tier')}>Risk tier</th>
                <th>Prediction</th>
              </tr>
            </thead>
            <tbody>
              {sorted.map((r) => (
                <tr key={r.row}>
                  <td>{r.row}</td>
                  <td>{(r.probability * 100).toFixed(1)}%</td>
                  <td><span className={`pill ${r.risk_tier}`}>{r.risk_tier}</span></td>
                  <td style={{ color: r.prediction === 'bankrupt' ? 'var(--risk-high)' : 'var(--muted)' }}>
                    {r.prediction === 'bankrupt' ? 'bankrupt' : 'not bankrupt'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          <div className="footnote">
            Ranked by predicted bankruptcy probability. Click a column header to re-sort.
            Required columns for this model are shown by the API; rows missing any are skipped.
          </div>
        </>
      )}
    </section>
  )
}
