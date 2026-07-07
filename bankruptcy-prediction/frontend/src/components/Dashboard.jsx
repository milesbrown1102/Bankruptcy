import { useMemo, useState } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer,
  LineChart, Line, CartesianGrid, Legend, ReferenceLine,
} from 'recharts'

const METRIC_OPTIONS = [
  { key: 'f1_bankrupt', label: 'F1 (bankrupt class)' },
  { key: 'recall_bankrupt', label: 'Recall (bankrupt class)' },
  { key: 'precision_bankrupt', label: 'Precision (bankrupt class)' },
  { key: 'pr_auc', label: 'PR-AUC' },
  { key: 'roc_auc', label: 'ROC-AUC' },
  { key: 'mcc', label: 'MCC' },
  { key: 'balanced_accuracy', label: 'Balanced accuracy' },
  { key: 'accuracy', label: 'Accuracy (for reference)' },
]

const MODEL_COLORS = {
  lr: '#5da9e8', rf: '#48c9b0', gb: '#f5b041', svm: '#ec7063',
  dt: '#af7ac5', mlp: '#52be80', xgb: '#53c8e8',
}

const tooltipStyle = {
  backgroundColor: '#16202f',
  border: '1px solid #2c3c56',
  borderRadius: 6,
  fontFamily: 'JetBrains Mono, monospace',
  fontSize: 12,
}

export default function Dashboard({ variant, summary }) {
  const [metric, setMetric] = useState('f1_bankrupt')
  const [curveModel, setCurveModel] = useState('rf')

  const models = summary ? summary[variant] : null

  const barData = useMemo(() => {
    if (!models) return []
    return Object.entries(models).map(([key, m]) => ({
      key,
      name: m.name,
      holdout: m.holdout[metric],
      cv: m.cv[metric],
    }))
  }, [models, metric])

  const best = useMemo(() => {
    if (!barData.length) return null
    return [...barData].sort((a, b) => b.holdout - a.holdout)[0]
  }, [barData])

  const rocData = useMemo(() => {
    if (!models) return []
    return models[curveModel].roc_curve.map((p) => ({ fpr: p.x, tpr: p.y }))
  }, [models, curveModel])

  const prData = useMemo(() => {
    if (!models) return []
    return models[curveModel].pr_curve.map((p) => ({ recall: p.x, precision: p.y }))
  }, [models, curveModel])

  if (!models) return <div className="panel"><div className="empty-note">Loading model analytics…</div></div>

  const baselinePrevalence = 220 / 6819

  return (
    <>
      <div className="metric-cards">
        <Card k="Best model (holdout F1)" v={bestByMetric(models, 'f1_bankrupt')} />
        <Card k="Best recall (bankrupt)" v={bestByMetric(models, 'recall_bankrupt')} />
        <Card k="Best PR-AUC" v={bestByMetric(models, 'pr_auc')} />
        <Card k="Naive baseline accuracy" v={{ val: '96.8%', who: 'always predict "not bankrupt"' }} />
      </div>

      <section className="panel">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
          <div className="panel-title" style={{ marginBottom: 0 }}>
            Model comparison — cross-validation vs. holdout
          </div>
          <select
            className="select"
            value={metric}
            onChange={(e) => setMetric(e.target.value)}
            aria-label="Metric"
          >
            {METRIC_OPTIONS.map((o) => (
              <option key={o.key} value={o.key}>{o.label}</option>
            ))}
          </select>
        </div>
        <ResponsiveContainer width="100%" height={320}>
          <BarChart data={barData} margin={{ top: 8, right: 8, left: -18, bottom: 0 }}>
            <CartesianGrid stroke="#1f2b3e" vertical={false} />
            <XAxis dataKey="key" tick={{ fill: '#8b99b0', fontSize: 12, fontFamily: 'JetBrains Mono' }}
                   tickFormatter={(k) => k.toUpperCase()} axisLine={{ stroke: '#2c3c56' }} tickLine={false} />
            <YAxis domain={[0, 1]} tick={{ fill: '#8b99b0', fontSize: 11, fontFamily: 'JetBrains Mono' }}
                   axisLine={false} tickLine={false} />
            <Tooltip
              contentStyle={tooltipStyle}
              cursor={{ fill: 'rgba(83,200,232,0.06)' }}
              formatter={(v, name) => [Number(v).toFixed(4), name === 'holdout' ? 'Holdout' : 'Cross-validation']}
              labelFormatter={(k) => barData.find((d) => d.key === k)?.name || k}
            />
            <Legend wrapperStyle={{ fontSize: 12, fontFamily: 'Inter' }}
                    formatter={(v) => (v === 'holdout' ? 'Holdout' : 'Cross-validation')} />
            <Bar dataKey="cv" fill="#2a6f86" radius={[3, 3, 0, 0]} />
            <Bar dataKey="holdout" radius={[3, 3, 0, 0]}
                 fill="#53c8e8" />
          </BarChart>
        </ResponsiveContainer>
        {best && (
          <div className="footnote">
            Best holdout {METRIC_OPTIONS.find((m) => m.key === metric).label}:{' '}
            <span className="num" style={{ color: 'var(--cyan)' }}>{best.name} · {best.holdout.toFixed(4)}</span>.
            Metrics use each model's tuned decision threshold; switch the metric
            to "Accuracy" to see why it flatters every model on data this imbalanced.
          </div>
        )}
      </section>

      <section className="panel">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
          <div className="panel-title" style={{ marginBottom: 0 }}>
            Discrimination curves — holdout set
          </div>
          <select className="select" value={curveModel} onChange={(e) => setCurveModel(e.target.value)} aria-label="Model">
            {Object.entries(models).map(([key, m]) => (
              <option key={key} value={key}>{m.name}</option>
            ))}
          </select>
        </div>
        <div className="chart-row">
          <div>
            <ResponsiveContainer width="100%" height={260}>
              <LineChart data={rocData} margin={{ top: 8, right: 12, left: -18, bottom: 4 }}>
                <CartesianGrid stroke="#1f2b3e" />
                <XAxis dataKey="fpr" type="number" domain={[0, 1]}
                       tick={{ fill: '#8b99b0', fontSize: 11, fontFamily: 'JetBrains Mono' }}
                       axisLine={{ stroke: '#2c3c56' }} tickLine={false}
                       label={{ value: 'False positive rate', position: 'insideBottom', offset: -2, fill: '#5a6880', fontSize: 11 }} />
                <YAxis dataKey="tpr" domain={[0, 1]}
                       tick={{ fill: '#8b99b0', fontSize: 11, fontFamily: 'JetBrains Mono' }}
                       axisLine={false} tickLine={false} />
                <Tooltip contentStyle={tooltipStyle}
                         formatter={(v) => Number(v).toFixed(3)}
                         labelFormatter={(l) => `FPR ${Number(l).toFixed(3)}`} />
                <ReferenceLine segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]} stroke="#2c3c56" strokeDasharray="4 4" />
                <Line type="monotone" dataKey="tpr" name="TPR" stroke={MODEL_COLORS[curveModel]}
                      strokeWidth={2} dot={false} activeDot={{ r: 4 }} />
              </LineChart>
            </ResponsiveContainer>
            <div className="footnote">ROC — AUC {models[curveModel].holdout.roc_auc.toFixed(3)}</div>
          </div>
          <div>
            <ResponsiveContainer width="100%" height={260}>
              <LineChart data={prData} margin={{ top: 8, right: 12, left: -18, bottom: 4 }}>
                <CartesianGrid stroke="#1f2b3e" />
                <XAxis dataKey="recall" type="number" domain={[0, 1]}
                       tick={{ fill: '#8b99b0', fontSize: 11, fontFamily: 'JetBrains Mono' }}
                       axisLine={{ stroke: '#2c3c56' }} tickLine={false}
                       label={{ value: 'Recall', position: 'insideBottom', offset: -2, fill: '#5a6880', fontSize: 11 }} />
                <YAxis dataKey="precision" domain={[0, 1]}
                       tick={{ fill: '#8b99b0', fontSize: 11, fontFamily: 'JetBrains Mono' }}
                       axisLine={false} tickLine={false} />
                <Tooltip contentStyle={tooltipStyle}
                         formatter={(v) => Number(v).toFixed(3)}
                         labelFormatter={(l) => `Recall ${Number(l).toFixed(3)}`} />
                <ReferenceLine y={baselinePrevalence} stroke="#2c3c56" strokeDasharray="4 4"
                               label={{ value: 'chance', fill: '#5a6880', fontSize: 10, position: 'right' }} />
                <Line type="monotone" dataKey="precision" name="Precision" stroke={MODEL_COLORS[curveModel]}
                      strokeWidth={2} dot={false} activeDot={{ r: 4 }} />
              </LineChart>
            </ResponsiveContainer>
            <div className="footnote">
              Precision–recall — AUC {models[curveModel].holdout.pr_auc.toFixed(3)}. On data
              this imbalanced (3.2% positive), this curve is more informative than ROC.
            </div>
          </div>
        </div>
      </section>

      <section className="panel">
        <div className="panel-title">Confusion matrices — holdout, tuned thresholds</div>
        <div className="cm-grid">
          {Object.entries(models).map(([key, m]) => (
            <ConfusionCard key={key} name={m.name} cm={m.holdout.confusion_matrix} />
          ))}
        </div>
        <div className="footnote">
          Rows are truth, columns are prediction. FN (missed bankruptcies, lower-left)
          is the costly cell in this domain.
        </div>
      </section>
    </>
  )
}

function bestByMetric(models, metricKey) {
  const entries = Object.values(models).map((m) => ({ name: m.name, v: m.holdout[metricKey] }))
  const top = entries.sort((a, b) => b.v - a.v)[0]
  return { val: top.v.toFixed(3), who: top.name }
}

function Card({ k, v }) {
  return (
    <div className="mcard">
      <div className="k">{k}</div>
      <div className="v">{v.val}</div>
      <div className="who">{v.who}</div>
    </div>
  )
}

function ConfusionCard({ name, cm }) {
  const [[tn, fp], [fn, tp]] = cm
  return (
    <div className="cm-card">
      <h4>{name}</h4>
      <div className="cm-table">
        <div className="cm-cell" style={{ background: 'rgba(56,201,139,0.12)', color: 'var(--risk-low)' }}>
          <span className="lab">TN</span>{tn}
        </div>
        <div className="cm-cell" style={{ background: 'rgba(240,169,59,0.12)', color: 'var(--risk-elevated)' }}>
          <span className="lab">FP</span>{fp}
        </div>
        <div className="cm-cell" style={{ background: 'rgba(240,67,92,0.16)', color: 'var(--risk-high)' }}>
          <span className="lab">FN</span>{fn}
        </div>
        <div className="cm-cell" style={{ background: 'rgba(83,200,232,0.12)', color: 'var(--cyan)' }}>
          <span className="lab">TP</span>{tp}
        </div>
      </div>
    </div>
  )
}
