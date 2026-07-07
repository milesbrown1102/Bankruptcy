// Diverging bar chart of SHAP contributions. Positive (red) pushes toward
// bankruptcy, negative (green) pulls toward safe. Centered on a zero axis.
export default function ShapWaterfall({ shap }) {
  if (!shap || shap.error || !shap.contributions?.length) {
    return <div className="empty-note">No explanation available for this model.</div>
  }
  const maxAbs = Math.max(...shap.contributions.map((c) => Math.abs(c.shap))) || 1

  return (
    <div className="shap-wrap">
      <div className="shap-legend">
        <span><span className="sw" style={{ background: 'var(--risk-high)' }} />raises risk</span>
        <span><span className="sw" style={{ background: 'var(--risk-low)' }} />lowers risk</span>
        <span>base rate {(shap.base_value * 100).toFixed(1)}%</span>
      </div>
      {shap.contributions.map((c) => {
        const pct = (Math.abs(c.shap) / maxAbs) * 48 // max half-width %
        const pos = c.shap > 0
        return (
          <div className="shap-row" key={c.feature}>
            <div className="shap-name" title={c.feature}>
              {c.feature.trim().replace(' (Yuan ??)', '')}
            </div>
            <div className="shap-bar-track">
              <div className="shap-axis" />
              <div
                className={`shap-bar ${pos ? 'pos' : 'neg'}`}
                style={{ width: `${pct}%` }}
                title={`SHAP ${c.shap > 0 ? '+' : ''}${c.shap}`}
              />
            </div>
            <div className="shap-val" style={{ color: pos ? 'var(--risk-high)' : 'var(--risk-low)' }}>
              {c.shap > 0 ? '+' : ''}{c.shap.toFixed(3)}
            </div>
          </div>
        )
      })}
    </div>
  )
}
