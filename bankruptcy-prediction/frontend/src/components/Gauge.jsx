// SVG half-arc gauge: average bankruptcy probability 0..1
const TIER_COLORS = {
  high: 'var(--risk-high)',
  elevated: 'var(--risk-elevated)',
  moderate: 'var(--risk-moderate)',
  low: 'var(--risk-low)',
}

export default function Gauge({ value, tier }) {
  const size = 168
  const stroke = 13
  const r = (size - stroke) / 2
  const cx = size / 2
  const cy = size / 2
  // Half-circle arc from 180° to 0°
  const circumference = Math.PI * r
  const filled = Math.max(0, Math.min(1, value)) * circumference
  const color = TIER_COLORS[tier] || 'var(--cyan)'

  const arcPath = `M ${stroke / 2} ${cy} A ${r} ${r} 0 0 1 ${size - stroke / 2} ${cy}`

  return (
    <svg
      width={size}
      height={size / 2 + 30}
      viewBox={`0 0 ${size} ${size / 2 + 30}`}
      role="img"
      aria-label={`Average bankruptcy probability ${(value * 100).toFixed(1)} percent`}
    >
      <path d={arcPath} fill="none" stroke="var(--line)" strokeWidth={stroke} strokeLinecap="round" />
      <path
        d={arcPath}
        fill="none"
        stroke={color}
        strokeWidth={stroke}
        strokeLinecap="round"
        strokeDasharray={`${filled} ${circumference}`}
        style={{ transition: 'stroke-dasharray 500ms ease, stroke 300ms' }}
      />
      <text
        x={cx}
        y={cy - 4}
        textAnchor="middle"
        fill="var(--text)"
        fontFamily="var(--font-mono)"
        fontSize="27"
        fontWeight="600"
      >
        {(value * 100).toFixed(1)}%
      </text>
      <text
        x={cx}
        y={cy + 17}
        textAnchor="middle"
        fill="var(--muted)"
        fontFamily="var(--font-mono)"
        fontSize="9.5"
        letterSpacing="1.5"
      >
        AVG P(BANKRUPT)
      </text>
    </svg>
  )
}
