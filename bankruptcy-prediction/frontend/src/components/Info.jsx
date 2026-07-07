import { useState } from 'react'

export default function Info({ text }) {
  const [open, setOpen] = useState(false)
  if (!text) return null
  return (
    <span
      className="info"
      onMouseEnter={() => setOpen(true)}
      onMouseLeave={() => setOpen(false)}
    >
      <button type="button" aria-label="What is this?" onFocus={() => setOpen(true)} onBlur={() => setOpen(false)}>?</button>
      {open && <span className="bubble" role="tooltip">{text}</span>}
    </span>
  )
}
