// Thin API client. In dev, Vite proxies /api to localhost:8000.
const BASE = '/api'

async function request(path, options = {}) {
  const res = await fetch(`${BASE}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  })
  if (!res.ok) {
    const body = await res.text()
    throw new Error(`API ${res.status}: ${body}`)
  }
  return res.json()
}

export const getSummary = () => request('/summary')
export const getFeatures = (variant) => request(`/features?variant=${variant}`)
export const postPredict = (payload) =>
  request('/predict', { method: 'POST', body: JSON.stringify(payload) })
export const postExplain = (payload) =>
  request('/explain', { method: 'POST', body: JSON.stringify(payload) })
export const postBatch = (payload) =>
  request('/batch', { method: 'POST', body: JSON.stringify(payload) })
