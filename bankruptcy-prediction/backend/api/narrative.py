"""
Plain-English risk narrative generation.

Uses a LOCAL, FREE LLM via Ollama (https://ollama.com) if it's available,
and falls back to a deterministic template if it isn't — so the app never
breaks for someone who hasn't installed Ollama.

To enable the LLM path:
    1. Install Ollama (one-time): https://ollama.com/download
    2. Pull a small model:  ollama pull llama3.2
    3. Ollama runs a server on localhost:11434 automatically.

If Ollama isn't running, narratives are still produced from a template that
reads the SHAP contributions directly — good enough to be useful, and it
degrades silently.
"""

import os
import json
import urllib.request
import urllib.error

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2")
OLLAMA_TIMEOUT = float(os.environ.get("OLLAMA_TIMEOUT", "30"))


def _clean_feature(name):
    return name.strip().replace(" (Yuan ??)", "").replace(" (Yuan)", "")


def _build_prompt(probability, risk_tier, prediction, contributions, model_name):
    raisers = [c for c in contributions if c["shap"] > 0][:4]
    lowerers = [c for c in contributions if c["shap"] < 0][:3]

    lines = [f"- {_clean_feature(c['feature'])}: value {c['value']}, "
             f"pushing risk {'UP' if c['shap'] > 0 else 'DOWN'} "
             f"(impact {abs(c['shap']):.3f})"
             for c in (raisers + lowerers)]
    factors = "\n".join(lines)

    return f"""You are a credit-risk analyst. Write a concise, factual 3-4 sentence assessment of a company's bankruptcy risk. Do not use bullet points, headers, or a greeting. Write in plain professional English for a business reader.

Model: {model_name}
Predicted bankruptcy probability: {probability:.1%}
Risk tier: {risk_tier}
Classification: {prediction.replace('_', ' ')}

Key financial factors driving this assessment (from SHAP analysis):
{factors}

Write the assessment now. Reference the specific factors above. Be direct about whether this company looks financially healthy or distressed, and why."""


def _call_ollama(prompt):
    payload = json.dumps({
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.3},
    }).encode("utf-8")

    req = urllib.request.Request(
        OLLAMA_URL, data=payload, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=OLLAMA_TIMEOUT) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data.get("response", "").strip()


def _template_narrative(probability, risk_tier, prediction, contributions, model_name):
    """Deterministic fallback when Ollama isn't available."""
    raisers = [c for c in contributions if c["shap"] > 0][:3]
    lowerers = [c for c in contributions if c["shap"] < 0][:2]

    tier_phrase = {
        "high": "a high risk of bankruptcy",
        "elevated": "an elevated risk of bankruptcy",
        "moderate": "a moderate risk of bankruptcy",
        "low": "a low risk of bankruptcy",
    }.get(risk_tier, "some risk of bankruptcy")

    parts = [f"This company shows {tier_phrase}, with a predicted probability of "
             f"{probability:.1%} ({model_name})."]

    if raisers:
        names = ", ".join(_clean_feature(c["feature"]) for c in raisers)
        parts.append(f"The main factors raising risk are {names}.")
    if lowerers:
        names = ", ".join(_clean_feature(c["feature"]) for c in lowerers)
        parts.append(f"Partially offsetting this, {names} pull the estimate toward safety.")

    parts.append(
        "Overall the model classifies this company as "
        f"{'likely to go bankrupt' if prediction == 'bankrupt' else 'financially stable'}."
    )
    return " ".join(parts)


def generate_narrative(probability, risk_tier, prediction, contributions, model_name):
    """
    Returns {"text": str, "source": "llm"|"template"}.
    Tries Ollama first; falls back to a template on any failure.
    """
    if contributions:
        prompt = _build_prompt(probability, risk_tier, prediction, contributions, model_name)
        try:
            text = _call_ollama(prompt)
            if text:
                return {"text": text, "source": "llm"}
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError, json.JSONDecodeError):
            pass  # fall through to template

    return {
        "text": _template_narrative(probability, risk_tier, prediction, contributions, model_name),
        "source": "template",
    }
