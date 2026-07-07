"""
Explainability + uncertainty services for the API.

  - shap_values_for(): per-prediction SHAP contributions (which features
    pushed THIS company toward / away from bankruptcy)
  - counterfactual_for(): minimal single-feature nudges that would flip a
    flagged company below its decision threshold (a lightweight, transparent
    greedy search — not full DiCE, but honest and fast)
  - conformal_pvalue(): split-conformal p-value + prediction set for a point
"""

import numpy as np
import pandas as pd
import shap


def _get_estimator(pipeline):
    """Return the final classifier step from a (possibly imblearn) pipeline."""
    return pipeline.steps[-1][1]


def _transform_through(pipeline, X):
    """Apply every step except the final classifier."""
    Xt = X
    for _, step in pipeline.steps[:-1]:
        # SMOTE has no transform at predict time; skip resamplers
        if hasattr(step, "transform"):
            Xt = step.transform(Xt)
    return Xt


def shap_values_for(pipeline, background, sample_df, feature_names, max_features=9):
    """
    Compute SHAP values for one sample. Uses TreeExplainer for tree models
    (fast, exact) and KernelExplainer as a fallback for the rest.
    Returns a list of {feature, value, shap} sorted by |shap| desc, plus the
    model's base value, all in probability space where possible.
    """
    estimator = _get_estimator(pipeline)
    model_type = type(estimator).__name__

    # Transform the sample + background through the pre-classifier steps so
    # the explainer sees what the classifier actually sees.
    bg_trans = _transform_through(pipeline, background)
    sample_trans = _transform_through(pipeline, sample_df)

    tree_models = {"RandomForestClassifier", "GradientBoostingClassifier",
                   "DecisionTreeClassifier", "XGBClassifier"}

    try:
        if model_type in tree_models:
            explainer = shap.TreeExplainer(estimator, feature_names=feature_names)
            sv = explainer.shap_values(sample_trans)
            base = explainer.expected_value
            # Binary classifiers may return a list [neg, pos] or a 3D array
            if isinstance(sv, list):
                sv_pos = np.asarray(sv[1])[0]
                base_pos = float(base[1] if np.ndim(base) else base)
            else:
                arr = np.asarray(sv)
                if arr.ndim == 3:          # (n, features, classes)
                    sv_pos = arr[0, :, 1]
                    base_pos = float(base[1] if np.ndim(base) else base)
                else:
                    sv_pos = arr[0]
                    base_pos = float(base if np.ndim(base) == 0 else base[0])
        else:
            # Model-agnostic fallback; predict positive-class probability
            f = lambda data: pipeline.predict_proba(
                pd.DataFrame(data, columns=feature_names))[:, 1]
            explainer = shap.KernelExplainer(
                f, shap.sample(background, 40, random_state=0))
            sv_pos = np.asarray(explainer.shap_values(sample_df, nsamples=100))[0]
            base_pos = float(explainer.expected_value)
    except Exception as e:
        return {"error": f"SHAP failed for {model_type}: {e}", "contributions": [], "base_value": None}

    contributions = [
        {"feature": feature_names[i],
         "value": round(float(sample_df.iloc[0, i]), 5),
         "shap": round(float(sv_pos[i]), 5)}
        for i in range(len(feature_names))
    ]
    contributions.sort(key=lambda c: abs(c["shap"]), reverse=True)
    return {
        "base_value": round(base_pos, 5),
        "contributions": contributions[:max_features],
    }


def counterfactual_for(pipeline, sample_df, feature_names, stats, threshold,
                        max_changes=3, steps=25):
    """
    Greedy search: find up to `max_changes` single-feature adjustments that
    would move the model's probability below its decision threshold. Each
    feature is swept across its observed [min, max] range; we pick the change
    that most reduces probability, apply it, and repeat.

    Returns a list of {feature, from, to, direction} plus the achieved prob.
    Only meaningful when the company is currently flagged (prob >= threshold).
    """
    current = sample_df.copy()
    base_prob = float(pipeline.predict_proba(current)[0][1])
    if base_prob < threshold:
        return {"applicable": False, "reason": "already below threshold",
                "changes": [], "final_probability": round(base_prob, 4)}

    changes = []
    prob = base_prob
    changed_features = set()

    for _ in range(max_changes):
        best = None
        for fi, fname in enumerate(feature_names):
            if fname in changed_features:
                continue
            lo, hi = stats[fname]["min"], stats[fname]["max"]
            if hi <= lo:
                continue
            for candidate in np.linspace(lo, hi, steps):
                trial = current.copy()
                trial.iloc[0, fi] = candidate
                p = float(pipeline.predict_proba(trial)[0][1])
                if best is None or p < best["prob"]:
                    best = {"fi": fi, "fname": fname, "candidate": float(candidate), "prob": p}
        if best is None:
            break
        # Apply the best single change
        current.iloc[0, best["fi"]] = best["candidate"]
        changed_features.add(best["fname"])
        direction = "increase" if best["candidate"] > sample_df.iloc[0, best["fi"]] else "decrease"
        changes.append({
            "feature": best["fname"],
            "from": round(float(sample_df.iloc[0, best["fi"]]), 5),
            "to": round(best["candidate"], 5),
            "direction": direction,
        })
        prob = best["prob"]
        if prob < threshold:
            break

    return {
        "applicable": True,
        "base_probability": round(base_prob, 4),
        "final_probability": round(prob, 4),
        "threshold": round(float(threshold), 4),
        "flipped": prob < threshold,
        "changes": changes,
    }


def conformal_pvalue(conformal_scores, proba):
    """
    Split-conformal p-values for both classes given a new point's positive
    probability. Returns p-values and a prediction set at alpha=0.1 (90%
    coverage). Larger p-value = the point conforms better to that class.

    Nonconformity: for positive class, 1 - proba; for negative, proba.
    """
    pos_scores = np.asarray(conformal_scores["cal_scores_pos"])
    neg_scores = np.asarray(conformal_scores["cal_scores_neg"])

    score_pos = 1 - proba
    score_neg = proba

    # p-value = fraction of calibration scores >= the new score (+1 smoothing)
    p_pos = (np.sum(pos_scores >= score_pos) + 1) / (len(pos_scores) + 1)
    p_neg = (np.sum(neg_scores >= score_neg) + 1) / (len(neg_scores) + 1)

    alpha = 0.1
    pred_set = []
    if p_pos > alpha:
        pred_set.append("bankrupt")
    if p_neg > alpha:
        pred_set.append("not_bankrupt")

    return {
        "p_bankrupt": round(float(p_pos), 4),
        "p_not_bankrupt": round(float(p_neg), 4),
        "prediction_set": pred_set,
        "coverage": 0.9,
        "ambiguous": len(pred_set) != 1,
    }
