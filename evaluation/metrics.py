"""
EventXBench evaluation metrics.

All functions accept plain Python lists and have no hard dependencies
on numpy or scikit-learn, though they may use them when available for
numerical stability.
"""

from __future__ import annotations

from collections import Counter
from typing import List, Optional, Sequence, Tuple


# ------------------------------------------------------------------ #
#  Macro-averaged F1                                                  #
# ------------------------------------------------------------------ #

def macro_f1(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Optional[List[str]] = None,
) -> float:
    """Macro-averaged F1 over *labels*.

    Parameters
    ----------
    y_true : sequence of str
        Ground-truth labels.
    y_pred : sequence of str
        Predicted labels (same length as *y_true*).
    labels : list of str, optional
        Label set.  If ``None``, derived from the union of *y_true*
        and *y_pred*.

    Returns
    -------
    float
        Macro-averaged F1 score in [0, 1].
    """
    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))

    f1_scores: list[float] = []
    for label in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        f1_scores.append(f1)

    return sum(f1_scores) / len(f1_scores) if f1_scores else 0.0


# ------------------------------------------------------------------ #
#  Precision at K (positive-class retrieval)                         #
# ------------------------------------------------------------------ #


def precision_at_k(
    y_true: Sequence[str],
    scores: Sequence[dict[str, float]],
    pos_class: str = "high_interest",
    k: int = 5,
) -> float:
    """Precision at K for the positive class.

    Ranks instances by the score of *pos_class* (descending), takes the
    top-K, and returns the fraction whose true label equals *pos_class*.

    Parameters
    ----------
    y_true : sequence of str
        Ground-truth labels.
    scores : sequence of dict
        Per-instance score dicts ``{label: confidence}``.  Must have the
        same length as *y_true*.
    pos_class : str
        The class to treat as positive.
    k : int
        Number of top-ranked instances to consider.

    Returns
    -------
    float
        Precision at K in ``[0, 1]``.  Returns 0 when *k* is 0 or when
        there are fewer than 1 instance(s).
    """
    n = len(y_true)
    if len(scores) != n:
        raise ValueError("y_true and scores must have the same length")
    if k <= 0 or n < 1:
        return 0.0
    k = min(k, n)
    ranked = sorted(
        zip(y_true, scores),
        key=lambda x: x[1].get(pos_class, 0.0),
        reverse=True,
    )
    topk = ranked[:k]
    return sum(1 for t, _ in topk if t == pos_class) / k


# ------------------------------------------------------------------ #
#  Accuracy                                                           #
# ------------------------------------------------------------------ #

def accuracy(y_true: Sequence, y_pred: Sequence) -> float:
    """Simple accuracy (fraction of exact matches)."""
    if len(y_true) == 0:
        return 0.0
    return sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)


# ------------------------------------------------------------------ #
#  Spearman rank correlation                                          #
# ------------------------------------------------------------------ #

def _rank(values: Sequence[float]) -> list[float]:
    """Assign average ranks to *values* (handles ties)."""
    n = len(values)
    indexed = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0] * n

    i = 0
    while i < n:
        j = i
        while j < n and values[indexed[j]] == values[indexed[i]]:
            j += 1
        avg_rank = (i + j + 1) / 2.0  # 1-based average
        for k in range(i, j):
            ranks[indexed[k]] = avg_rank
        i = j
    return ranks


def spearman_rho(x: Sequence[float], y: Sequence[float]) -> float:
    """Spearman rank correlation coefficient.

    Implemented via Pearson correlation on ranks so that no scipy
    dependency is required.
    """
    if len(x) != len(y):
        raise ValueError("x and y must have the same length")
    n = len(x)
    if n < 2:
        return 0.0

    rx = _rank(list(x))
    ry = _rank(list(y))

    mean_rx = sum(rx) / n
    mean_ry = sum(ry) / n

    cov = sum((a - mean_rx) * (b - mean_ry) for a, b in zip(rx, ry))
    std_x = sum((a - mean_rx) ** 2 for a in rx) ** 0.5
    std_y = sum((b - mean_ry) ** 2 for b in ry) ** 0.5

    if std_x == 0.0 or std_y == 0.0:
        return 0.0
    return cov / (std_x * std_y)


# ------------------------------------------------------------------ #
#  Quadratic Weighted Kappa                                           #
# ------------------------------------------------------------------ #

def quadratic_weighted_kappa(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    num_classes: int,
) -> float:
    """Quadratic Weighted Kappa for ordinal classification.

    Parameters
    ----------
    y_true, y_pred : sequence of int
        Integer class indices in ``[0, num_classes)``.
    num_classes : int
        Number of ordinal classes.
    """
    n = len(y_true)
    if n == 0:
        return 0.0

    # Observed confusion matrix
    O = [[0] * num_classes for _ in range(num_classes)]
    for t, p in zip(y_true, y_pred):
        O[t][p] += 1

    # Weight matrix (quadratic)
    W = [[0.0] * num_classes for _ in range(num_classes)]
    for i in range(num_classes):
        for j in range(num_classes):
            W[i][j] = (i - j) ** 2 / ((num_classes - 1) ** 2)

    # Expected matrix under independence
    hist_true = [0] * num_classes
    hist_pred = [0] * num_classes
    for t, p in zip(y_true, y_pred):
        hist_true[t] += 1
        hist_pred[p] += 1

    E = [[0.0] * num_classes for _ in range(num_classes)]
    for i in range(num_classes):
        for j in range(num_classes):
            E[i][j] = hist_true[i] * hist_pred[j] / n

    num = sum(W[i][j] * O[i][j] for i in range(num_classes) for j in range(num_classes))
    den = sum(W[i][j] * E[i][j] for i in range(num_classes) for j in range(num_classes))

    if den == 0.0:
        return 1.0
    return 1.0 - num / den


# ------------------------------------------------------------------ #
#  Unweighted Cohen's Kappa                                           #
# ------------------------------------------------------------------ #

def cohen_kappa(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    num_classes: int,
) -> float:
    """Unweighted Cohen's kappa for classification.

    Parameters
    ----------
    y_true, y_pred : sequence of int
        Integer class indices in ``[0, num_classes)``.
    num_classes : int
        Number of classes.
    """
    n = len(y_true)
    if n == 0:
        return 0.0

    hist_true = [0] * num_classes
    hist_pred = [0] * num_classes
    agree = 0
    for t, p in zip(y_true, y_pred):
        hist_true[t] += 1
        hist_pred[p] += 1
        if t == p:
            agree += 1

    po = agree / n
    pe = sum(hist_true[i] * hist_pred[i] for i in range(num_classes)) / (n * n)

    if pe == 1.0:
        return 1.0
    return (po - pe) / (1 - pe)


# ------------------------------------------------------------------ #
#  Mean Reciprocal Rank                                               #
# ------------------------------------------------------------------ #

def mrr(
    ranked_lists: Sequence[Sequence],
    gold_indices: Sequence,
) -> float:
    """Mean Reciprocal Rank.

    Parameters
    ----------
    ranked_lists : list of lists
        Each inner list is a ranked sequence of candidate IDs.
    gold_indices : list
        The correct ID for each query.

    Returns
    -------
    float
        MRR in [0, 1].
    """
    if len(ranked_lists) == 0:
        return 0.0

    total = 0.0
    for ranked, gold in zip(ranked_lists, gold_indices):
        for rank, item in enumerate(ranked, start=1):
            if item == gold:
                total += 1.0 / rank
                break
    return total / len(ranked_lists)


# ------------------------------------------------------------------ #
#  Direction accuracy                                                 #
# ------------------------------------------------------------------ #

def direction_accuracy(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    """Accuracy restricted to direction labels (up / down / flat)."""
    return accuracy(y_true, y_pred)


# ------------------------------------------------------------------ #
#  Derive direction and magnitude from delta_2h                       #
# ------------------------------------------------------------------ #

def derive_direction_magnitude(
    delta_2h: float,
) -> Tuple[str, str]:
    """Derive direction and magnitude labels from a 2-hour price delta.

    Thresholds follow the EventX label definitions:
      - direction:  up if delta > 0.02, down if delta < -0.02, else flat
      - magnitude:  small if |delta| <= 0.02, medium if 0.02 < |delta| <= 0.08,
                    large if |delta| > 0.08

    Returns
    -------
    (direction, magnitude) : tuple of str
    """
    if delta_2h > 0.02:
        direction = "up"
    elif delta_2h < -0.02:
        direction = "down"
    else:
        direction = "flat"

    abs_delta = abs(delta_2h)
    if abs_delta <= 0.02:
        magnitude = "small"
    elif abs_delta <= 0.08:
        magnitude = "medium"
    else:
        magnitude = "large"

    return direction, magnitude
