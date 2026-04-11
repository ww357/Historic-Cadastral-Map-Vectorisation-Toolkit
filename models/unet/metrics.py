"""
APL (Added Path Length) and path-length metrics for boundary prediction.

APL = FN_length + FP_length  (symmetric, in skeleton pixels)
  FN: GT skeleton pixels not within τ of any predicted skeleton pixel
      → sections the user must draw in
  FP: predicted skeleton pixels not within τ of any GT skeleton pixel
      → sections the user must delete

Why this matters: a prediction offset by even a few pixels accumulates cost
in both directions simultaneously — the GT section is "missed" and the
predicted section is spurious — which matches exactly what a human experiences
when dragging a misplaced polygon edge.

Mending time estimate (log-linear, boundaries category):
  mending_time = 0.3421 * APL^0.647   R²=0.317
Use for relative comparison between model versions, not absolute scheduling.
Text (R²=0.734) and water (R²=0.619) have more reliable estimates if added later.
"""

import numpy as np
from skimage.morphology import skeletonize
from scipy.ndimage import distance_transform_edt


# Log-linear regression coefficients — boundaries category
MENDING_C = 0.3421
MENDING_K = 0.647


def compute_path_metrics(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    tau: int = 3,
    pixel_size_m: float | None = None,
) -> dict:
    """
    Compute APL and path-length F1 between a thresholded prediction and GT mask.

    Parameters
    ----------
    pred_mask    : binary uint8/bool array, already thresholded
    gt_mask      : binary uint8/bool array
    tau          : coverage tolerance in pixels (default 3)
    pixel_size_m : metres per pixel — if given, APL/FN/FP are reported in metres

    Returns
    -------
    dict with keys:
        apl               — Added Path Length (pixels or metres)
        fn_length         — GT skeleton pixels not covered by prediction
        fp_length         — Predicted skeleton pixels not covered by GT
        path_recall       — fraction of GT skeleton covered by prediction
        path_precision    — fraction of predicted skeleton covered by GT
        path_f1           — harmonic mean; use this for checkpointing
        est_mending_time  — estimated mending time in minutes
                            R²=0.317 — reliable for model comparison, not scheduling
    """
    gt_skel   = skeletonize(gt_mask   > 0)
    pred_skel = skeletonize(pred_mask > 0)

    gt_len   = int(gt_skel.sum())
    pred_len = int(pred_skel.sum())

    # Both empty → perfect
    if gt_len == 0 and pred_len == 0:
        return dict(apl=0.0, fn_length=0.0, fp_length=0.0,
                    path_recall=1.0, path_precision=1.0, path_f1=1.0,
                    est_mending_time=0.0)

    # Distance from every pixel to the nearest skeleton pixel
    dist_to_pred = (distance_transform_edt(~pred_skel)
                    if pred_len > 0 else np.full(gt_skel.shape, np.inf))
    dist_to_gt   = (distance_transform_edt(~gt_skel)
                    if gt_len   > 0 else np.full(pred_skel.shape, np.inf))

    fn_px = int((gt_skel   & (dist_to_pred > tau)).sum())   # missed GT
    fp_px = int((pred_skel & (dist_to_gt   > tau)).sum())   # spurious pred
    apl_px = fn_px + fp_px

    recall    = 1.0 - fn_px / gt_len   if gt_len   > 0 else (1.0 if pred_len == 0 else 0.0)
    precision = 1.0 - fp_px / pred_len if pred_len > 0 else (1.0 if gt_len   == 0 else 0.0)
    f1 = (2 * recall * precision / (recall + precision)
          if (recall + precision) > 0 else 0.0)

    scale = pixel_size_m if pixel_size_m else 1.0
    apl   = apl_px * scale
    fn    = fn_px  * scale
    fp    = fp_px  * scale

    est_time = MENDING_C * (apl ** MENDING_K) if apl > 0 else 0.0

    return dict(
        apl              = round(apl,  2),
        fn_length        = round(fn,   2),
        fp_length        = round(fp,   2),
        path_recall      = round(recall,    4),
        path_precision   = round(precision, 4),
        path_f1          = round(f1,        4),
        est_mending_time = round(est_time,  2),
    )
