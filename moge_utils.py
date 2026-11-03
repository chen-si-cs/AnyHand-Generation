import torch
import numpy as np
from moge.model.v2 import MoGeModel
import cv2

_MOGE_STATE = {"model": None, "device": None, "name": None}  # <-- add this


def warmup_moge(model_name: str = "/moge-2-vitl-normal", device=None):
    """Optional: call once at startup to ensure the big model is loaded."""
    _get_moge(model_name=model_name, device=device)
    print("[MoGe] Model Loaded.")
    return _MOGE_STATE["model"], _MOGE_STATE["device"]


def _get_moge(model_name: str = "/moge-2-vitl-normal", device=None):
    """
    Load MoGe once and reuse. Returns (model, torch.device).
    """
    global _MOGE_STATE
    if _MOGE_STATE["model"] is None or _MOGE_STATE["name"] != model_name:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        model = MoGeModel.from_pretrained(model_name).to(device).eval()
        _MOGE_STATE.update(
            {"model": model, "device": torch.device(device), "name": model_name}
        )
        print(f"[MoGe] loaded {model_name} on {_MOGE_STATE['device']}")
    return _MOGE_STATE["model"], _MOGE_STATE["device"]


def moge_depth_from_rgb(
    rgb_uint8: np.ndarray, model_name: str = "/moge-2-vitl-normal"
) -> np.ndarray:
    """
    Args:
        rgb_uint8: (H, W, 3|4) uint8, RGB or RGBA.
    Returns:
        depth: (H, W) float32, MoGe depth prediction.
    """
    # normalize input
    if rgb_uint8.ndim == 2:
        rgb_uint8 = np.stack([rgb_uint8] * 3, axis=-1)
    if rgb_uint8.shape[-1] == 4:
        rgb_uint8 = rgb_uint8[..., :3]  # drop alpha
    img = rgb_uint8.astype(np.float32) / 255.0  # RGB in [0,1]

    model, device = _get_moge(model_name=model_name)
    x = torch.from_numpy(img).permute(2, 0, 1).to(device)  # (3,H,W), float32
    with torch.no_grad():
        out = model.infer(x)  # MoGe v2 expects (3,H,W)
    depth = out["depth"]
    if torch.is_tensor(depth):
        depth = depth.detach().cpu().numpy()
    return depth.astype(np.float32)


# def _ring_masks(mask01: np.ndarray, ring: int = 3) -> tuple[np.ndarray, np.ndarray]:
#     """
#     Get a one-pixel 'inner ring' just inside the hand, and an 'outer ring' just outside.
#     Returns boolean masks (inner, outer).
#     """
#     m = (mask01 > 0.5).astype(np.uint8)
#     k = np.ones((ring, ring), np.uint8)
#     dil = cv2.dilate(m, k, iterations=1)
#     ero = cv2.erode(m,  k, iterations=1)
#     inner = (m & (~ero)) > 0
#     outer = ((dil & (~m)) > 0)
#     return inner, outer


def _ring_masks(mask01: np.ndarray, ring: int = 3):
    m = (mask01 > 0).astype(np.uint8)
    k = np.ones((ring, ring), np.uint8)
    dil = cv2.dilate(m, k, iterations=1)
    ero = cv2.erode(m, k, iterations=1)
    inner = (m & (~ero)) > 0
    outer = (dil & (~m)) > 0

    # Fallback widening if empty
    if not inner.any() or not outer.any():
        k2 = np.ones((max(1, ring * 2), max(1, ring * 2)), np.uint8)
        dil2 = cv2.dilate(m, k2, iterations=1)
        ero2 = cv2.erode(m, k2, iterations=1)
        inner = (m & (~ero2)) > 0
        outer = (dil2 & (~m)) > 0
    return inner, outer


def adjust_bg_depth_layering(
    bg_depth: np.ndarray,
    hand_depth: np.ndarray,
    mask01: np.ndarray,
    mode: str = "dealign",
    target_gap: float = 0.05,  # meters
    ring_px: int = 3,
    fit_scale: bool = False,
    sanitize_bg: bool = True,
    bg_range_mode: str = "percentile",  # "percentile" | "global" | "iqr"
    bg_target_range: tuple = (
        0.0,
        10.0,
    ),  # map background into this [min,max] in meters
    bg_pct: tuple = (1.0, 99.0),
    enforce_positive: bool = True,
    pos_floor: float = 0.0,
    pos_margin: float = 1,
    enforce_positive_after: bool = True,
    post_clip_range=None,
) -> np.ndarray:
    """
    Align or de-align background depth relative to hand depth at the boundary.

    mode:
      - "align"   : force bg ≈ hand (gap≈0)
      - "dealign" : force bg ≈ hand + target_gap (bg farther)
      - "none"    : no adjustment
    """
    if mode not in ("align", "dealign"):
        # no adjustment requested — but still ensure size match for downstream use
        H, W = hand_depth.shape[:2]
        if bg_depth.shape[:2] != (H, W):
            bg_depth = cv2.resize(
                bg_depth.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR
            )
            # if sanitize_bg:
            #     bg_depth = _clamp_and_rescale_bg(bg_depth, mask01=None,
            #                                     target=bg_target_range,
            #                                     mode=bg_range_mode,
            #                                     pct=bg_pct)
        if sanitize_bg:
            bg_depth = _clamp_and_rescale_bg(
                bg_depth,
                mask01=None,
                target=bg_target_range,
                mode=bg_range_mode,
                pct=bg_pct,
            )
        if enforce_positive:
            bg_depth = _shift_bg_positive(
                bg_depth, mask01=None, floor=pos_floor, margin=pos_margin
            )
        return bg_depth.astype(np.float32)

    # --- Ensure same spatial size for all inputs ---
    H, W = hand_depth.shape[:2]
    if bg_depth.shape[:2] != (H, W):
        bg_depth = cv2.resize(
            bg_depth.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR
        )
    if mask01.shape[:2] != (H, W):
        mask01 = cv2.resize(
            (mask01 > 0.5).astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST
        )

    # if sanitize_bg:
    #     bg_depth = _clamp_and_rescale_bg(bg_depth, mask01=mask01,
    #                                      target=bg_target_range,
    #                                      mode=bg_range_mode,
    #                                      pct=bg_pct)

    if sanitize_bg:
        # 1) rescale BG to bounded range
        bg_depth = _clamp_and_rescale_bg(
            bg_depth,
            mask01=mask01,
            target=bg_target_range,
            mode=bg_range_mode,
            pct=bg_pct,
        )
    if enforce_positive:
        # 2) shift BG so min(BG) >= pos_floor, then 3) do boundary dealign/align
        bg_depth = _shift_bg_positive(
            bg_depth, mask01=mask01, floor=pos_floor, margin=pos_margin
        )

    # --- Build boundary rings safely ---
    inner, outer = _ring_masks(mask01.astype(np.uint8), ring=ring_px)
    if not inner.any() or not outer.any():
        # Fallback: use hand pixels vs non-hand pixels
        inner = mask01 > 0
        outer = mask01 == 0
        if not inner.any() or not outer.any():
            # Nothing to compute — return resized bg_depth unchanged
            return bg_depth.astype(np.float32)

    # --- Robust medians at boundary ---
    m_hand = float(np.median(hand_depth[inner]))
    m_bg = float(np.median(bg_depth[outer]))

    # Optional robust scale fit (IQR-based)
    s = 1.0
    if fit_scale:
        qh = np.percentile(hand_depth[inner], [25, 75])
        qb = np.percentile(bg_depth[outer], [25, 75])
        denom = qb[1] - qb[0]
        if abs(denom) > 1e-6:
            s = (qh[1] - qh[0]) / denom

    desired_bg_at_boundary = m_hand + (target_gap if mode == "dealign" else 0.0)
    t = desired_bg_at_boundary - (s * m_bg)

    # out = (bg_depth * s + t).astype(np.float32)
    # 3) apply boundary align/dealign
    out = (bg_depth * s + t).astype(np.float32)

    # 4) (NEW) keep BG non-negative *after* the global shift
    if enforce_positive_after:
        out = _shift_bg_positive(out, mask01=mask01, floor=pos_floor, margin=pos_margin)

    # Optional global clip after align/dealign
    if post_clip_range is not None:
        lo, hi = float(post_clip_range[0]), float(post_clip_range[1])
        out = np.clip(out, lo, hi, out=out)

    return out


def compose_hand_bg_depth(
    hand_depth: np.ndarray, bg_depth: np.ndarray, mask01: np.ndarray
) -> np.ndarray:
    """
    Replace background (mask==0) with bg_depth and keep hand (mask==1) from SAPIEN.
    """
    if bg_depth.shape != hand_depth.shape:
        bg_depth = cv2.resize(
            bg_depth,
            (hand_depth.shape[1], hand_depth.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
    comp = np.where(mask01 > 0.5, hand_depth, bg_depth).astype(np.float32)
    return comp


def _clamp_and_rescale_bg(
    bg_depth: np.ndarray,
    mask01: np.ndarray,
    target: tuple = (0.0, 10.0),
    mode: str = "percentile",
    pct: tuple = (1.0, 99.0),
    rescale_when: str = "auto",  # "auto" | "always" | "never"
    auto_tol: float = 0.05,  # meters tolerance around target bounds
) -> np.ndarray:
    """
    Robustly map background depth to a bounded metric range before alignment.
    - If mask01 is given, statistics are computed on BG pixels (mask==0).
    - mode:
        * "percentile": use low/high percentiles (default)
        * "global":     use min/max of BG
        * "iqr":        use Tukey range around IQR (q1-1.5*IQR, q3+1.5*IQR)
    - rescale_when:
        * "auto"   : only rescale if the estimated BG range lies outside `target`
        * "always" : always rescale into `target` (previous behavior)
        * "never"  : never rescale; only clip to `target`
    - auto_tol: slack (in meters) when deciding if the range is outside target
    """
    d = bg_depth.astype(np.float32).copy()
    finite = np.isfinite(d)
    if mask01 is not None and mask01.shape[:2] == d.shape[:2]:
        bgmask = (mask01 <= 0.5) & finite
    else:
        bgmask = finite
    if not np.any(bgmask):
        bgmask = finite

    vals = d[bgmask]
    if vals.size == 0:
        return np.clip(np.where(finite, d, target[1]), min(target), max(target)).astype(
            np.float32
        )

    if mode == "percentile":
        lo, hi = np.percentile(vals, [float(pct[0]), float(pct[1])])
    elif mode == "global":
        lo, hi = float(np.min(vals)), float(np.max(vals))
    elif mode == "iqr":
        q1, q3 = np.percentile(vals, [25.0, 75.0])
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    else:
        lo, hi = np.percentile(vals, [1.0, 99.0])

    # Guard against degenerate or nonsensical ranges
    if (not np.isfinite(lo)) or (not np.isfinite(hi)) or (hi <= lo + 1e-6):
        lo, hi = float(target[0]), float(target[1])

    t0, t1 = float(min(target)), float(max(target))
    # s = (t1 - t0) / max(hi - lo, 1e-6)
    # d = (d - lo) * s + t0
    # d[~finite] = t1
    # d = np.clip(d, t0, t1)
    # return d.astype(np.float32)

    # Decide rescale policy
    do_rescale = False
    if rescale_when == "always":
        do_rescale = True
    elif rescale_when == "never":
        do_rescale = False
    else:  # "auto"
        # rescale only if estimated range is outside the target by more than tolerance
        if (lo < (t0 - auto_tol)) or (hi > (t1 + auto_tol)):
            do_rescale = True

    if do_rescale:
        # Affine rescale into target window
        s = (t1 - t0) / max(hi - lo, 1e-6)
        d = (d - lo) * s + t0
        d[~finite] = t1
        d = np.clip(d, t0, t1)
    else:
        # Keep original metric depths; only clip outliers and set invalids
        d[~finite] = t1
        d = np.clip(d, t0, t1)
    return d.astype(np.float32)


def _shift_bg_positive(
    bg_depth: np.ndarray, mask01: np.ndarray, floor: float = 0.0, margin: float = 1e-4
) -> np.ndarray:
    d = bg_depth.astype(np.float32).copy()
    finite = np.isfinite(d)
    if mask01 is not None and mask01.shape[:2] == d.shape[:2]:
        bgmask = (mask01 <= 0.5) & finite
    else:
        bgmask = finite
    if not np.any(bgmask):
        bgmask = finite
    if not np.any(bgmask):
        return d  # nothing to do
    min_bg = float(np.min(d[bgmask]))
    thresh = float(floor) + float(margin)
    if min_bg < thresh:
        bias = thresh - min_bg
        d[finite] += bias
    return d
