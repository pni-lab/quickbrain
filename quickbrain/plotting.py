"""quickbrain.plotting – one-call brain surface visualisation."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Union
from xml.etree import ElementTree as ET

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MplPath
from nilearn.plotting import plot_surf, plot_surf_stat_map
from nilearn.surface import vol_to_surf
from scipy.ndimage import gaussian_filter

from quickbrain._resources import get_curvature, get_mesh, normalize_resolution


def _make_figure(figsize, dpi=100) -> Figure:
    """Create a Figure with an Agg canvas that is NOT registered with pyplot.

    This avoids double-display in Jupyter (pyplot auto-shows tracked figures
    at cell end, which duplicates the explicit return-value display).
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    fig = Figure(figsize=figsize, dpi=dpi)
    FigureCanvasAgg(fig)
    return fig

_OVERLAY_DIR = Path(__file__).resolve().parent / "resources" / "overlays"

_ALLOWED_HEMIS = ("left", "right")
_ALLOWED_VIEWS = ("lateral", "medial")
_ALLOWED_SURF_TYPES = ("pial", "inflated")
_ALLOWED_BACKGROUNDS = ("white", "transparent")

_NILEARN_FORBIDDEN_KWARGS = frozenset({
    "surf_mesh",
    "stat_map",
    "engine",
    "title",
})

# Small padding (fraction of bbox size) added around the brain bbox so that
# the blur doesn't clip and the overlay has a tiny breathing room.
_BRAIN_BBOX_PAD_FRAC = 0.02


def _resolve_stat_map(
    stat_map,
    mesh: tuple[np.ndarray, np.ndarray],
    vol_to_surf_radius: float,
) -> np.ndarray:
    """Accept a per-vertex array or a Nifti-like volume and return a 1-D array."""
    if isinstance(stat_map, np.ndarray):
        return stat_map
    try:
        import nibabel as nib
        if isinstance(stat_map, (str, Path)):
            stat_map = nib.load(str(stat_map))
        if hasattr(stat_map, "get_fdata"):
            return np.asarray(
                vol_to_surf(stat_map, mesh, radius=vol_to_surf_radius, interpolation="linear", kind="ball", n_samples=10),
                dtype=np.float32,
            )
    except Exception:
        pass
    raise TypeError(
        "stat_map must be a numpy array (per-vertex), a path to a NIfTI file, "
        f"or a nibabel image. Got {type(stat_map)!r}."
    )


def _resolve_bg_map(
    bg_map: Union[str, float, np.ndarray, None],
    side: str,
    res: int,
    n_vertices: int,
) -> np.ndarray | None:
    """Return a per-vertex background array."""
    if bg_map is None:
        return None
    if isinstance(bg_map, np.ndarray):
        return bg_map
    if isinstance(bg_map, str) and bg_map == "curvature":
        return get_curvature(side=side, res=res)
    if isinstance(bg_map, (int, float)):
        val = float(bg_map)
        if not 0.0 <= val <= 1.0:
            raise ValueError(f"bg_map float must be in [0, 1], got {val}")
        return np.full(n_vertices, val, dtype=np.float32)
    raise TypeError(
        f"bg_map must be 'curvature', a float in [0,1], an ndarray, or None. Got {bg_map!r}."
    )


# ---------------------------------------------------------------------------
# SVG overlay helpers
# ---------------------------------------------------------------------------

def _get_overlay_path(
    overlay: str | Path | None,
    surf_type: str,
    side: str,
    view: str,
) -> tuple[Path | None, bool]:
    """Resolve overlay SVG path.

    Returns ``(path, mirror)`` where *mirror* is True when a
    left-hemisphere SVG is being reused for the right hemisphere (or vice
    versa).

    Search order for ``"default"``:
        1. ``overlays/{surf_type}/{side}_{view}.svg``
        2. ``overlays/{side}_{view}.svg``  (shared fallback, same hemi)
        3. ``overlays/{surf_type}/{opposite}_{view}.svg``  (mirror)
        4. ``overlays/{opposite}_{view}.svg``  (shared fallback, mirror)
    """
    if overlay is None:
        return None, False
    if overlay == "default":
        opposite = "right" if side == "left" else "left"
        other_surf = "inflated" if surf_type == "pial" else "pial"
        candidates: list[tuple[Path, bool]] = [
            # Exact match
            (_OVERLAY_DIR / surf_type / f"{side}_{view}.svg", False),
            # Other surf_type, same hemi
            (_OVERLAY_DIR / other_surf / f"{side}_{view}.svg", False),
            # Shared (no surf_type), same hemi
            (_OVERLAY_DIR / f"{side}_{view}.svg", False),
            # Exact match, opposite hemi (mirror)
            (_OVERLAY_DIR / surf_type / f"{opposite}_{view}.svg", True),
            # Other surf_type, opposite hemi (mirror)
            (_OVERLAY_DIR / other_surf / f"{opposite}_{view}.svg", True),
            # Shared, opposite hemi (mirror)
            (_OVERLAY_DIR / f"{opposite}_{view}.svg", True),
        ]
        for cand, mirror in candidates:
            if cand.is_file():
                return cand, mirror
        return None, False
    p = Path(overlay)
    return (p, False) if p.is_file() else (None, False)


def _parse_transform(t: str | None) -> np.ndarray:
    """Parse an SVG ``transform`` attribute into a 3×3 affine matrix."""
    mat = np.eye(3)
    if t is None:
        return mat
    for m in re.finditer(r'(matrix|translate|scale|rotate)\(([^)]+)\)', t):
        kind = m.group(1)
        args = [float(x) for x in re.split(r'[\s,]+', m.group(2).strip())]
        if kind == "matrix" and len(args) == 6:
            a, b, c, d, e, f = args
            mat = mat @ np.array([[a, c, e], [b, d, f], [0, 0, 1]])
        elif kind == "translate":
            tx = args[0]
            ty = args[1] if len(args) > 1 else 0
            mat = mat @ np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]])
        elif kind == "scale":
            sx = args[0]
            sy = args[1] if len(args) > 1 else sx
            mat = mat @ np.array([[sx, 0, 0], [0, sy, 0], [0, 0, 1]])
    return mat


def _transform_mpl_path(mpl_path: MplPath, mat: np.ndarray) -> MplPath:
    """Apply a 3×3 affine to an MplPath's vertices."""
    verts = mpl_path.vertices
    ones = np.ones((verts.shape[0], 1))
    hom = np.hstack([verts, ones])
    transformed = (mat @ hom.T).T[:, :2]
    return MplPath(transformed, mpl_path.codes)


def _parse_svg_paths(svg_path: Path) -> tuple[list, tuple[float, float, float, float] | None]:
    """Parse SVG with nested group transforms; return (paths, viewBox)."""
    tree = ET.parse(svg_path)
    root = tree.getroot()

    vb = root.get("viewBox")
    viewbox = None
    if vb:
        parts = [float(x) for x in vb.split()]
        if len(parts) == 4:
            viewbox = tuple(parts)

    paths: list[tuple[MplPath, str, float]] = []

    try:
        from svgpath2mpl import parse_path as svg_to_mpl
    except ImportError:
        return paths, viewbox

    def _collect(elem, ctm: np.ndarray):
        tag = elem.tag
        if "}" in tag:
            tag = tag.split("}", 1)[1]

        local_ctm = ctm
        t = elem.get("transform")
        if t:
            local_ctm = ctm @ _parse_transform(t)

        if tag == "path":
            d = elem.get("d")
            if d is None:
                return
            mpl_p: MplPath = svg_to_mpl(d)
            mpl_p = _transform_mpl_path(mpl_p, local_ctm)
            style = elem.get("style", "")
            stroke = "#000000"
            lw = 0.5
            for token in style.split(";"):
                k, _, v = token.partition(":")
                k, v = k.strip(), v.strip()
                if k == "stroke":
                    stroke = v
                elif k == "stroke-width":
                    try:
                        lw = float(v.replace("px", ""))
                    except ValueError:
                        pass
            paths.append((mpl_p, stroke, lw))
        else:
            for child in elem:
                _collect(child, local_ctm)

    _collect(root, np.eye(3))
    return paths, viewbox


# ---------------------------------------------------------------------------
# Pixel-space helpers
# ---------------------------------------------------------------------------

def _brain_bbox_from_axes(ax_pos, ih: int, iw: int) -> tuple[int, int, int, int]:
    """Convert a matplotlib Bbox (in figure-fraction coords) to pixel (y0, y1, x0, x1)."""
    x0 = int(ax_pos.x0 * iw)
    x1 = int(ax_pos.x1 * iw)
    y0 = int((1 - ax_pos.y1) * ih)
    y1 = int((1 - ax_pos.y0) * ih)
    # Add small padding
    pad_y = max(1, int((y1 - y0) * _BRAIN_BBOX_PAD_FRAC))
    pad_x = max(1, int((x1 - x0) * _BRAIN_BBOX_PAD_FRAC))
    return (max(y0 - pad_y, 0), min(y1 + pad_y, ih),
            max(x0 - pad_x, 0), min(x1 + pad_x, iw))


def _axes_pos_to_pixels_raw(ax_pos, ih: int, iw: int) -> tuple[int, int, int, int]:
    """Figure-fraction Bbox → pixel ``(y0, y1, x0, x1)`` (no extra padding)."""
    x0 = int(ax_pos.x0 * iw)
    x1 = int(np.ceil(ax_pos.x1 * iw))
    y0 = int((1 - ax_pos.y1) * ih)
    y1 = int(np.ceil((1 - ax_pos.y0) * ih))
    return (max(y0, 0), min(y1, ih), max(x0, 0), min(x1, iw))


def _union_pixel_bboxes(
    *boxes: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    y0 = min(b[0] for b in boxes)
    y1 = max(b[1] for b in boxes)
    x0 = min(b[2] for b in boxes)
    x1 = max(b[3] for b in boxes)
    return y0, y1, x0, x1


def _tight_figure_crop_bbox(
    brain_pixel_bbox: tuple[int, int, int, int],
    sibling_axis_positions: list,
    ih: int,
    iw: int,
    pad_px: int = 3,
) -> tuple[int, int, int, int]:
    """Bounding box that tightly wraps brain + colorbars (drops nilearn margins)."""
    regions = [brain_pixel_bbox]
    for pos in sibling_axis_positions:
        regions.append(_axes_pos_to_pixels_raw(pos, ih, iw))
    y0, y1, x0, x1 = _union_pixel_bboxes(*regions)
    y0 = max(y0 - pad_px, 0)
    y1 = min(y1 + pad_px, ih)
    x0 = max(x0 - pad_px, 0)
    x1 = min(x1 + pad_px, iw)
    if y1 <= y0 or x1 <= x0:
        return 0, ih, 0, iw
    return y0, y1, x0, x1


def _find_brain_bbox_pixels(
    rgba: np.ndarray,
    search_region: tuple[int, int, int, int] | None = None,
    min_pixels: int = 20,
) -> tuple[int, int, int, int]:
    """Return (y0, y1, x0, x1) tight bbox of the rendered brain.

    If *search_region* ``(y0, y1, x0, x1)`` is given the scan is limited to
    that rectangle (avoids picking up title / colorbar text).

    *min_pixels* is the minimum number of non-white pixels a row or column
    must have to count as part of the brain (filters out title text and
    anti-aliasing artefacts).
    """
    if search_region is not None:
        ry0, ry1, rx0, rx1 = search_region
        region = rgba[ry0:ry1, rx0:rx1]
    else:
        ry0, rx0 = 0, 0
        region = rgba

    gray = region[:, :, :3].mean(axis=2)
    mask = gray < 254
    row_counts = mask.sum(axis=1)
    col_counts = mask.sum(axis=0)
    rows = row_counts >= min_pixels
    cols = col_counts >= min_pixels
    if not rows.any():
        return 0, rgba.shape[0], 0, rgba.shape[1]
    y0, y1 = np.where(rows)[0][[0, -1]]
    x0, x1 = np.where(cols)[0][[0, -1]]
    return (int(y0) + ry0, int(y1) + 1 + ry0,
            int(x0) + rx0, int(x1) + 1 + rx0)


def _apply_post_blur(
    rgba: np.ndarray,
    sigma_px: float,
    brain_bbox: tuple[int, int, int, int],
    brain_mask: np.ndarray,
) -> np.ndarray:
    """Blur the full bounding box, then stamp back only brain pixels.

    *brain_mask* is a boolean array the same size as *rgba* — True where
    the brain is.
    """
    if sigma_px <= 0:
        return rgba
    y0, y1, x0, x1 = brain_bbox
    roi = rgba[y0:y1, x0:x1].astype(np.float32)
    roi_mask = brain_mask[y0:y1, x0:x1]

    blurred = gaussian_filter(roi, sigma=(sigma_px, sigma_px, 0))
    np.clip(blurred, 0, 255, out=blurred)

    mask3d = roi_mask[:, :, np.newaxis]
    roi[:] = np.where(mask3d, blurred, roi)
    rgba[y0:y1, x0:x1] = roi.astype(np.uint8)
    return rgba


def _rasterize_figure(fig: Figure, *, draw: bool = True) -> np.ndarray:
    """Return the canvas as an RGBA uint8 array (H, W, 4).

    Set *draw=False* if `fig.canvas.draw()` has already been called and the
    canvas contents are still valid.
    """
    if draw:
        fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    return np.array(buf, dtype=np.uint8).copy()


def _render_overlay_on_image(
    img: np.ndarray,
    svg_path: Path,
    brain_bbox: tuple[int, int, int, int],
    lw_scale: float = 1.0,
    mirror: bool = False,
) -> np.ndarray:
    """Burn SVG paths into the image, scaled to match the brain bounding box.

    Uses the SVG ``viewBox`` as the authoritative reference frame so that
    the overlay aligns correctly even when internal sulcal lines don't extend
    to the brain's outer edges.  Falls back to the path bounding box when no
    ``viewBox`` is present.

    If *mirror* is True the paths are flipped horizontally (used when a
    left-hemisphere SVG is re-used for the right hemisphere).
    """
    paths, viewbox = _parse_svg_paths(svg_path)
    if not paths:
        return img

    # Use viewBox as the reference frame; fall back to path bounding box.
    if viewbox is not None:
        svg_x0, svg_y0, svg_w, svg_h = viewbox
    else:
        all_verts = np.vstack([p.vertices for p, _, _ in paths])
        svg_x0 = all_verts[:, 0].min()
        svg_y0 = all_verts[:, 1].min()
        svg_w = all_verts[:, 0].max() - svg_x0
        svg_h = all_verts[:, 1].max() - svg_y0
    if svg_w < 1e-6 or svg_h < 1e-6:
        return img

    y0, y1, x0, x1 = brain_bbox
    bh, bw = y1 - y0, x1 - x0

    sx = bw / svg_w
    sy = bh / svg_h
    mean_scale = (sx + sy) / 2

    # Render only the brain bounding-box region (much smaller than full image)
    dpi = 100
    fig_ov = plt.figure(figsize=(bw / dpi, bh / dpi), dpi=dpi)
    fig_ov.patch.set_alpha(0.0)
    ax = fig_ov.add_axes([0, 0, 1, 1], frameon=False)
    ax.patch.set_alpha(0.0)
    ax.set_xlim(0, bw)
    ax.set_ylim(bh, 0)
    ax.set_axis_off()

    for mpl_path, stroke, lw in paths:
        verts = mpl_path.vertices.copy()
        verts[:, 0] = (verts[:, 0] - svg_x0) * sx
        verts[:, 1] = (verts[:, 1] - svg_y0) * sy
        if mirror:
            verts[:, 0] = bw - verts[:, 0]
        scaled_path = MplPath(verts, mpl_path.codes)
        patch = PathPatch(
            scaled_path, fill=False, edgecolor=stroke,
            linewidth=max(0.3, lw * mean_scale * lw_scale),
        )
        ax.add_patch(patch)

    fig_ov.canvas.draw()
    ov = np.array(fig_ov.canvas.buffer_rgba(), dtype=np.uint8)
    plt.close(fig_ov)

    # Resize to exact bbox in case of rounding (bw×bh may differ by ±1 px)
    if ov.shape[0] != bh or ov.shape[1] != bw:
        from PIL import Image
        ov = np.array(Image.fromarray(ov).resize((bw, bh), Image.LANCZOS))

    # Alpha-composite into the brain region
    mask = ov[:, :, 3] > 0
    if mask.any():
        alpha = ov[mask, 3:4].astype(np.float32) / 255.0
        roi = img[y0:y1, x0:x1]
        roi[mask, :3] = (
            ov[mask, :3].astype(np.float32) * alpha
            + roi[mask, :3].astype(np.float32) * (1.0 - alpha)
        ).astype(np.uint8)
        roi[mask, 3] = np.maximum(roi[mask, 3], ov[mask, 3])

    return img


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------

def plot_brain(
    stat_map=None,
    *,
    bg_map: Union[str, float, np.ndarray, None] = "curvature",
    threshold: float | None = None,
    hemi: str = "left",
    view: str = "lateral",
    surf_type: str = "pial",
    res: Union[int, str] = "low",
    vol_to_surf_radius: float = 3.0,
    overlay: str | Path | None = "default",
    overlay_linewidth: float = 1.0,
    background: str = "white",
    post_blur: float = 3.0,
    axes: Axes | None = None,
    figure: Figure | None = None,
    title: str | None = None,
    colorbar: bool = True,
    output_file: str | Path | None = None,
    **kwargs,
) -> Figure:
    """Plot a statistical map on the MNI152 brain surface.

    Parameters
    ----------
    stat_map : array-like, path-like, or nibabel image, optional
        Per-vertex data, a path to a NIfTI volume, or a loaded ``nibabel``
        image.  Volumes are projected with ``nilearn.surface.vol_to_surf``.
        If *None*, only the background (curvature) is shown.
    bg_map : ``"curvature"`` | float | ndarray | None, default ``"curvature"``
        Background map.  ``"curvature"`` uses the curvature map.
        A float in ``[0, 1]`` produces a uniform grey background.
    threshold : float or None, default None
        Stat-map values whose absolute value is below *threshold* are
        transparent.
    hemi : ``"left"`` or ``"right"``, default ``"left"``
    view : ``"lateral"`` or ``"medial"``, default ``"lateral"``
    surf_type : ``"pial"`` or ``"inflated"``, default ``"pial"``
        Volume-to-surface projection is always done on the **pial** geometry.
    Advanced options:
    res : ``1`` | ``10`` | ``"high"`` | ``"low"``, default ``"low"``
        ``1`` / ``"high"`` = fine mesh; ``10`` / ``"low"`` = coarse (faster).
    vol_to_surf_radius : float, default 3.0
    overlay : ``"default"``, path, or None, default ``"default"``
        SVG contour overlay.  Looks for
        ``overlays/{surf_type}/{side}_{view}.svg`` first, then
        ``overlays/{side}_{view}.svg``.
    overlay_linewidth : float, default 1.0
        Multiplier for the SVG stroke widths.  Values > 1 make the
        contours thicker, < 1 thinner.
    background : ``"white"`` or ``"transparent"``, default ``"white"``
    post_blur : float, default 3.0
        Pixel-space Gaussian blur (sigma in px) applied to the brain
        bounding box *after* rendering but *before* the SVG overlay.
        Title and colorbar are never blurred.  Set to 0 to disable.
    axes : matplotlib ``Axes`` or None
        Existing 3-D axes to draw into.  The composited result replaces
        the axes content.
    figure : matplotlib ``Figure`` or None
    title : str or None
    colorbar : bool, default True
    output_file : path-like or None
    **kwargs
        Forwarded to ``nilearn.plotting.plot_surf_stat_map``.  Forbidden:
        ``surf_mesh``, ``engine``, ``title``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    bad = _NILEARN_FORBIDDEN_KWARGS & set(kwargs)
    if bad:
        raise TypeError(f"These kwargs are managed by quickbrain and must not be passed: {bad}")
    if hemi not in _ALLOWED_HEMIS:
        raise ValueError(f"hemi must be one of {_ALLOWED_HEMIS}, got {hemi!r}")
    if view not in _ALLOWED_VIEWS:
        raise ValueError(f"view must be one of {_ALLOWED_VIEWS}, got {view!r}")
    if surf_type not in _ALLOWED_SURF_TYPES:
        raise ValueError(f"surf_type must be one of {_ALLOWED_SURF_TYPES}, got {surf_type!r}")
    if background not in _ALLOWED_BACKGROUNDS:
        raise ValueError(f"background must be one of {_ALLOWED_BACKGROUNDS}, got {background!r}")

    res = normalize_resolution(res)

    # --- meshes ---------------------------------------------------------
    pial_coords, pial_faces = get_mesh(side=hemi, res=res, surf_type="pial")
    pial_mesh = (pial_coords, pial_faces)
    n_vertices = pial_coords.shape[0]
    display_mesh = (
        pial_mesh if surf_type == "pial"
        else get_mesh(side=hemi, res=res, surf_type=surf_type)
    )

    bg_arr = _resolve_bg_map(bg_map, side=hemi, res=res, n_vertices=n_vertices)

    # --- decide whether to enter the pixel pipeline ---------------------
    need_pixel_ops = (post_blur > 0) or (overlay is not None) or (background == "transparent")
    user_axes = axes  # remember the caller-supplied axes (may be None)

    # When pixel ops are needed we always render into a fresh temporary
    # figure so that blur / overlay / transparency can be applied cleanly.
    # The result is later pasted into *user_axes* if one was provided.
    if need_pixel_ops:
        render_axes = None
        render_figure = None
    else:
        render_axes = axes
        render_figure = figure

    nilearn_kwargs = dict(
        surf_mesh=display_mesh, hemi=hemi, view=view, engine="matplotlib",
        axes=render_axes, figure=render_figure,
    )

    if stat_map is not None:
        stat_arr = _resolve_stat_map(stat_map, pial_mesh, vol_to_surf_radius)
        fig = plot_surf_stat_map(
            stat_map=stat_arr, bg_map=bg_arr, threshold=threshold,
            title=None, colorbar=colorbar,
            **nilearn_kwargs, **kwargs,
        )
    else:
        if "cmap" not in kwargs:
            kwargs["cmap"] = "gray_r"
        fig = plot_surf(
            surf_map=bg_arr,
            title=None, colorbar=colorbar,
            **nilearn_kwargs, **kwargs,
        )

    if not need_pixel_ops:
        if title is not None and fig.axes:
            fig.axes[0].set_title(title, fontsize=12)
        if output_file is not None:
            fig.savefig(str(output_file), bbox_inches="tight")
        return fig

    # --- pixel-space pipeline -------------------------------------------
    # Find the 3-D brain axes and any siblings (colorbar) so we can
    # constrain the brain pixel search to the 3-D region only.
    brain_ax = None
    sibling_axes = []
    for a in fig.axes:
        if hasattr(a, "get_zlim"):
            brain_ax = a
        else:
            sibling_axes.append(a)

    fig.canvas.draw()
    rgba = _rasterize_figure(fig, draw=False)
    # Positions must be read before closing the figure.
    sibling_positions = [a.get_position() for a in sibling_axes]
    plt.close(fig)

    ih, iw = rgba.shape[:2]

    if brain_ax is not None:
        search_bbox = _brain_bbox_from_axes(brain_ax.get_position(), ih, iw)
        for sax in sibling_axes:
            spos = sax.get_position()
            sx0 = int(spos.x0 * iw)
            sx1 = int(spos.x1 * iw)
            sy0, sy1, sbx0, sbx1 = search_bbox
            if sx0 > sbx0 and sx0 < sbx1:
                sbx1 = sx0
            if sx1 > sbx0 and sx1 < sbx1 and sx0 < sbx0:
                sbx0 = sx1
            search_bbox = (sy0, sy1, sbx0, sbx1)
        brain_pixel_bbox = _find_brain_bbox_pixels(rgba, search_region=search_bbox)
    else:
        brain_pixel_bbox = _find_brain_bbox_pixels(rgba)

    # Build the brain mask from the rasterised image.  The background is
    # exactly RGB (255,255,255); anything that isn't *exactly* white within
    # the brain axes search region is brain.  This is robust even when the
    # colormap contains white because nilearn's 3-D lighting/shading always
    # shifts pure-white mesh faces by at least ±1 level.
    brain_mask = np.zeros((ih, iw), dtype=bool)
    if brain_ax is not None:
        by0, by1, bx0, bx1 = search_bbox
    else:
        by0, by1, bx0, bx1 = 0, ih, 0, iw
    region = rgba[by0:by1, bx0:bx1, :3]
    exact_white = np.all(region == 255, axis=2)
    brain_mask[by0:by1, bx0:bx1] = ~exact_white

    if post_blur > 0:
        rgba = _apply_post_blur(rgba, post_blur, brain_pixel_bbox, brain_mask)

    if background == "transparent":
        # Only make *white background* pixels transparent; preserve the
        # colorbar, title, and other non-brain, non-white content.
        is_white = rgba[:, :, :3].mean(axis=2) > 252
        rgba[is_white & ~brain_mask, 3] = 0

    svg_path, svg_mirror = _get_overlay_path(overlay, surf_type, hemi, view)
    if svg_path is not None:
        rgba = _render_overlay_on_image(
            rgba, svg_path, brain_pixel_bbox,
            lw_scale=overlay_linewidth, mirror=svg_mirror,
        )

    # Drop nilearn figure margins: keep only the tight brain region + colorbars.
    cy0, cy1, cx0, cx1 = _tight_figure_crop_bbox(
        brain_pixel_bbox, sibling_positions, ih, iw,
    )
    rgba = np.ascontiguousarray(rgba[cy0:cy1, cx0:cx1])
    ih, iw = rgba.shape[:2]

    # --- output ----------------------------------------------------------
    if user_axes is not None:
        # Paste the composited image into the caller's axes, converting
        # it from a 3-D projection axes to a flat image display.
        parent_fig = user_axes.figure
        pos = user_axes.get_position()
        user_axes.remove()
        ax_out = parent_fig.add_axes(pos, frameon=False)
        ax_out.imshow(rgba, aspect="equal")
        ax_out.set_axis_off()
        if title is not None:
            ax_out.set_title(title, fontsize=12, pad=3)
        if output_file is not None:
            transparent = background == "transparent"
            parent_fig.savefig(str(output_file), dpi=100, bbox_inches="tight",
                               transparent=transparent, pad_inches=0)
        return parent_fig

    dpi = 100 #todo
    img_w_in, img_h_in = iw / dpi, ih / dpi
    title_band_in = 0.18 if title is not None else 0.0
    out_fig = _make_figure(
        figsize=(img_w_in, img_h_in + title_band_in), dpi=dpi,
    )
    if title is not None:
        h_frac = img_h_in / (img_h_in + title_band_in)
        ax_out = out_fig.add_axes([0, 0, 1, h_frac], frameon=False)
    else:
        ax_out = out_fig.add_axes([0, 0, 1, 1], frameon=False)
    ax_out.imshow(rgba, aspect="equal")
    ax_out.set_axis_off()
    if title is not None:
        ax_out.set_title(title, fontsize=12, pad=2)

    if output_file is not None:
        transparent = background == "transparent"
        out_fig.savefig(str(output_file), dpi=dpi, bbox_inches="tight",
                        transparent=transparent, pad_inches=0)

    return out_fig
