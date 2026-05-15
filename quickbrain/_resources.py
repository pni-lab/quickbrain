"""Lazy-loading resource cache for pre-built meshes and curvature maps.

IMPORTANT: nilearn's matplotlib backend mutates mesh coordinates in-place
(``coords -= mean``).  The internal ``_load_*`` helpers are LRU-cached and
hold the master copy; the public ``get_*`` functions always return fresh
copies so the cache stays intact.
"""

from __future__ import annotations

import functools
from importlib.resources import as_file, files
from typing import Literal, Union

import numpy as np
from nilearn.surface import load_surf_mesh

_RESOURCE_PACKAGE = "quickbrain.resources"

# Voxel sizes used when building meshes: 1 = high detail, 10 = low (fast).
RES_HIGH = 1
RES_LOW = 10
_AVAILABLE_RES = (RES_HIGH, RES_LOW)

_RES_ALIASES = {
    "high": RES_HIGH,
    "low": RES_LOW,
    "1": RES_HIGH,
    "10": RES_LOW,
}

ExampleImage = Literal["pain_response", "left_hippocampus"]

_EXAMPLE_IMAGES = {
    "pain_response": "placebo_metaanalysis_full_pain_g_random.nii.gz",
    "left_hippocampus": "harvardoxford-subcortical_prob_Left Hippocampus.nii.gz",
}


def normalize_resolution(res: Union[int, str]) -> int:
    """Map ``res`` to voxel size ``1`` (high) or ``10`` (low).

    Accepts ``1``, ``10``, or the strings ``\"high\"`` / ``\"low\"``.
    """
    if isinstance(res, str):
        key = res.strip().lower()
        if key not in _RES_ALIASES:
            raise ValueError(
                f"res string must be 'high' or 'low' (or '1' / '10'), got {res!r}"
            )
        return _RES_ALIASES[key]
    if res not in _AVAILABLE_RES:
        raise ValueError(
            f"res must be {RES_HIGH} (high), {RES_LOW} (low), "
            f"or the strings 'high' / 'low'; got {res!r}"
        )
    return int(res)


_SIDES = ("left", "right")
_TYPES = ("pial", "inflated")


def _resource(name: str):
    return files(_RESOURCE_PACKAGE).joinpath(name)


def _resource_label(name: str) -> str:
    return f"{_RESOURCE_PACKAGE.replace('.', '/')}/{name}"


def load_example_image(name: ExampleImage = "pain_response"):
    """Load a bundled example NIfTI image.

    #### Parameters
    - name : ``"pain_response"`` or ``"left_hippocampus"``
        Example image to load.

    #### Returns
    - nibabel.spatialimages.SpatialImage
        Loaded NIfTI image.

    #### Example
    ```python
    from quickbrain import load_example_image, plot_brain
    image = load_example_image("pain_response")
    plot_brain(image, hemi="left", view="lateral", threshold=0.01)
    ```
    """
    try:
        filename = _EXAMPLE_IMAGES[name]
    except KeyError as exc:
        valid = ", ".join(repr(key) for key in _EXAMPLE_IMAGES)
        raise ValueError(f"name must be one of {valid}; got {name!r}") from exc

    resource = _resource(filename)
    with as_file(resource) as path:
        if not path.is_file():
            raise FileNotFoundError(
                f"Packaged example image not found: {_resource_label(filename)}. "
                "Install a wheel/source tree that includes quickbrain package data."
            )

        import nibabel as nib

        return nib.load(path)


def _check_params(side: str, res: int, surf_type: str) -> None:
    if side not in _SIDES:
        raise ValueError(f"side must be one of {_SIDES}, got {side!r}")
    if res not in _AVAILABLE_RES:
        raise ValueError(f"res must be one of {_AVAILABLE_RES}, got {res!r}")
    if surf_type not in _TYPES:
        raise ValueError(f"surf_type must be one of {_TYPES}, got {surf_type!r}")


@functools.lru_cache(maxsize=32)
def _load_mesh(side: str, res: int, surf_type: str) -> tuple[np.ndarray, np.ndarray]:
    _check_params(side, res, surf_type)
    name = f"mni152_side-{side}_res-{res}_type-{surf_type}.gii.gz"
    resource = _resource(name)
    with as_file(resource) as path:
        if not path.is_file():
            raise FileNotFoundError(
                f"Packaged mesh not found: {_resource_label(name)}. "
                "Install a wheel/source tree that includes quickbrain package data, "
                "or rebuild the package resources before publishing."
            )
        coords, faces = load_surf_mesh(str(path))
    return np.asarray(coords, dtype=np.float32), np.asarray(faces, dtype=np.int32)


def get_mesh(
    side: str = "left",
    res: Union[int, str] = RES_LOW,
    surf_type: str = "pial",
) -> tuple[np.ndarray, np.ndarray]:
    """Return (coords, faces) for a pre-built hemisphere mesh.

    Every call returns **fresh copies** (safe from in-place mutation).

    Parameters
    ----------
    side : ``"left"`` or ``"right"``
    res : ``1`` | ``10`` | ``"high"`` | ``"low"``
        ``1`` / ``"high"`` = fine mesh; ``10`` / ``"low"`` = coarse (default).
    surf_type : ``"pial"`` or ``"inflated"``
    """
    res_i = normalize_resolution(res)
    coords, faces = _load_mesh(side, res_i, surf_type)
    return coords.copy(), faces.copy()


@functools.lru_cache(maxsize=32)
def _load_curvature(side: str, res: int) -> np.ndarray:
    if side not in _SIDES:
        raise ValueError(f"side must be one of {_SIDES}, got {side!r}")
    if res not in _AVAILABLE_RES:
        raise ValueError(f"res must be one of {_AVAILABLE_RES}, got {res!r}")
    name = f"curvature_side-{side}_res-{res}.txt"
    resource = _resource(name)
    with as_file(resource) as path:
        if not path.is_file():
            raise FileNotFoundError(
                f"Packaged curvature map not found: {_resource_label(name)}. "
                "Install a wheel/source tree that includes quickbrain package data, "
                "or rebuild the package resources before publishing."
            )
        return np.loadtxt(path, dtype=np.float32)


def get_curvature(side: str = "left", res: Union[int, str] = RES_LOW) -> np.ndarray:
    """Return per-vertex signed curvature array for a hemisphere/resolution.

    Every call returns a **fresh copy**.
    """
    res_i = normalize_resolution(res)
    return _load_curvature(side, res_i).copy()
