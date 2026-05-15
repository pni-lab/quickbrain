---
title: "quickbrain"
subtitle: "Fast, good-looking brain plots for Python"
short_title: "quickbrain"
---

:::{figure} #nice-brains
:::

The `quickbrain` Python package provides a compact Python API for creating beautiful brain images quickly and easily.

### When to use quickbrain

- useful both for quick & dirty analyses and for publication-quality images 
- to provide a comprehensive, interpretable overview of whole-brain activation patterns
- including the cerebellum (often omitted in other visualizations)

### When to not use quickbrain

- If you want high-resolution surface projections, and don't care about the outline contour and the cerebellum, use the `nilearn.surface.plot_surf_stat_map` function.
- If you want precise localization, don't use surface projection-based visualizations; use volumetric visualizations instead, e.g. `nilearn.plotting.plot_stat_map`.

::::{grid} 1 1 2 2

:::{card}
:header: **PRO:** 
- simple
- fast, viable for large composite images, e.g. for quick overview of whole-brain activation patterns
- includes the cerebellum
- overlays a contour to aid in interpretation
- inflated meshes for better visualization of the insula and other deep sulcal structures
- looks good!
:::

:::{card}
:header: **CONTRA:**
- simple
- as all visualizations that are based on meshes, it is suboptimal if you want fine-grained localization
- only lateral and medial views
- no nearest neighbor interpolation
:::

::::


## Install

```bash
pip install "quickbrain @ git+https://github.com/pni-lab/quickbrain.git"
```

:::{note}
PyPI package coming soon!
:::

## Dependencies

`quickbrain` keeps its runtime dependencies intentionally small:

- `matplotlib`
- `nibabel`
- `nilearn`
- `numpy`
- `scipy`
- `svgpath2mpl`

The documentation site is built with MyST Markdown in GitHub Actions; MyST is
installed there with `npm` and is not a Python package dependency of
`quickbrain`.

## Example

```python
import nibabel as nib
from quickbrain import plot_brain
image = nib.load("path/to/your/image.nii.gz")
plot_brain(image)
```

:::{figure} #ex
:::

## More examples

Open [](getting-started) for a notebook walkthrough, then see
[](documentation) for the API generated from package docstrings.

:::{warning}
The `quickbrain` package is still in development and the API is subject to change.
Bug reports and feature requests: https://github.com/pni-lab/quickbrain/issues
:::

## Under the hood

The `quickbrain` package is built on top of the `nilearn` package, which provides a comprehensive set of tools for neuroimaging data analysis.
The trick is that we don't need a high-resolution surface if we only want an overview of the whole-brain pattern, instead of detailed localization. Thus, the `quickbrain` package uses a custom-built low-resolution mesh (including the cerebellum!) and the `nilearn.surface.vol_to_surf` and the `nilearn.surface.plot_surf_stat_map` functions to project the statistical map onto the brain surface and plot the brain surface. This makes it lightning fast. But, low resolution means less details. So we put back the details by adding beautiful contour overlays, to guide the eye across sulci and gyri.

## Developers, contributors and contact

```{card} vibe-coded, cleaned, maintains
:header: **Tamas Spisak**
- Center for Translational Neuro- and Behavioral Sciences (C-TNBS), University Medicine Essen, Germany
- orcid: 0000-0002-2942-0821
- email: tamas.spisak@uk-essen.de
```
