# Infinite Dipole Array Solver

A Streamlit interface for the normal-incidence coupled-dipole model in the accompanying research code.

## What it calculates

- Electric and magnetic dipole Mie polarizabilities of a sphere.
- A homogeneous-medium finite real-space dyadic lattice sum.
- Effective polarizabilities \(\alpha_{e,m}^{\rm eff}=[\alpha_{e,m}^{-1}-S_{xx,yy}]^{-1}\).
- Reflectance, transmittance, and absorbance for a square or rectangular array at normal incidence.
- Rayleigh-anomaly guide lines and inverse-polarizability diagnostics.

## Model boundaries

This version intentionally **does not** claim to provide an exact Ewald/Kambe result. It uses a direct real-space sum, regularized by

\[
k \rightarrow k(1+i\gamma).
\]

The convergence damping and truncation order must be tested. Because the exponential damping is numerical regularization, \(A=1-R-T\) can include an artificial loss term even for a lossless particle. Features extremely near the Rayleigh anomaly should only be reported after they remain stable under a convergence sweep.

The original repository contains substrate functions, but the current reflectance/transmittance equation is a **homogeneous-medium** expression. A substrate selector would therefore be physically misleading without a consistent asymmetric Green tensor and far-field scattering model. It is not exposed in this app.

## Run locally

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

## Optical-constant input

Upload a text/CSV/DAT file with numeric columns:

```text
wavelength   n   k
500           0.97 1.87
510           0.91 1.95
...
```

Choose the correct wavelength unit in the sidebar. The input range must cover the full simulation range; the app intentionally refuses extrapolation.

## Recommended GitHub layout

```text
app.py
physics.py
requirements.txt
README.md
material_data/          # only materials you have permission to redistribute
```

Do not commit unpublished numerical sweeps, confidential lab data, or proprietary optical-constant files without permission.
