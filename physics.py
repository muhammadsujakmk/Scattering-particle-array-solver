"""Core solver for a normal-incidence, homogeneous-medium dipole array.

The polarizabilities use the electric and magnetic dipole Mie coefficients of a
sphere. The lattice interaction is evaluated by a finite, exponentially
regularized direct real-space dyadic sum. It reproduces the structure of the
original research script but is vectorized for interactive use.

Units: SI internally. Wavelength and geometry inputs to public functions are metres.
"""
from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.special import spherical_jn, spherical_yn


@dataclass(frozen=True)
class MaterialData:
    wavelength_m: np.ndarray
    n_complex: np.ndarray
    label: str = "uploaded material"

    def evaluate(self, wavelength_m: np.ndarray) -> np.ndarray:
        wavelength_m = np.asarray(wavelength_m, dtype=float)
        lo, hi = self.wavelength_m.min(), self.wavelength_m.max()
        tolerance = max(1e-15, 1e-10 * (hi - lo))
        if wavelength_m.min() < lo - tolerance or wavelength_m.max() > hi + tolerance:
            raise ValueError(
                "The requested wavelength range extends outside the uploaded optical-constant data "
                f"({lo * 1e9:.1f}–{hi * 1e9:.1f} nm)."
            )
        # Tiny endpoint roundoff is harmless; real extrapolation is intentionally refused.
        wavelength_m = np.clip(wavelength_m, lo, hi)
        re_interp = PchipInterpolator(self.wavelength_m, self.n_complex.real)
        im_interp = PchipInterpolator(self.wavelength_m, self.n_complex.imag)
        return re_interp(wavelength_m) + 1j * im_interp(wavelength_m)


def load_material_table(raw: bytes, wavelength_unit: str = "nm", label: str = "uploaded material") -> MaterialData:
    """Read a 3-column table: wavelength, n, k. Header lines are permitted."""
    if wavelength_unit not in {"nm", "um"}:
        raise ValueError("wavelength_unit must be 'nm' or 'um'.")

    try:
        frame = pd.read_csv(BytesIO(raw), sep=None, engine="python", comment="#", header=None)
    except Exception as exc:
        raise ValueError("Could not read the material file. Use a text/CSV file with wavelength, n, k columns.") from exc

    numeric = frame.apply(pd.to_numeric, errors="coerce").dropna(how="all")
    if numeric.shape[1] < 3:
        raise ValueError("The material file must contain at least three numeric columns: wavelength, n, k.")
    numeric = numeric.iloc[:, :3].dropna()
    if len(numeric) < 4:
        raise ValueError("At least four numeric material-data rows are required for interpolation.")

    factor = 1e-9 if wavelength_unit == "nm" else 1e-6
    wavelength_m = numeric.iloc[:, 0].to_numpy(dtype=float) * factor
    n_complex = numeric.iloc[:, 1].to_numpy(dtype=float) + 1j * numeric.iloc[:, 2].to_numpy(dtype=float)

    order = np.argsort(wavelength_m)
    wavelength_m = wavelength_m[order]
    n_complex = n_complex[order]
    unique = np.concatenate(([True], np.diff(wavelength_m) > 0))
    wavelength_m, n_complex = wavelength_m[unique], n_complex[unique]

    if np.any(wavelength_m <= 0):
        raise ValueError("Wavelength values must be positive.")
    return MaterialData(wavelength_m=wavelength_m, n_complex=n_complex, label=label)


def constant_material(n_real: float, k_imag: float, wavelength_min_m: float, wavelength_max_m: float) -> MaterialData:
    if n_real <= 0 or k_imag < 0:
        raise ValueError("The real refractive index must be positive and k must be non-negative.")
    grid = np.array([wavelength_min_m, wavelength_min_m * 1.001, wavelength_max_m * 0.999, wavelength_max_m])
    values = np.full(grid.shape, n_real + 1j * k_imag, dtype=complex)
    return MaterialData(grid, values, label="constant-index demonstration")


def mie_dipole_polarizabilities(
    wavelength_m: np.ndarray,
    radius_m: float,
    n_medium: float,
    n_particle: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return alpha_e and alpha_m in m^3 using the exp(+ikr) convention.

    This is the vectorized n=1 version of the user's alpha_Mie/recmethod chain.
    """
    wavelength_m = np.asarray(wavelength_m, dtype=float)
    n_particle = np.asarray(n_particle, dtype=complex)
    if radius_m <= 0 or n_medium <= 0:
        raise ValueError("Radius and medium refractive index must be positive.")

    k = 2 * np.pi * n_medium / wavelength_m
    x = k * radius_m
    m = n_particle / n_medium
    mx = m * x

    # Riccati-Bessel functions and their derivatives for l = 1.
    psi_x = x * spherical_jn(1, x)
    psi_x_prime = spherical_jn(1, x) + x * spherical_jn(1, x, derivative=True)
    xi_x = x * (spherical_jn(1, x) + 1j * spherical_yn(1, x))
    xi_x_prime = (
        spherical_jn(1, x) + 1j * spherical_yn(1, x)
        + x * (spherical_jn(1, x, derivative=True) + 1j * spherical_yn(1, x, derivative=True))
    )

    psi_mx = mx * spherical_jn(1, mx)
    psi_mx_prime = spherical_jn(1, mx) + mx * spherical_jn(1, mx, derivative=True)

    a1_num = m * psi_mx * psi_x_prime - psi_x * psi_mx_prime
    a1_den = m * psi_mx * xi_x_prime - xi_x * psi_mx_prime
    b1_num = psi_mx * psi_x_prime - m * psi_x * psi_mx_prime
    b1_den = psi_mx * xi_x_prime - m * xi_x * psi_mx_prime
    a1 = a1_num / a1_den
    b1 = b1_num / b1_den

    alpha_e = 1j * 6 * np.pi * a1 / k**3
    alpha_m = 1j * 6 * np.pi * b1 / k**3
    return alpha_e, alpha_m


@dataclass(frozen=True)
class LatticeGeometry:
    Px_m: float
    Py_m: float
    order: int
    x_m: np.ndarray
    y_m: np.ndarray
    r_m: np.ndarray
    gxx: np.ndarray
    gyy: np.ndarray


def prepare_lattice(Px_m: float, Py_m: float, order: int) -> LatticeGeometry:
    if Px_m <= 0 or Py_m <= 0:
        raise ValueError("Both lattice periods must be positive.")
    if not isinstance(order, (int, np.integer)) or order < 1:
        raise ValueError("The truncation order must be an integer of at least 1.")

    nx, ny = np.meshgrid(np.arange(-order, order + 1), np.arange(-order, order + 1), indexing="ij")
    mask = ~((nx == 0) & (ny == 0))
    x = (nx[mask] * Px_m).astype(float)
    y = (ny[mask] * Py_m).astype(float)
    r = np.hypot(x, y)
    return LatticeGeometry(Px_m, Py_m, int(order), x, y, r, x**2 / r**2, y**2 / r**2)


def lattice_sum_full_direct(k_m: complex, geometry: LatticeGeometry, damping: float = 1e-2) -> tuple[complex, complex]:
    """Direct finite real-space dyadic sum, regularized by k -> k(1+i*damping).

    A finite direct sum is a numerical approximation—not an exact Ewald/Kambe
    sum. The damping makes the oscillatory far-field series converge but also
    modifies the calculated spectrum, especially extremely near a Rayleigh anomaly.
    """
    if damping < 0:
        raise ValueError("Damping must be non-negative.")
    kd = complex(k_m) * (1 + 1j * damping)
    r = geometry.r_m
    kr = kd * r
    prefactor = kd**2 / (4 * np.pi) * np.exp(1j * kr) / r
    A = 1 + (1j * kr - 1) / (kd**2 * r**2)
    B = (3 - 3j * kr - (kd * r) ** 2) / (kd**2 * r**2)
    sxx = np.sum(prefactor * (A + B * geometry.gxx))
    syy = np.sum(prefactor * (A + B * geometry.gyy))
    return complex(sxx), complex(syy)


def rayleigh_anomaly_nm(Px_m: float, Py_m: float, n_medium: float, order_pairs: Iterable[tuple[int, int]]) -> dict[str, float]:
    out: dict[str, float] = {}
    for m, n in order_pairs:
        denominator = np.sqrt((m / Px_m) ** 2 + (n / Py_m) ** 2)
        if denominator == 0:
            continue
        lam_m = n_medium / denominator
        out[f"({m:+d},{n:+d})"] = lam_m * 1e9
    return out


def solve_normal_incidence_array(
    wavelength_nm: np.ndarray,
    radius_nm: float,
    Px_nm: float,
    Py_nm: float,
    n_medium: float,
    material: MaterialData,
    truncation_order: int = 30,
    damping: float = 1e-2,
) -> pd.DataFrame:
    """Calculate R/T/A and diagnostic inverse-polarizability/lattice-sum terms."""
    wavelength_nm = np.asarray(wavelength_nm, dtype=float)
    if wavelength_nm.ndim != 1 or len(wavelength_nm) < 2:
        raise ValueError("Wavelength needs at least two points.")
    if np.any(np.diff(wavelength_nm) <= 0):
        raise ValueError("Wavelength must be strictly increasing.")
    if n_medium <= 0:
        raise ValueError("Medium refractive index must be positive.")

    lam_m = wavelength_nm * 1e-9
    n_particle = material.evaluate(lam_m)
    radius_m, Px_m, Py_m = radius_nm * 1e-9, Px_nm * 1e-9, Py_nm * 1e-9
    alpha_e, alpha_m = mie_dipole_polarizabilities(lam_m, radius_m, n_medium, n_particle)
    geometry = prepare_lattice(Px_m, Py_m, truncation_order)
    area = Px_m * Py_m

    sxx = np.empty_like(lam_m, dtype=complex)
    syy = np.empty_like(lam_m, dtype=complex)
    reflectance = np.empty_like(lam_m)
    transmittance = np.empty_like(lam_m)
    absorbance = np.empty_like(lam_m)
    alpha_e_eff = np.empty_like(lam_m, dtype=complex)
    alpha_m_eff = np.empty_like(lam_m, dtype=complex)

    for i, wavelength in enumerate(lam_m):
        k = 2 * np.pi * n_medium / wavelength
        sxx[i], syy[i] = lattice_sum_full_direct(k, geometry, damping=damping)
        alpha_e_eff[i] = 1 / (1 / alpha_e[i] - sxx[i])
        alpha_m_eff[i] = 1 / (1 / alpha_m[i] - syy[i])
        factor = 1j * k / (2 * area)
        r_amp = factor * (alpha_e_eff[i] - alpha_m_eff[i])
        t_amp = 1 + factor * (alpha_e_eff[i] + alpha_m_eff[i])
        reflectance[i] = float(abs(r_amp) ** 2)
        transmittance[i] = float(abs(t_amp) ** 2)
        absorbance[i] = 1.0 - transmittance[i] - reflectance[i]

    return pd.DataFrame(
        {
            "wavelength_nm": wavelength_nm,
            "R": reflectance,
            "T": transmittance,
            "A": absorbance,
            "Sxx_real_m-3": sxx.real,
            "Sxx_imag_m-3": sxx.imag,
            "Syy_real_m-3": syy.real,
            "Syy_imag_m-3": syy.imag,
            "inv_alpha_e_real_m-3": (1 / alpha_e).real,
            "inv_alpha_e_imag_m-3": (1 / alpha_e).imag,
            "inv_alpha_m_real_m-3": (1 / alpha_m).real,
            "inv_alpha_m_imag_m-3": (1 / alpha_m).imag,
            "alpha_e_eff_real_m3": alpha_e_eff.real,
            "alpha_e_eff_imag_m3": alpha_e_eff.imag,
            "alpha_m_eff_real_m3": alpha_m_eff.real,
            "alpha_m_eff_imag_m3": alpha_m_eff.imag,
            "n_particle_real": n_particle.real,
            "n_particle_imag": n_particle.imag,
        }
    )
