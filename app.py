from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from physics import (
    constant_material,
    load_material_table,
    rayleigh_anomaly_nm,
    solve_normal_incidence_array,
)

st.set_page_config(page_title="Infinite Dipole Array Solver", page_icon="◌", layout="wide")
st.title("Infinite Dipole Array Solver")
st.caption("Normal-incidence coupled electric/magnetic dipole model in a homogeneous medium")

with st.expander("Model scope and numerical caution", expanded=True):
    st.markdown(
        r"""
        This app implements the **same physical scope as the supplied RTA script**: a spherical particle,
        identical homogeneous superstrate/substrate, normal incidence, and electric/magnetic dipole Mie terms.
        It does **not** yet provide a validated substrate-supported R/T model or an exact Ewald/Kambe implementation.

        The direct lattice sum is truncated and exponentially regularized by \(k \rightarrow k(1+i\gamma)\).
        Test convergence by increasing the truncation order and decreasing \(\gamma\); do not interpret a sharp
        feature near a Rayleigh anomaly until it is stable under both tests.
        """
    )

with st.sidebar:
    st.header("Geometry and medium")
    radius_nm = st.number_input("Sphere radius (nm)", min_value=1.0, value=80.0, step=1.0)
    Px_nm = st.number_input("Period Px (nm)", min_value=10.0, value=600.0, step=5.0)
    Py_nm = st.number_input("Period Py (nm)", min_value=10.0, value=600.0, step=5.0)
    n_medium = st.number_input("Homogeneous medium index", min_value=1.0, value=1.0, step=0.01, format="%.3f")

    st.header("Spectral grid")
    lam_start = st.number_input("Start wavelength (nm)", min_value=100.0, value=500.0, step=5.0)
    lam_stop = st.number_input("Stop wavelength (nm)", min_value=101.0, value=1000.0, step=5.0)
    n_points = st.slider("Number of wavelength points", min_value=51, max_value=1001, value=251, step=10)

    st.header("Lattice sum")
    truncation_order = st.slider("Direct-sum truncation order N", min_value=5, max_value=100, value=30, step=5)
    damping = st.number_input("Convergence damping γ", min_value=0.0, max_value=0.10, value=0.01, step=0.001, format="%.3f")

    st.header("Particle optical constants")
    material_mode = st.radio("Source", ["Upload n,k table", "Constant-index demonstration"])
    material = None
    if material_mode == "Upload n,k table":
        upload = st.file_uploader("Optical constants file", type=["txt", "csv", "dat"])
        material_unit = st.selectbox("Wavelength unit in file", ["nm", "um"], index=0)
        st.caption("Expected numeric columns: wavelength, n, k. Header/comment lines are allowed.")
        if upload is not None:
            try:
                material = load_material_table(upload.getvalue(), material_unit, label=upload.name)
                st.success(f"Loaded {upload.name}: {len(material.wavelength_m)} rows")
            except ValueError as exc:
                st.error(str(exc))
    else:
        n_real = st.number_input("Particle n", min_value=0.01, value=1.50, step=0.01, format="%.3f")
        k_imag = st.number_input("Particle k", min_value=0.0, value=0.0, step=0.01, format="%.3f")
        if lam_stop > lam_start:
            material = constant_material(n_real, k_imag, lam_start * 1e-9, lam_stop * 1e-9)

run = st.button("Run calculation", type="primary", use_container_width=True)

if run:
    if lam_stop <= lam_start:
        st.error("Stop wavelength must be larger than start wavelength.")
        st.stop()
    if material is None:
        st.warning("Upload the particle optical-constant table, or choose the constant-index demonstration mode.")
        st.stop()

    wavelength_nm = np.linspace(lam_start, lam_stop, n_points)
    try:
        with st.spinner("Calculating Mie polarizabilities and lattice interactions..."):
            result = solve_normal_incidence_array(
                wavelength_nm=wavelength_nm,
                radius_nm=radius_nm,
                Px_nm=Px_nm,
                Py_nm=Py_nm,
                n_medium=n_medium,
                material=material,
                truncation_order=truncation_order,
                damping=damping,
            )
    except (ValueError, FloatingPointError) as exc:
        st.error(str(exc))
        st.stop()

    ra = rayleigh_anomaly_nm(Px_nm * 1e-9, Py_nm * 1e-9, n_medium, [(1, 0), (0, 1), (1, 1)])
    st.session_state["result"] = result
    st.session_state["ra"] = ra

if "result" in st.session_state:
    result: pd.DataFrame = st.session_state["result"]
    ra: dict[str, float] = st.session_state["ra"]

    top1, top2, top3 = st.columns(3)
    max_row = result.loc[result["R"].idxmax()]
    top1.metric("Maximum R", f"{max_row['R']:.4f}", f"at {max_row['wavelength_nm']:.1f} nm")
    top2.metric("Maximum |A|", f"{np.max(np.abs(result['A'])):.4f}", "Energy-balance diagnostic")
    top3.metric("Lattice terms", f"{(2 * truncation_order + 1)**2 - 1:,}", f"N = {truncation_order}")

    fig_rta = go.Figure()
    for column, name in [("R", "Reflectance"), ("T", "Transmittance"), ("A", "A = 1 − R − T (diagnostic)")]:
        fig_rta.add_trace(go.Scatter(x=result["wavelength_nm"], y=result[column], mode="lines", name=name))
    for label, wavelength in ra.items():
        if result["wavelength_nm"].min() <= wavelength <= result["wavelength_nm"].max():
            fig_rta.add_vline(x=wavelength, line_dash="dot", annotation_text=f"RA {label}", annotation_position="top")
    fig_rta.update_layout(
        title="R / T / A",
        xaxis_title="Wavelength (nm)",
        yaxis_title="Coefficient",
        hovermode="x unified",
        legend_title_text="",
    )
    st.plotly_chart(fig_rta, use_container_width=True)

    col_left, col_right = st.columns(2)
    with col_left:
        fig_s = go.Figure()
        fig_s.add_trace(go.Scatter(x=result["wavelength_nm"], y=result["Sxx_real_m-3"], mode="lines", name="Re(Sxx)"))
        fig_s.add_trace(go.Scatter(x=result["wavelength_nm"], y=result["inv_alpha_e_real_m-3"], mode="lines", name="Re(1/αe)"))
        fig_s.update_layout(title="Electric resonance condition", xaxis_title="Wavelength (nm)", yaxis_title="m⁻³", hovermode="x unified")
        st.plotly_chart(fig_s, use_container_width=True)
    with col_right:
        fig_i = go.Figure()
        fig_i.add_trace(go.Scatter(x=result["wavelength_nm"], y=result["Sxx_imag_m-3"], mode="lines", name="Im(Sxx)"))
        fig_i.add_trace(go.Scatter(x=result["wavelength_nm"], y=result["inv_alpha_e_imag_m-3"], mode="lines", name="Im(1/αe)"))
        fig_i.update_layout(title="Electric linewidth condition", xaxis_title="Wavelength (nm)", yaxis_title="m⁻³", hovermode="x unified")
        st.plotly_chart(fig_i, use_container_width=True)

    with st.expander("Download and inspect numerical output", expanded=False):
        st.download_button(
            "Download CSV",
            data=result.to_csv(index=False).encode("utf-8"),
            file_name="infinite_dipole_array_result.csv",
            mime="text/csv",
        )
        st.dataframe(result, use_container_width=True, height=360)
