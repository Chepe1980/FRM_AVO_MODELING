import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.colors as colors
from pyavo.seismodel import tuning_prestack as tp
from pyavo.seismodel import wavelet

# Set page config
st.set_page_config(layout="wide", page_title="Seismic Fluid Replacement Modeling")

# Custom CSS
st.markdown("""
<style>
    .stPlot, .stDataFrame {
        border-radius: 10px;
        box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
    }
    .stSlider { padding: 0 20px; }
    .sidebar .sidebar-content { background-color: #f8f9fa; }
</style>
""", unsafe_allow_html=True)

# Title
st.title("Seismic Fluid Replacement Modeling & AVO Analysis")
st.markdown("Interactive app for FRM and AVO analysis of well log data")

# Helper functions
def vrh(volumes, k, mu):
    f = np.array(volumes).T
    k = np.resize(np.array(k), np.shape(f))
    mu = np.resize(np.array(mu), np.shape(f))
    k_u = np.sum(f*k, axis=1)
    k_l = 1./np.sum(f/k, axis=1)
    mu_u = np.sum(f*mu, axis=1)
    mu_l = 1./np.sum(f/mu, axis=1)
    return k_u, k_l, mu_u, mu_l, (k_u+k_l)/2, (mu_u+mu_l)/2

def frm(vp1, vs1, rho1, rho_f1, k_f1, rho_f2, k_f2, k0, phi):
    vp1, vs1 = vp1/1000., vs1/1000.
    mu1 = rho1 * vs1**2
    k_s1 = rho1 * vp1**2 - (4/3)*mu1
    kdry = (k_s1*((phi*k0)/k_f1+1-phi)-k0)/((phi*k0)/k_f1+(k_s1/k0)-1-phi)
    k_s2 = kdry + (1-(kdry/k0))**2/((phi/k_f2)+((1-phi)/k0)-(kdry/k0**2))
    rho2 = rho1 - phi*rho_f1 + phi*rho_f2
    vp2 = np.sqrt((k_s2+(4/3)*mu1)/rho2)*1000
    vs2 = np.sqrt(mu1/rho2)*1000
    return vp2, vs2, rho2, k_s2

# Sidebar inputs
with st.sidebar:
    st.header("Input Parameters")
    uploaded_file = st.file_uploader("Upload Well Log CSV", type=['csv'])
    
    if uploaded_file:
        logs = pd.read_csv(uploaded_file)
        st.subheader("Column Selection")
        cols = logs.columns
        depth_col = st.selectbox("Depth Column", cols)
        vp_col = st.selectbox("Vp Column", cols)
        vs_col = st.selectbox("Vs Column", cols)
        rho_col = st.selectbox("Density Column", cols)
        vsh_col = st.selectbox("Vshale Column", cols)
        sw_col = st.selectbox("Sw Column", cols)
        phi_col = st.selectbox("Porosity Column", cols)

        st.subheader("Mineral Properties")
        rho_qz = st.number_input("Quartz Density (g/cc)", 2.65)
        k_qz = st.number_input("Quartz Bulk Modulus (GPa)", 37.0)
        mu_qz = st.number_input("Quartz Shear Modulus (GPa)", 44.0)
        rho_sh = st.number_input("Shale Density (g/cc)", 2.81)
        k_sh = st.number_input("Shale Bulk Modulus (GPa)", 15.0)
        mu_sh = st.number_input("Shale Shear Modulus (GPa)", 5.0)

        st.subheader("Fluid Properties")
        rho_b = st.number_input("Brine Density (g/cc)", 1.09)
        k_b = st.number_input("Brine Bulk Modulus (GPa)", 2.8)
        rho_o = st.number_input("Oil Density (g/cc)", 0.78)
        k_o = st.number_input("Oil Bulk Modulus (GPa)", 0.94)
        rho_g = st.number_input("Gas Density (g/cc)", 0.25)
        k_g = st.number_input("Gas Bulk Modulus (GPa)", 0.06)

        st.subheader("Modeling Parameters")
        ztop = st.number_input("Top Depth (m)", float(logs[depth_col].min()))
        zbot = st.number_input("Bottom Depth (m)", float(logs[depth_col].max()))
        sand_cutoff = st.slider("Sand Cutoff (Vshale)", 0.0, 1.0, 0.12)
        freq = st.slider("Wavelet Frequency (Hz)", 10, 100, 30)
        wlt_length = st.slider("Wavelet Length (ms)", 50, 300, 128)
        sample_rate = st.slider("Sample Rate (ms)", 0.01, 0.5, 0.1)
        max_angle = st.slider("Maximum Angle (degrees)", 10, 60, 45)
        excursion = st.slider("Trace Excursion", 1, 5, 2)
        thickness = st.slider("Layer Thickness (ms)", 10, 100, 37)

# Main processing
if uploaded_file:
    try:
        # Perform FRM calculations
        shale = logs[vsh_col].values
        sand = 1 - shale - logs[phi_col].values
        shaleN, sandN = shale/(shale+sand), sand/(shale+sand)
        k_u, k_l, mu_u, mu_l, k0, mu0 = vrh([shaleN, sandN], [k_sh, k_qz], [mu_sh, mu_qz])
        
        water, hc = logs[sw_col].values, 1-logs[sw_col].values
        _, k_fl, *_ = vrh([water, hc], [k_b, k_o], [0, 0])
        rho_fl = water*rho_b + hc*rho_o

        # Apply FRM
        vpb, vsb, rhob, _ = frm(logs[vp_col], logs[vs_col], logs[rho_col], rho_fl, k_fl, rho_b, k_b, k0, logs[phi_col])
        vpo, vso, rhoo, _ = frm(logs[vp_col], logs[vs_col], logs[rho_col], rho_fl, k_fl, rho_o, k_o, k0, logs[phi_col])
        vpg, vsg, rhog, _ = frm(logs[vp_col], logs[vs_col], logs[rho_col], rho_fl, k_fl, rho_g, k_g, k0, logs[phi_col])

        # Lithology classification
        brine_sand = (logs[vsh_col] <= sand_cutoff) & (logs[sw_col] >= 0.65)
        oil_sand = (logs[vsh_col] <= sand_cutoff) & (logs[sw_col] < 0.65)
        shale_flag = logs[vsh_col] > sand_cutoff

        # Add results to dataframe
        for fluid, vp, vs, rho in [('B', vpb, vsb, rhob), ('O', vpo, vso, rhoo), ('G', vpg, vsg, rhog)]:
            logs[f'VP_FRM{fluid}'] = np.where(brine_sand | oil_sand, vp, logs[vp_col])
            logs[f'VS_FRM{fluid}'] = np.where(brine_sand | oil_sand, vs, logs[vs_col])
            logs[f'RHO_FRM{fluid}'] = np.where(brine_sand | oil_sand, rho, logs[rho_col])
            logs[f'IP_FRM{fluid}'] = logs[f'VP_FRM{fluid}'] * logs[f'RHO_FRM{fluid}']
            logs[f'VPVS_FRM{fluid}'] = logs[f'VP_FRM{fluid}'] / logs[f'VS_FRM{fluid}']
            logs[f'LFC_{fluid}'] = np.where(shale_flag, 4, np.where(brine_sand | oil_sand, {'B':1, 'O':2, 'G':3}[fluid], 0))

        # Create tabs
        tab1, tab2, tab3, tab4 = st.tabs(["Well Logs", "Crossplots", "Brine Case", "Oil & Gas Cases"])

        with tab1:
            st.header("Well Log Visualization")
            ll = logs[(logs[depth_col] >= ztop) & (logs[depth_col] <= zbot)]
            
            # Create facies display
            cluster_data = np.expand_dims(ll['LFC_B'].values, axis=1)
            cluster = np.repeat(cluster_data, 100, axis=1)
            cmap = colors.ListedColormap(['#B3B3B3', 'blue', 'green', 'red', '#996633'])

            fig, ax = plt.subplots(1, 4, figsize=(12, 8))
            ax[0].plot(ll[vsh_col], ll[depth_col], '-g', label='Vsh')
            ax[0].plot(ll[sw_col], ll[depth_col], '-b', label='Sw')
            ax[0].plot(ll[phi_col], ll[depth_col], '-k', label='phi')
            ax[1].plot(ll.IP_FRMG, ll[depth_col], '-r', label='Gas')
            ax[1].plot(ll.IP_FRMB, ll[depth_col], '-b', label='Brine')
            ax[2].plot(ll.VPVS_FRMG, ll[depth_col], '-r')
            ax[2].plot(ll.VPVS_FRMB, ll[depth_col], '-b')
            ax[3].imshow(cluster, aspect='auto', cmap=cmap, vmin=0, vmax=4)
            
            # Formatting code...
            st.pyplot(fig)

        with tab2:
            st.header("Crossplot Analysis")
            fig, ax = plt.subplots(1, 4, figsize=(16, 6))
            fluids = ['B', 'O', 'G']
            for i, fluid in enumerate(fluids, 1):
                ax[i].scatter(logs[f'IP_FRM{fluid}'], logs[f'VPVS_FRM{fluid}'], 
                            c=logs[f'LFC_{fluid}'], cmap=cmap, vmin=0, vmax=4)
            st.pyplot(fig)

        with tab3:
            st.header("Brine Case AVO Modeling")
        try:
        # Get properties for selected zone
            vp = logs.loc[(logs[depth_col] >= ztop) & (logs[depth_col] <= zbot), 'VP_FRMB'].values
            vs = logs.loc[(logs[depth_col] >= ztop) & (logs[depth_col] <= zbot), 'VS_FRMB'].values
            rho = logs.loc[(logs[depth_col] >= ztop) & (logs[depth_col] <= zbot), 'RHO_FRMB'].values
        
        # Create two layers with 5% contrast
        
           vp_mean = np.nanmean(vp)
           vs_mean = np.nanmean(vs)
           rho_mean = np.nanmean(rho)
        
           vp_data = np.array([vp_mean*1.05, vp_mean*0.95], dtype=np.float64).reshape(-1, 1)
           vs_data = np.array([vs_mean*1.05, vs_mean*0.95], dtype=np.float64).reshape(-1, 1)
           rho_data = np.array([rho_mean*1.05, rho_mean*0.95], dtype=np.float64).reshape(-1, 1)

        # AVO calculation
           nangles = tp.n_angles(0, max_angle)
           rc_zoep = []
           angles = np.linspace(0, max_angle, nangles)
        
        for angle in angles:
            try:
                theta1_samp, rc1, rc2 = tp.calc_theta_rc(
                    theta1_min=0,
                    theta1_step=1,
                    vp=vp_data,
                    vs=vs_data,
                    rho=rho_data,
                    ang=angle
                )
                # Only take the interface between our two layers
                if rc1.shape[0] >= 2 and rc2.shape[0] >= 2:
                    rc_zoep.append([float(rc1[1, 0]), float(rc2[1, 0])])
                else:
                    st.warning(f"Unexpected RC shape at angle {angle}: {rc1.shape}, {rc2.shape}")
                    continue
            except Exception as e:
                st.error(f"Error at angle {angle}: {str(e)}")
                continue

        if not rc_zoep:
            st.error("No valid AVO responses calculated")
            st.stop()

        # Generate wavelet
        wlt_time, wlt_amp = wavelet.ricker(
            sample_rate=sample_rate/1000,
            length=wlt_length/1000,
            c_freq=freq
        )
        
        # Generate synthetic gathers
        syn_zoep = []
        t_samp = tp.time_samples(t_min=0, t_max=0.5)
        
        for i, angle in enumerate(angles):
            try:
                z_int = tp.int_depth(h_int=[500.0], thickness=10)
                t_int = tp.calc_times(z_int, vp_data)
                rc = tp.mod_digitize(rc_zoep[i], t_int, t_samp)
                syn_zoep.append(tp.syn_seis(ref_coef=rc, wav_amp=wlt_amp))
            except Exception as e:
                st.error(f"Error generating synthetic for angle {angle}: {str(e)}")
                continue

        # Create figure
        fig = plt.figure(figsize=(18, 6))
        gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.5])
        
        # AVO Gather
        ax1 = fig.add_subplot(gs[0])
        tp.syn_angle_gather(0.1, 0.25, [t_int]*len(angles), thickness, 
                           [], [], [], [], [], np.array(syn_zoep), 
                           np.array(rc_zoep), t_samp, excursion)
        ax1.set_title(f'Brine Case - {freq}Hz Wavelet')
        
        # AVO Curves
        ax2 = fig.add_subplot(gs[1])
        ax2.plot(angles, [rc[0] for rc in rc_zoep], 'b-', label='Upper Interface')
        ax2.plot(angles, [rc[1] for rc in rc_zoep], 'r-', label='Lower Interface')
        ax2.set_xlabel('Angle (degrees)')
        ax2.set_ylabel('Reflection Coefficient')
        ax2.legend()
        ax2.grid()
        
        # Wavelet Plot
        ax3 = fig.add_subplot(gs[2])
        ax3.plot(wlt_time*1000, wlt_amp, 'k-')
        ax3.set_title(f'Wavelet ({freq}Hz)')
        ax3.set_xlabel('Time (ms)')
        ax3.grid()

        plt.tight_layout()
        st.pyplot(fig)

    except Exception as e:
        st.error(f"AVO Modeling Error: {str(e)}")
        st.error("Please check your input parameters and data quality")

        with tab4:
            st.header("Oil & Gas Cases")
            # Similar implementation as Brine case but for oil and gas
            # ...

    except Exception as e:
        st.error(f"Processing Error: {str(e)}")
else:
    st.info("Please upload a well log CSV file to begin analysis")
