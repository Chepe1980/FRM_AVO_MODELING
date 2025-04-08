import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.colors as colors
from pyavo.seismodel import tuning_prestack as tp
from pyavo.seismodel import wavelet

# Set page config
st.set_page_config(layout="wide", page_title="Seismic Fluid Replacement Modeling")

# Custom CSS for better visualization
st.markdown("""
<style>
    .stPlot {
        border-radius: 10px;
        box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
    }
    .stDataFrame {
        border-radius: 10px;
        box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
    }
    .stSlider {
        padding: 0 20px;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.title("Seismic Fluid Replacement Modeling & AVO Analysis")
st.markdown("""
This interactive app performs fluid replacement modeling (FRM) on well log data and generates synthetic AVO gathers for brine, oil, and gas scenarios.
""")

# VRH function
def vrh(volumes, k, mu):
    f = np.array(volumes).T
    k = np.resize(np.array(k), np.shape(f))
    mu = np.resize(np.array(mu), np.shape(f))

    k_u = np.sum(f*k, axis=1)
    k_l = 1. / np.sum(f/k, axis=1)
    mu_u = np.sum(f*mu, axis=1)
    mu_l = 1. / np.sum(f/mu, axis=1)
    k0 = (k_u+k_l) / 2.
    mu0 = (mu_u+mu_l) / 2.
    return k_u, k_l, mu_u, mu_l, k0, mu0

# FRM function
def frm(vp1, vs1, rho1, rho_f1, k_f1, rho_f2, k_f2, k0, phi):
    vp1 = vp1 / 1000.
    vs1 = vs1 / 1000.
    mu1 = rho1 * vs1**2.
    k_s1 = rho1 * vp1**2 - (4./3.)*mu1

    kdry = (k_s1 * ((phi*k0)/k_f1+1-phi)-k0) / ((phi*k0)/k_f1+(k_s1/k0)-1-phi)

    k_s2 = kdry + (1- (kdry/k0))**2 / ( (phi/k_f2) + ((1-phi)/k0) - (kdry/k0**2) )
    rho2 = rho1-phi * rho_f1+phi * rho_f2
    mu2 = mu1
    vp2 = np.sqrt(((k_s2+(4./3)*mu2))/rho2)
    vs2 = np.sqrt((mu2/rho2))

    return vp2*1000, vs2*1000, rho2, k_s2

# Sidebar for user inputs
with st.sidebar:
    st.header("Input Parameters")
    
    # File upload
    uploaded_file = st.file_uploader("Upload Well Log CSV", type=['csv'])
    
    if uploaded_file is not None:
        logs = pd.read_csv(uploaded_file)
        
        # Show available columns
        st.subheader("Available Columns in Data:")
        st.write(logs.columns.tolist())
        
        # Column selection
        depth_col = st.selectbox("Select Depth Column", logs.columns)
        vp_col = st.selectbox("Select Vp Column", logs.columns)
        vs_col = st.selectbox("Select Vs Column", logs.columns)
        rho_col = st.selectbox("Select Density Column", logs.columns)
        vsh_col = st.selectbox("Select Vshale Column", logs.columns)
        sw_col = st.selectbox("Select Sw Column", logs.columns)
        phi_col = st.selectbox("Select Porosity Column", logs.columns)
        
        # Mineral properties
        st.subheader("Mineral Properties")
        col1, col2 = st.columns(2)
        with col1:
            rho_qz = st.number_input("Quartz Density (g/cc)", value=2.65, step=0.01)
            k_qz = st.number_input("Quartz Bulk Modulus (GPa)", value=37.0, step=0.1)
            mu_qz = st.number_input("Quartz Shear Modulus (GPa)", value=44.0, step=0.1)
        with col2:
            rho_sh = st.number_input("Shale Density (g/cc)", value=2.81, step=0.01)
            k_sh = st.number_input("Shale Bulk Modulus (GPa)", value=15.0, step=0.1)
            mu_sh = st.number_input("Shale Shear Modulus (GPa)", value=5.0, step=0.1)
        
        # Fluid properties
        st.subheader("Fluid Properties")
        col1, col2, col3 = st.columns(3)
        with col1:
            rho_b = st.number_input("Brine Density (g/cc)", value=1.09, step=0.01)
            k_b = st.number_input("Brine Bulk Modulus (GPa)", value=2.8, step=0.1)
        with col2:
            rho_o = st.number_input("Oil Density (g/cc)", value=0.78, step=0.01)
            k_o = st.number_input("Oil Bulk Modulus (GPa)", value=0.94, step=0.1)
        with col3:
            rho_g = st.number_input("Gas Density (g/cc)", value=0.25, step=0.01)
            k_g = st.number_input("Gas Bulk Modulus (GPa)", value=0.06, step=0.1)
        
        # Layer selection
        st.subheader("Layer Selection for AVO Modeling")
        ztop = st.number_input("Top Depth (m)", value=float(logs[depth_col].min()))
        zbot = st.number_input("Bottom Depth (m)", value=float(logs[depth_col].max()))
        
        # Sand cutoff
        sand_cutoff = st.slider("Sand Cutoff (Vshale)", 0.0, 1.0, 0.12, 0.01)
        
        # Wavelet parameters
        st.subheader("Wavelet Parameters")
        freq = st.slider("Center Frequency (Hz)", 10, 100, 30)
        wlt_length = st.slider("Wavelet Length (ms)", 50, 300, 128)
        sample_rate = st.slider("Sample Rate (ms)", 0.01, 0.5, 0.1, 0.01)
        
        # AVO parameters
        st.subheader("AVO Parameters")
        max_angle = st.slider("Maximum Angle (degrees)", 10, 60, 45)
        excursion = st.slider("Trace Excursion", 1, 5, 2)
        thickness = st.slider("Layer Thickness (ms)", 10, 100, 37)

# Main content area
if uploaded_file is not None:
    # Process the data
    try:
        # Perform FRM
        shale = logs[vsh_col].values
        sand = 1 - shale - logs[phi_col].values
        shaleN = shale / (shale+sand)
        sandN = sand / (shale+sand)
        k_u, k_l, mu_u, mu_l, k0, mu0 = vrh([shaleN, sandN], [k_sh, k_qz], [mu_sh, mu_qz])

        water = logs[sw_col].values
        hc = 1 - logs[sw_col].values
        tmp, k_fl, tmp, tmp, tmp, tmp = vrh([water, hc], [k_b, k_o], [0, 0])
        rho_fl = water*rho_b + hc*rho_o

        # Apply FRM
        vpb, vsb, rhob, kb = frm(logs[vp_col], logs[vs_col], logs[rho_col], rho_fl, k_fl, rho_b, k_b, k0, logs[phi_col])
        vpo, vso, rhoo, ko = frm(logs[vp_col], logs[vs_col], logs[rho_col], rho_fl, k_fl, rho_o, k_o, k0, logs[phi_col])
        vpg, vsg, rhog, kg = frm(logs[vp_col], logs[vs_col], logs[rho_col], rho_fl, k_fl, rho_g, k_g, k0, logs[phi_col])

        # Lithology classification
        brine_sand = ((logs[vsh_col] <= sand_cutoff) & (logs[sw_col] >= 0.65))
        oil_sand = ((logs[vsh_col] <= sand_cutoff) & (logs[sw_col] < 0.65))
        shale_flag = (logs[vsh_col] > sand_cutoff)

        # Add results to dataframe
        logs['VP_FRMB'] = logs[vp_col].copy()
        logs['VS_FRMB'] = logs[vs_col].copy()
        logs['RHO_FRMB'] = logs[rho_col].copy()
        logs.loc[brine_sand|oil_sand, 'VP_FRMB'] = vpb[brine_sand|oil_sand]
        logs.loc[brine_sand|oil_sand, 'VS_FRMB'] = vsb[brine_sand|oil_sand]
        logs.loc[brine_sand|oil_sand, 'RHO_FRMB'] = rhob[brine_sand|oil_sand]
        logs['IP_FRMB'] = logs.VP_FRMB*logs.RHO_FRMB
        logs['IS_FRMB'] = logs.VS_FRMB*logs.RHO_FRMB
        logs['VPVS_FRMB'] = logs.VP_FRMB/logs.VS_FRMB

        logs['VP_FRMO'] = logs[vp_col].copy()
        logs['VS_FRMO'] = logs[vs_col].copy()
        logs['RHO_FRMO'] = logs[rho_col].copy()
        logs.loc[brine_sand|oil_sand, 'VP_FRMO'] = vpo[brine_sand|oil_sand]
        logs.loc[brine_sand|oil_sand, 'VS_FRMO'] = vso[brine_sand|oil_sand]
        logs.loc[brine_sand|oil_sand, 'RHO_FRMO'] = rhoo[brine_sand|oil_sand]
        logs['IP_FRMO'] = logs.VP_FRMO*logs.RHO_FRMO
        logs['IS_FRMO'] = logs.VS_FRMO*logs.RHO_FRMO
        logs['VPVS_FRMO'] = logs.VP_FRMO/logs.VS_FRMO

        logs['VP_FRMG'] = logs[vp_col].copy()
        logs['VS_FRMG'] = logs[vs_col].copy()
        logs['RHO_FRMG'] = logs[rho_col].copy()
        logs.loc[brine_sand|oil_sand, 'VP_FRMG'] = vpg[brine_sand|oil_sand]
        logs.loc[brine_sand|oil_sand, 'VS_FRMG'] = vsg[brine_sand|oil_sand]
        logs.loc[brine_sand|oil_sand, 'RHO_FRMG'] = rhog[brine_sand|oil_sand]
        logs['IP_FRMG'] = logs.VP_FRMG*logs.RHO_FRMG
        logs['IS_FRMG'] = logs.VS_FRMG*logs.RHO_FRMG
        logs['VPVS_FRMG'] = logs.VP_FRMG/logs.VS_FRMG

        # LFC flags
        logs['LFC_B'] = np.where(shale_flag, 4, np.where(brine_sand | oil_sand, 1, 0))
        logs['LFC_O'] = np.where(shale_flag, 4, np.where(brine_sand | oil_sand, 2, 0))
        logs['LFC_G'] = np.where(shale_flag, 4, np.where(brine_sand | oil_sand, 3, 0))

        # Create tabs for different visualizations
        tab1, tab2, tab3, tab4 = st.tabs(["Well Logs", "Crossplots", "Brine Case", "Oil & Gas Cases"])

        with tab1:
            st.header("Well Log Visualization")
    
    # Filter logs for selected depth range
             ll = logs.loc[(logs[depth_col] >= ztop) & (logs[depth_col] <= zbot)]
    
    # Create cluster for facies display - FIXED VERSION
    cluster_data = np.expand_dims(ll['LFC_B'].values, axis=1)
    cluster = np.repeat(cluster_data, 100, axis=1)
    
    ccc = ['#B3B3B3', 'blue', 'green', 'red', '#996633']
    cmap_facies = colors.ListedColormap(ccc[0:len(ccc)], 'indexed')

    # Rest of the plotting code remains the same...
    fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(12, 8))
    ax[0].plot(ll[vsh_col], ll[depth_col], '-g', label='Vsh')
    ax[0].plot(ll[sw_col], ll[depth_col], '-b', label='Sw')
    ax[0].plot(ll[phi_col], ll[depth_col], '-k', label='phi')
    ax[1].plot(ll.IP_FRMG, ll[depth_col], '-r')
    ax[1].plot(ll.IP_FRMB, ll[depth_col], '-b')
    ax[1].plot(ll[vp_col]*ll[rho_col], ll[depth_col], '-', color='0.5')
    ax[2].plot(ll.VPVS_FRMG, ll[depth_col], '-r')
    ax[2].plot(ll.VPVS_FRMB, ll[depth_col], '-b')
    ax[2].plot(ll[vp_col]/ll[vs_col], ll[depth_col], '-', color='0.5')
    im = ax[3].imshow(cluster, interpolation='none', aspect='auto', cmap=cmap_facies, vmin=0, vmax=4)

    # Format plots
    for i in ax[:-1]:
        i.set_ylim(ztop, zbot)
        i.invert_yaxis()
        i.grid()
        i.locator_params(axis='x', nbins=4)
    
    ax[0].legend(fontsize='small', loc='lower right')
    ax[0].set_xlabel("Vcl/phi/Sw")
    ax[0].set_xlim(-.1, 1.1)
    ax[1].set_xlabel("Ip [m/s*g/cc]")
    ax[1].set_xlim(6000, 15000)
    ax[2].set_xlabel("Vp/Vs")
    ax[2].set_xlim(1.5, 2)
    ax[3].set_xlabel('LFC')
    ax[1].set_yticklabels([])
    ax[2].set_yticklabels([])
    ax[3].set_yticklabels([])
    ax[3].set_xticklabels([])
    
    plt.tight_layout()
    st.pyplot(fig)

        with tab2:
            st.header("Crossplot Analysis")
            
            fig, ax = plt.subplots(nrows=1, ncols=4, sharey=True, sharex=True, figsize=(16, 6))
            
            ax[0].scatter(logs[vp_col]*logs[rho_col], logs[vp_col]/logs[vs_col], 20, logs.LFC_B, 
                         marker='o', edgecolors='none', alpha=0.5, cmap=cmap_facies, vmin=0, vmax=4)
            ax[1].scatter(logs.IP_FRMB, logs.VPVS_FRMB, 20, logs.LFC_B, 
                         marker='o', edgecolors='none', alpha=0.5, cmap=cmap_facies, vmin=0, vmax=4)
            ax[2].scatter(logs.IP_FRMO, logs.VPVS_FRMO, 20, logs.LFC_O, 
                         marker='o', edgecolors='none', alpha=0.5, cmap=cmap_facies, vmin=0, vmax=4)
            ax[3].scatter(logs.IP_FRMG, logs.VPVS_FRMG, 20, logs.LFC_G, 
                         marker='o', edgecolors='none', alpha=0.5, cmap=cmap_facies, vmin=0, vmax=4)
            
            ax[0].set_xlim(3000, 16000)
            ax[0].set_ylim(1.5, 3)
            ax[0].set_title('Original Data')
            ax[1].set_title('FRM to Brine')
            ax[2].set_title('FRM to Oil')
            ax[3].set_title('FRM to Gas')
            
            plt.tight_layout()
            st.pyplot(fig)

        with tab3:
            st.header("Brine Case AVO Modeling")
            
            try:
                # Get average properties for the selected zone
                vp_u = logs.loc[(logs[depth_col] >= ztop) & (logs[depth_col] <= zbot), 'VP_FRMB'].values
                vs_u = logs.loc[(logs[depth_col] >= ztop) & (logs[depth_col] <= zbot), 'VS_FRMB'].values
                rho_u = logs.loc[(logs[depth_col] >= ztop) & (logs[depth_col] <= zbot), 'RHO_FRMB'].values
                
                # Convert to numpy arrays and handle potential NaN values
                vp_u = np.array(vp_u, dtype=np.float64)
                vs_u = np.array(vs_u, dtype=np.float64)
                rho_u = np.array(rho_u, dtype=np.float64)
                
                # Data validation
                if np.isnan(vp_u).any() or np.isnan(vs_u).any() or np.isnan(rho_u).any():
                    vp_u = np.nan_to_num(vp_u, nan=np.nanmean(vp_u))
                    vs_u = np.nan_to_num(vs_u, nan=np.nanmean(vs_u))
                    rho_u = np.nan_to_num(rho_u, nan=np.nanmean(rho_u))
                
                # Create properly formatted numpy arrays
                vp_mean = float(np.nanmean(vp_u))
                vs_mean = float(np.nanmean(vs_u))
                rho_mean = float(np.nanmean(rho_u))
                
                vp_data = np.array([vp_mean, vp_mean * 0.95, vp_mean * 1.05], dtype=np.float64)
                vs_data = np.array([vs_mean, vs_mean * 0.95, vs_mean * 1.05], dtype=np.float64)
                rho_data = np.array([rho_mean, rho_mean * 0.95, rho_mean * 1.05], dtype=np.float64)
                
                # Reshape if needed by the AVO function
                if hasattr(tp.calc_theta_rc, '__code__') and tp.calc_theta_rc.__code__.co_argcount > 3:
                    vp_data = vp_data.reshape(1, -1)
                    vs_data = vs_data.reshape(1, -1)
                    rho_data = rho_data.reshape(1, -1)
                
                # Generate AVO response
                nangles = tp.n_angles(0, max_angle)
                rc_zoep = []
                theta1 = []
                
                for angle in range(0, nangles):
                    try:
                        theta1_samp, rc_1, rc_2 = tp.calc_theta_rc(
                            theta1_min=0, 
                            theta1_step=1, 
                            vp=vp_data, 
                            vs=vs_data, 
                            rho=rho_data, 
                            ang=angle
                        )
                        rc_zoep.append([float(rc_1[0, 0]), float(rc_2[0, 0])])
                        theta1.append(float(theta1_samp))
                    except Exception as e:
                        st.error(f"Error at angle {angle}: {str(e)}")
                        continue
                
                if not rc_zoep:
                    st.error("No AVO responses calculated - check input parameters")
                    st.stop()
                
                # Generate wavelet
                wlt_time, wlt_amp = wavelet.ricker(
                    sample_rate=sample_rate/1000, 
                    length=wlt_length/1000, 
                    c_freq=freq
                )
                t_samp = tp.time_samples(t_min=0, t_max=0.5)
                
                # Generate synthetic gathers
                syn_zoep = []
                lyr_times = []
                
                for angle in range(0, nangles):
                    z_int = tp.int_depth(h_int=[500.0], thickness=10)
                    t_int = tp.calc_times(z_int, vp_data)
                    lyr_times.append(t_int)
                    rc = tp.mod_digitize(rc_zoep[angle], t_int, t_samp)
                    s = tp.syn_seis(ref_coef=rc, wav_amp=wlt_amp)
                    syn_zoep.append(s)
                
                syn_zoep = np.array(syn_zoep)
                rc_zoep = np.array(rc_zoep)
                t = np.array(t_samp)
                lyr_times = np.array(lyr_times)
                
                # Create figure with 3 panels
                fig = plt.figure(figsize=(18, 6))
                gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.5])
                
                # AVO Gather
                ax1 = fig.add_subplot(gs[0])
                tp.syn_angle_gather(0.1, 0.25, lyr_times, thickness, [], [], 
                                   [], [], [], syn_zoep, rc_zoep, t, excursion)
                ax1.set_title(f'Brine Case - {freq}Hz Wavelet')
                
                # AVO Curves
                ax2 = fig.add_subplot(gs[1])
                angles = np.arange(0, max_angle+1)
                ax2.plot(angles, rc_zoep[:, 0], 'b-', label='Upper Interface')
                ax2.plot(angles, rc_zoep[:, 1], 'r-', label='Lower Interface')
                ax2.set_xlabel('Angle (degrees)')
                ax2.set_ylabel('Reflection Coefficient')
                ax2.set_title('AVO Response')
                ax2.grid()
                ax2.legend()
                
                # Wavelet Plot
                ax3 = fig.add_subplot(gs[2])
                ax3.plot(wlt_time*1000, wlt_amp, 'k-', linewidth=2)
                ax3.set_title(f'Wavelet ({freq}Hz)')
                ax3.set_xlabel('Time (ms)')
                ax3.set_ylabel('Amplitude')
                ax3.grid()
                
                plt.tight_layout()
                st.pyplot(fig)

            except Exception as e:
                st.error(f"An error occurred in Brine Case AVO Modeling: {str(e)}")
                st.error(f"Debug info - vp_data: {vp_data if 'vp_data' in locals() else 'Not defined'}, type: {type(vp_data) if 'vp_data' in locals() else 'Not defined'}")

        with tab4:
            st.header("Oil & Gas Cases AVO Modeling")
            
            try:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Oil Case")
                    
                    # Get average properties for oil case
                    vp_u = logs.loc[(logs[depth_col] >= ztop) & (logs[depth_col] <= zbot), 'VP_FRMO'].values
                    vs_u = logs.loc[(logs[depth_col] >= ztop) & (logs[depth_col] <= zbot), 'VS_FRMO'].values
                    rho_u = logs.loc[(logs[depth_col] >= ztop) & (logs[depth_col] <= zbot), 'RHO_FRMO'].values
                    
                    # Convert to numpy arrays and handle potential NaN values
                    vp_u = np.array(vp_u, dtype=np.float64)
                    vs_u = np.array(vs_u, dtype=np.float64)
                    rho_u = np.array(rho_u, dtype=np.float64)
                    
                    if np.isnan(vp_u).any() or np.isnan(vs_u).any() or np.isnan(rho_u).any():
                        vp_u = np.nan_to_num(vp_u, nan=np.nanmean(vp_u))
                        vs_u = np.nan_to_num(vs_u, nan=np.nanmean(vs_u))
                        rho_u = np.nan_to_num(rho_u, nan=np.nanmean(rho_u))
                    
                    # Create properly formatted numpy arrays
                    vp_mean = float(np.nanmean(vp_u))
                    vs_mean = float(np.nanmean(vs_u))
                    rho_mean = float(np.nanmean(rho_u))
                    
                    vp_data = np.array([vp_mean, vp_mean * 0.95, vp_mean * 1.05], dtype=np.float64)
                    vs_data = np.array([vs_mean, vs_mean * 0.95, vs_mean * 1.05], dtype=np.float64)
                    rho_data = np.array([rho_mean, rho_mean * 0.95, rho_mean * 1.05], dtype=np.float64)
                    
                    if hasattr(tp.calc_theta_rc, '__code__') and tp.calc_theta_rc.__code__.co_argcount > 3:
                        vp_data = vp_data.reshape(1, -1)
                        vs_data = vs_data.reshape(1, -1)
                        rho_data = rho_data.reshape(1, -1)
                    
                    # Generate AVO response for oil
                    rc_zoep_o = []
                    for angle in range(0, nangles):
                        theta1_samp, rc_1, rc_2 = tp.calc_theta_rc(
                            theta1_min=0, 
                            theta1_step=1, 
                            vp=vp_data, 
                            vs=vs_data, 
                            rho=rho_data, 
                            ang=angle
                        )
                        rc_zoep_o.append([float(rc_1[0, 0]), float(rc_2[0, 0])])
                    
                    # Generate synthetic gathers for oil
                    syn_zoep_o = []
                    for angle in range(0, nangles):
                        rc = tp.mod_digitize(rc_zoep_o[angle], lyr_times[angle], t_samp)
                        s = tp.syn_seis(ref_coef=rc, wav_amp=wlt_amp)
                        syn_zoep_o.append(s)
                    
                    syn_zoep_o = np.array(syn_zoep_o)
                    rc_zoep_o = np.array(rc_zoep_o)
                    
                    # Create figure for oil case
                    fig = plt.figure(figsize=(18, 6))
                    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.5])
                    
                    # AVO Gather
                    ax1 = fig.add_subplot(gs[0])
                    tp.syn_angle_gather(0.1, 0.25, lyr_times, thickness, [], [], 
                                      [], [], [], syn_zoep_o, rc_zoep_o, t, excursion)
                    ax1.set_title('Oil Case')
                    
                    # AVO Curves
                    ax2 = fig.add_subplot(gs[1])
                    ax2.plot(angles, rc_zoep_o[:, 0], 'b-', label='Upper Interface')
                    ax2.plot(angles, rc_zoep_o[:, 1], 'r-', label='Lower Interface')
                    ax2.set_xlabel('Angle (degrees)')
                    ax2.set_ylabel('Reflection Coefficient')
                    ax2.set_title('Oil AVO Response')
                    ax2.grid()
                    ax2.legend()
                    
                    # Wavelet Plot
                    ax3 = fig.add_subplot(gs[2])
                    ax3.plot(wlt_time*1000, wlt_amp, 'k-', linewidth=2)
                    ax3.set_title(f'Wavelet ({freq}Hz)')
                    ax3.set_xlabel('Time (ms)')
                    ax3.set_ylabel('Amplitude')
                    ax3.grid()
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                
                with col2:
                    st.subheader("Gas Case")
                    
                    # Get average properties for gas case
                    vp_u = logs.loc[(logs[depth_col] >= ztop) & (logs[depth_col] <= zbot), 'VP_FRMG'].values
                    vs_u = logs.loc[(logs[depth_col] >= ztop) & (logs[depth_col] <= zbot), 'VS_FRMG'].values
                    rho_u = logs.loc[(logs[depth_col] >= ztop) & (logs[depth_col] <= zbot), 'RHO_FRMG'].values
                    
                    # Convert to numpy arrays and handle potential NaN values
                    vp_u = np.array(vp_u, dtype=np.float64)
                    vs_u = np.array(vs_u, dtype=np.float64)
                    rho_u = np.array(rho_u, dtype=np.float64)
                    
                    if np.isnan(vp_u).any() or np.isnan(vs_u).any() or np.isnan(rho_u).any():
                        vp_u = np.nan_to_num(vp_u, nan=np.nanmean(vp_u))
                        vs_u = np.nan_to_num(vs_u, nan=np.nanmean(vs_u))
                        rho_u = np.nan_to_num(rho_u, nan=np.nanmean(rho_u))
                    
                    # Create properly formatted numpy arrays
                    vp_mean = float(np.nanmean(vp_u))
                    vs_mean = float(np.nanmean(vs_u))
                    rho_mean = float(np.nanmean(rho_u))
                    
                    vp_data = np.array([vp_mean, vp_mean * 0.95, vp_mean * 1.05], dtype=np.float64)
                    vs_data = np.array([vs_mean, vs_mean * 0.95, vs_mean * 1.05], dtype=np.float64)
                    rho_data = np.array([rho_mean, rho_mean * 0.95, rho_mean * 1.05], dtype=np.float64)
                    
                    if hasattr(tp.calc_theta_rc, '__code__') and tp.calc_theta_rc.__code__.co_argcount > 3:
                        vp_data = vp_data.reshape(1, -1)
                        vs_data = vs_data.reshape(1, -1)
                        rho_data = rho_data.reshape(1, -1)
                    
                    # Generate AVO response for gas
                    rc_zoep_g = []
                    for angle in range(0, nangles):
                        theta1_samp, rc_1, rc_2 = tp.calc_theta_rc(
                            theta1_min=0, 
                            theta1_step=1, 
                            vp=vp_data, 
                            vs=vs_data, 
                            rho=rho_data, 
                            ang=angle
                        )
                        rc_zoep_g.append([float(rc_1[0, 0]), float(rc_2[0, 0])])
                    
                    # Generate synthetic gathers for gas
                    syn_zoep_g = []
                    for angle in range(0, nangles):
                        rc = tp.mod_digitize(rc_zoep_g[angle], lyr_times[angle], t_samp)
                        s = tp.syn_seis(ref_coef=rc, wav_amp=wlt_amp)
                        syn_zoep_g.append(s)
                    
                    syn_zoep_g = np.array(syn_zoep_g)
                    rc_zoep_g = np.array(rc_zoep_g)
                    
                    # Create figure for gas case
                    fig = plt.figure(figsize=(18, 6))
                    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.5])
                    
                    # AVO Gather
                    ax1 = fig.add_subplot(gs[0])
                    tp.syn_angle_gather(0.1, 0.25, lyr_times, thickness, [], [], 
                                      [], [], [], syn_zoep_g, rc_zoep_g, t, excursion)
                    ax1.set_title('Gas Case')
                    
                    # AVO Curves
                    ax2 = fig.add_subplot(gs[1])
                    ax2.plot(angles, rc_zoep_g[:, 0], 'b-', label='Upper Interface')
                    ax2.plot(angles, rc_zoep_g[:, 1], 'r-', label='Lower Interface')
                    ax2.set_xlabel('Angle (degrees)')
                    ax2.set_ylabel('Reflection Coefficient')
                    ax2.set_title('Gas AVO Response')
                    ax2.grid()
                    ax2.legend()
                    
                    # Wavelet Plot
                    ax3 = fig.add_subplot(gs[2])
                    ax3.plot(wlt_time*1000, wlt_amp, 'k-', linewidth=2)
                    ax3.set_title(f'Wavelet ({freq}Hz)')
                    ax3.set_xlabel('Time (ms)')
                    ax3.set_ylabel('Amplitude')
                    ax3.grid()
                    
                    plt.tight_layout()
                    st.pyplot(fig)

            except Exception as e:
                st.error(f"An error occurred in Oil & Gas Cases AVO Modeling: {str(e)}")
                st.error(f"Debug info - vp_u: {vp_u[:5] if 'vp_u' in locals() else 'Not defined'}, type: {type(vp_u) if 'vp_u' in locals() else 'Not defined'}")

    except Exception as e:
        st.error(f"An error occurred during data processing: {str(e)}")
else:
    st.info("Please upload a well log CSV file to begin analysis.")
