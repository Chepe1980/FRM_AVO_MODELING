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

# Title
st.title("Seismic Fluid Replacement Modeling & AVO Analysis")
st.markdown("Interactive well log analysis with fluid substitution and AVO modeling")

# Custom plotting function
def plot_avo_gather(min_plot_time, max_plot_time, lyr_times, thickness, 
                   syn_zoep, rc_zoep, t, excursion, title="", figsize=(10, 6)):
    """Custom AVO gather plotting function"""
    fig, ax = plt.subplots(figsize=figsize)
    ntrc, nsamp = syn_zoep.shape
    
    # Plot each trace
    for i in range(ntrc):
        trace = syn_zoep[i, :]
        offset = excursion * (i - ntrc/2)
        ax.plot(t, trace + offset, 'k', linewidth=0.5)
    
    # Add layer markers
    for i in range(ntrc):
        ax.axhline(lyr_times[i][0], color='r', linestyle='--', linewidth=0.5)
        ax.axhline(lyr_times[i][1], color='b', linestyle='--', linewidth=0.5)
    
    ax.set_xlabel('Trace Number')
    ax.set_ylabel('Time (s)')
    ax.set_title(title)
    ax.set_ylim(max_plot_time, min_plot_time)
    ax.grid(True)
    fig.set_facecolor('white')
    return fig

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

# FRM function with error handling
def frm(vp1, vs1, rho1, rho_f1, k_f1, rho_f2, k_f2, k0, phi):
    try:
        vp1 = vp1 / 1000.
        vs1 = vs1 / 1000.
        
        # Input validation
        if np.any(vp1 <= 0) or np.any(vs1 <= 0) or np.any(rho1 <= 0):
            raise ValueError("Negative values in Vp, Vs, or Density")
        if np.any(phi < 0) or np.any(phi > 1):
            raise ValueError("Porosity must be 0-1")
        if np.any(k0 <= 0):
            raise ValueError("Bulk modulus must be positive")

        mu1 = rho1 * vs1**2
        k_s1 = rho1 * vp1**2 - (4./3.)*mu1
        
        if np.any(k_s1 <= 0):
            raise ValueError("Invalid bulk modulus calculation")

        denominator = ((phi*k0)/k_f1 + (k_s1/k0) - 1 - phi)
        if np.any(denominator == 0):
            raise ValueError("Division by zero in dry rock calculation")
            
        kdry = (k_s1 * ((phi*k0)/k_f1 + 1 - phi) - k0) / denominator
        
        denominator_gassmann = (phi/k_f2 + (1-phi)/k0 - kdry/k0**2)
        if np.any(denominator_gassmann == 0):
            raise ValueError("Division by zero in Gassmann's equation")
            
        k_s2 = kdry + (1 - (kdry/k0))**2 / denominator_gassmann
        
        if np.any(k_s2 <= 0) or np.any(mu1 <= 0):
            raise ValueError("Invalid modulus values")

        rho2 = rho1 - phi*rho_f1 + phi*rho_f2
        mu2 = mu1
        vp2 = np.sqrt((k_s2 + (4./3)*mu2)/rho2)
        vs2 = np.sqrt(mu2/rho2)
        
        return vp2*1000, vs2*1000, rho2, k_s2
        
    except Exception as e:
        st.error(f"FRM calculation error: {str(e)}")
        return np.nan*vp1, np.nan*vs1, np.nan*rho1, np.nan*k0

# Sidebar inputs
with st.sidebar:
    st.header("Input Parameters")
    uploaded_file = st.file_uploader("Upload Well Log CSV", type=['csv'])
    
    if uploaded_file is not None:
        logs = pd.read_csv(uploaded_file)
        
        st.subheader("Data Columns")
        depth_col = st.selectbox("Depth Column", logs.columns)
        vp_col = st.selectbox("Vp Column", logs.columns)
        vs_col = st.selectbox("Vs Column", logs.columns)
        rho_col = st.selectbox("Density Column", logs.columns)
        vsh_col = st.selectbox("Vshale Column", logs.columns)
        sw_col = st.selectbox("Sw Column", logs.columns)
        phi_col = st.selectbox("Porosity Column", logs.columns)
        
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
        
        st.subheader("Model Parameters")
        ztop = st.number_input("Top Depth (m)", value=float(logs[depth_col].min()))
        zbot = st.number_input("Bottom Depth (m)", value=float(logs[depth_col].max()))
        sand_cutoff = st.slider("Sand Cutoff (Vshale)", 0.0, 1.0, 0.12, 0.01)
        
        st.subheader("Wavelet Parameters")
        freq = st.slider("Center Frequency (Hz)", 10, 100, 30)
        wlt_length = st.slider("Wavelet Length (ms)", 50, 300, 128)
        sample_rate = st.slider("Sample Rate (ms)", 0.01, 0.5, 0.1, 0.01)
        
        st.subheader("AVO Parameters")
        max_angle = st.slider("Maximum Angle (degrees)", 10, 60, 45)
        excursion = st.slider("Trace Excursion", 1, 5, 2)
        thickness = st.slider("Layer Thickness (ms)", 10, 100, 37)

# Main app
if uploaded_file is not None:
    try:
        # Data validation
        for col in [vp_col, vs_col, rho_col]:
            logs[col] = logs[col].clip(lower=0.01)
        logs[phi_col] = logs[phi_col].clip(lower=0, upper=1)
        logs[sw_col] = logs[sw_col].clip(lower=0, upper=1)

        # Perform FRM
        shale = logs[vsh_col].values
        sand = 1 - shale - logs[phi_col].values
        shaleN = shale / (shale+sand)
        sandN = sand / (shale+sand)
        k_u, k_l, mu_u, mu_l, k0, mu0 = vrh([shaleN, sandN], [k_sh, k_qz], [mu_sh, k_qz])

        water = logs[sw_col].values
        hc = 1 - logs[sw_col].values
        _, k_fl, _, _, _, _ = vrh([water, hc], [k_b, k_o], [0, 0])
        rho_fl = water*rho_b + hc*rho_o

        # Apply FRM
        vpb, vsb, rhob, kb = frm(logs[vp_col], logs[vs_col], logs[rho_col], rho_fl, k_fl, rho_b, k_b, k0, logs[phi_col])
        vpo, vso, rhoo, ko = frm(logs[vp_col], logs[vs_col], logs[rho_col], rho_fl, k_fl, rho_o, k_o, k0, logs[phi_col])
        vpg, vsg, rhog, kg = frm(logs[vp_col], logs[vs_col], logs[rho_col], rho_fl, k_fl, rho_g, k_g, k0, logs[phi_col])

        # Lithology classification
        brine_sand = ((logs[vsh_col] <= sand_cutoff) & (logs[sw_col] >= 0.65))
        oil_sand = ((logs[vsh_col] <= sand_cutoff) & (logs[sw_col] < 0.65))
        shale_flag = (logs[vsh_col] > sand_cutoff)

        # Add results to logs
        for fluid in ['B', 'O', 'G']:
            logs[f'VP_FRM{fluid}'] = logs[vp_col]
            logs[f'VS_FRM{fluid}'] = logs[vs_col]
            logs[f'RHO_FRM{fluid}'] = logs[rho_col]
        
        logs.loc[brine_sand|oil_sand, 'VP_FRMB'] = vpb[brine_sand|oil_sand]
        logs.loc[brine_sand|oil_sand, 'VS_FRMB'] = vsb[brine_sand|oil_sand]
        logs.loc[brine_sand|oil_sand, 'RHO_FRMB'] = rhob[brine_sand|oil_sand]
        
        logs.loc[brine_sand|oil_sand, 'VP_FRMO'] = vpo[brine_sand|oil_sand]
        logs.loc[brine_sand|oil_sand, 'VS_FRMO'] = vso[brine_sand|oil_sand]
        logs.loc[brine_sand|oil_sand, 'RHO_FRMO'] = rhoo[brine_sand|oil_sand]
        
        logs.loc[brine_sand|oil_sand, 'VP_FRMG'] = vpg[brine_sand|oil_sand]
        logs.loc[brine_sand|oil_sand, 'VS_FRMG'] = vsg[brine_sand|oil_sand]
        logs.loc[brine_sand|oil_sand, 'RHO_FRMG'] = rhog[brine_sand|oil_sand]

        # Calculate elastic parameters
        for fluid in ['', '_FRMB', '_FRMO', '_FRMG']:
            logs[f'IP{fluid}'] = logs[f'VP{fluid}'] * logs[f'RHO{fluid}']
            logs[f'IS{fluid}'] = logs[f'VS{fluid}'] * logs[f'RHO{fluid}']
            logs[f'VPVS{fluid}'] = logs[f'VP{fluid}'] / logs[f'VS{fluid}']

        # Create LFC flags
        for fluid, val in zip(['B', 'O', 'G'], [1, 2, 3]):
            logs[f'LFC_{fluid}'] = np.where(shale_flag, 4, 
                                          np.where(brine_sand | oil_sand, val, 0))

        # Create tabs
        tab1, tab2, tab3, tab4 = st.tabs(["Well Logs", "Crossplots", "Brine Case", "Oil & Gas Cases"])

        with tab1:
            st.header("Well Log Visualization")
            ll = logs.loc[(logs[depth_col] >= ztop) & (logs[depth_col] <= zbot)]
            
            # Facies colors
            ccc = ['#B3B3B3', 'blue', 'green', 'red', '#996633']
            cmap_facies = colors.ListedColormap(ccc[0:5], 'indexed')
            cluster = np.repeat(np.expand_dims(ll['LFC_B'].values, 1), 100, 1)

            fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(12, 8))
            ax[0].plot(ll[vsh_col], ll[depth_col], '-g', label='Vsh')
            ax[0].plot(ll[sw_col], ll[depth_col], '-b', label='Sw')
            ax[0].plot(ll[phi_col], ll[depth_col], '-k', label='phi')
            ax[1].plot(ll.IP_FRMG, ll[depth_col], '-r', label='Gas')
            ax[1].plot(ll.IP_FRMB, ll[depth_col], '-b', label='Brine')
            ax[1].plot(ll.IP, ll[depth_col], '-', color='0.5', label='Original')
            ax[2].plot(ll.VPVS_FRMG, ll[depth_col], '-r')
            ax[2].plot(ll.VPVS_FRMB, ll[depth_col], '-b')
            ax[2].plot(ll.VPVS, ll[depth_col], '-', color='0.5')
            im = ax[3].imshow(cluster, interpolation='none', aspect='auto', 
                            cmap=cmap_facies, vmin=0, vmax=4)

            for i in ax[:-1]:
                i.set_ylim(ztop, zbot)
                i.invert_yaxis()
                i.grid()
                i.locator_params(axis='x', nbins=4)
            
            ax[0].legend(fontsize='small')
            ax[1].legend(fontsize='small')
            ax[0].set_xlabel("Vcl/phi/Sw")
            ax[1].set_xlabel("Ip [m/s*g/cc]")
            ax[2].set_xlabel("Vp/Vs")
            ax[3].set_xlabel('LFC')
            
            plt.tight_layout()
            st.pyplot(fig)

        with tab2:
            st.header("Crossplot Analysis")
            fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(16, 6))
            
            fluids = ['', '_FRMB', '_FRMO', '_FRMG']
            titles = ['Original', 'Brine', 'Oil', 'Gas']
            lfcs = ['LFC_B', 'LFC_B', 'LFC_O', 'LFC_G']
            
            for i, (fluid, title, lfc) in enumerate(zip(fluids, titles, lfcs)):
                ax[i].scatter(logs[f'IP{fluid}'], logs[f'VPVS{fluid}'], 20, 
                             logs[lfc], marker='o', edgecolors='none', alpha=0.5, 
                             cmap=cmap_facies, vmin=0, vmax=4)
                ax[i].set_title(title)
                ax[i].set_xlabel("Ip [m/s*g/cc]")
                if i == 0:
                    ax[i].set_ylabel("Vp/Vs")
                ax[i].set_xlim(3000, 16000)
                ax[i].set_ylim(1.5, 3)
                ax[i].grid(True)
            
            plt.tight_layout()
            st.pyplot(fig)

        with tab3:
            st.header("Brine Case AVO Modeling")
            
            # Get average properties
            vp_u = logs.loc[(logs[depth_col] >= ztop) & (logs[depth_col] <= zbot), 'VP_FRMB'].values
            vs_u = logs.loc[(logs[depth_col] >= ztop) & (logs[depth_col] <= zbot), 'VS_FRMB'].values
            rho_u = logs.loc[(logs[depth_col] >= ztop) & (logs[depth_col] <= zbot), 'RHO_FRMB'].values
            
            vp_data = [vp_u.mean(), vp_u.mean()*0.95, vp_u.mean()*1.05]
            vs_data = [vs_u.mean(), vs_u.mean()*0.95, vs_u.mean()*1.05]
            rho_data = [rho_u.mean(), rho_u.mean()*0.95, rho_u.mean()*1.05]
            
            # Generate AVO response
            nangles = tp.n_angles(0, max_angle)
            rc_zoep = []
            
            for angle in range(0, nangles):
                _, rc_1, rc_2 = tp.calc_theta_rc(theta1_min=0, theta1_step=1, 
                                                vp=vp_data, vs=vs_data, rho=rho_data, ang=angle)
                rc_zoep.append([rc_1[0, 0], rc_2[0, 0]])
            
            rc_zoep = np.array(rc_zoep)
            
            # Generate synthetic
            wlt_time, wlt_amp = wavelet.ricker(sample_rate=sample_rate/1000, 
                                              length=wlt_length/1000, c_freq=freq)
            t_samp = tp.time_samples(t_min=0, t_max=0.5)
            
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
            t = np.array(t_samp)
            lyr_times = np.array(lyr_times)
            
            # Plot
            col1, col2 = st.columns(2)
            with col1:
                fig1 = plot_avo_gather(0.1, 0.25, lyr_times, thickness, 
                                     syn_zoep, rc_zoep, t, excursion, "Brine Case AVO Gather")
                st.pyplot(fig1)
            
            with col2:
                fig2, ax = plt.subplots(figsize=(8, 4))
                angles = np.arange(0, max_angle+1)
                ax.plot(angles, rc_zoep[:, 0], 'b-', label='Upper Interface')
                ax.plot(angles, rc_zoep[:, 1], 'r-', label='Lower Interface')
                ax.set_xlabel('Angle (degrees)')
                ax.set_ylabel('Reflection Coefficient')
                ax.set_title('Brine AVO Response')
                ax.grid()
                ax.legend()
                st.pyplot(fig2)

        with tab4:
            st.header("Oil & Gas Cases AVO Modeling")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Oil Case")
                vp_o = logs.loc[(logs[depth_col] >= ztop) & (logs[depth_col] <= zbot), 'VP_FRMO'].values
                vs_o = logs.loc[(logs[depth_col] >= ztop) & (logs[depth_col] <= zbot), 'VS_FRMO'].values
                rho_o = logs.loc[(logs[depth_col] >= ztop) & (logs[depth_col] <= zbot), 'RHO_FRMO'].values
                
                vp_data_o = [vp_o.mean(), vp_o.mean()*0.95, vp_o.mean()*1.05]
                vs_data_o = [vs_o.mean(), vs_o.mean()*0.95, vs_o.mean()*1.05]
                rho_data_o = [rho_o.mean(), rho_o.mean()*0.95, rho_o.mean()*1.05]
                
                rc_zoep_o = []
                for angle in range(0, nangles):
                    _, rc_1, rc_2 = tp.calc_theta_rc(theta1_min=0, theta1_step=1, 
                                                    vp=vp_data_o, vs=vs_data_o, rho=rho_data_o, ang=angle)
                    rc_zoep_o.append([rc_1[0, 0], rc_2[0, 0]])
                
                rc_zoep_o = np.array(rc_zoep_o)
                
                syn_zoep_o = []
                for angle in range(0, nangles):
                    z_int = tp.int_depth(h_int=[500.0], thickness=10)
                    t_int = tp.calc_times(z_int, vp_data_o)
                    rc = tp.mod_digitize(rc_zoep_o[angle], t_int, t_samp)
                    s = tp.syn_seis(ref_coef=rc, wav_amp=wlt_amp)
                    syn_zoep_o.append(s)
                
                syn_zoep_o = np.array(syn_zoep_o)
                
                fig_o = plot_avo_gather(0.1, 0.25, lyr_times, thickness, 
                                      syn_zoep_o, rc_zoep_o, t, excursion, "Oil Case AVO Gather")
                st.pyplot(fig_o)
            
            with col2:
                st.subheader("Gas Case")
                vp_g = logs.loc[(logs[depth_col] >= ztop) & (logs[depth_col] <= zbot), 'VP_FRMG'].values
                vs_g = logs.loc[(logs[depth_col] >= ztop) & (logs[depth_col] <= zbot), 'VS_FRMG'].values
                rho_g = logs.loc[(logs[depth_col] >= ztop) & (logs[depth_col] <= zbot), 'RHO_FRMG'].values
                
                vp_data_g = [vp_g.mean(), vp_g.mean()*0.95, vp_g.mean()*1.05]
                vs_data_g = [vs_g.mean(), vs_g.mean()*0.95, vs_g.mean()*1.05]
                rho_data_g = [rho_g.mean(), rho_g.mean()*0.95, rho_g.mean()*1.05]
                
                rc_zoep_g = []
                for angle in range(0, nangles):
                    _, rc_1, rc_2 = tp.calc_theta_rc(theta1_min=0, theta1_step=1, 
                                                    vp=vp_data_g, vs=vs_data_g, rho=rho_data_g, ang=angle)
                    rc_zoep_g.append([rc_1[0, 0], rc_2[0, 0]])
                
                rc_zoep_g = np.array(rc_zoep_g)
                
                syn_zoep_g = []
                for angle in range(0, nangles):
                    z_int = tp.int_depth(h_int=[500.0], thickness=10)
                    t_int = tp.calc_times(z_int, vp_data_g)
                    rc = tp.mod_digitize(rc_zoep_g[angle], t_int, t_samp)
                    s = tp.syn_seis(ref_coef=rc, wav_amp=wlt_amp)
                    syn_zoep_g.append(s)
                
                syn_zoep_g = np.array(syn_zoep_g)
                
                fig_g = plot_avo_gather(0.1, 0.25, lyr_times, thickness, 
                                      syn_zoep_g, rc_zoep_g, t, excursion, "Gas Case AVO Gather")
                st.pyplot(fig_g)

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
else:
    st.info("Please upload a well log CSV file to begin analysis")
