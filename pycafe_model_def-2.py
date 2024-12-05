#!/usr/bin/env python
# coding: utf-8

"""
PYCAFE MODEL (December, 2024)

Description:
    Computes daily net primary production following the Carbon, Absorption, Fluorescence
    Euphotic-resolving (CAFE) model. For a full description of the original model, please see:

        Silsbe, G.M., M.J. Behrenfeld, K.H. Halsey, A.J. Milligan, and T.K. Westberry. 2016.
        The CAFE model. A net production model for global ocean phytoplankton. Global
        Biogeochemical Cycles. doi: 10.1002/2016GB005521.
        
    'PyCafe' model was designed to perform computations across latitude and longitude chunks 
        for efficient processing of spatial data.

Model Inputs:
    PAR (shape: lat, lon)     : Daily photosynthetic active radiation [mol photons m^-2 day^-1].
    chl (shape: lat, lon)     : Chlorophyll concentration, NASA OCI algorithm [mg m^-3].
    mld (shape: lat, lon)     : Mixed layer depth [m].
    lat_grid (shape: lat, lon): Latitude [degrees, north positive].
    yd (int)  : Day of the year.
    aph_443 (shape: lat, lon) : Absorption due to phytoplankton at 443 nm, NASA GIOP algorithm [m^-1].
    adg_443 (shape: lat, lon) : Absorption due to gelbstoff and detrital material at 443 nm, NASA GIOP model [m^-1].
    bbp_443 (shape: lat, lon) : Particulate backscatter at 443 nm, NASA GIOP model [m^-1].
    bbp_s (shape: lat, lon)   : Backscattering spectral parameter for GIOP model [dimensionless].
    sst (shape: lat, lon)     : Sea surface temperature [degrees Celsius]. Only used to compute backscattering of pure seawater in function betasw_ZHH2009.

Modeling Steps:
    1. Declare all local variables.
    2. Derive inherent optical properties (IOPs) at 10 nm increments from 400 to 700 nm.
    3. Calculate the amount of energy in the euphotic zone absorbed by phytoplankton.
    4. Derive the spectral attenuation coefficient and the attenuation coefficient of PAR.
    5. Derive conversion factor (Eu) that converts downwelling planar irradiance to scalar irradiance.
    6. Calculate the photoacclimation parameter (Ek) through depth.
    7. Scale Ek to a spectrally explicit parameter (KPUR).
    8. Model the maximum quantum efficiency of net carbon fixation (phi_max).
    9. If the mixed layer depth (MLD) is shallower than the euphotic depth, apply a scaling factor to
       phytoplankton absorption beneath the MLD and adjust the energy absorbed by phytoplankton.
    10. Derive net phytoplankton production (NPP).

Defined constants and parameters:
    - Wavelengths (shape: 31): Range from 400 to 700 nm in 10 nm increments.
    - Absorption due to pure water (aw): Predefined constants for absorption at each wavelength.
    - Spectral shape of phytoplankton absorption (A_Bricaud and E_Bricaud): Predefined arrays.
    - Spectral shape of PAR (PAR_spectrum): Predefined distribution of PAR across wavelengths.

Computed variables and parameters:
    - aphi : Absorption due to phytoplankton. (shape: lat, lon, wavelength)
    - bbw  : Backscattering of pure water. (shape: lat, lon, wavelength)
    - a  : Total absorption. (shape: lat, lon, wavelength)
    - bb : Total backscattering. (shape: lat, lon, wavelength)
    
    - decl : Solar declination [radians]. 
    - DL : Day length [fraction of day]. (shape: lat, lon)
    - solzen : Solar zenith angle [degrees]. (shape: lat, lon)
    - m0 : Coefficient to calculate kd (Lee et al. 2005) [m-1].
    - kd : Downwelling attenuation coefficient of irradiance [m^-1]. (shape: lat, lon, wavelength)
    - kdpar: Downwelling attenuation coefficient of PAR [m-1]. (shape: lat, lon)
    - zeu : Euphotic depth [m]. (shape: lat, lon)
    
    - tseq : Fractional time of day - 51 increments. (shape: time) 
    - zseq : Euphotic depth divided into 101 increments. (shape: lat, lon, depth)
    - delz : Depth of zseq. (shape: lat, lon)
    - absorbed_photons : Energy absorbed by phytoplankton [mol photons m^-2 day^-1]. (shape: lat, lon)
    - PAR_noon : PAR at solar noon [mol photons m-2 day-1 wv-]. (shape: lat, lon, wavelength)
    - E_tzw : Irradiance through time, depth and wv. (shape: lat, lon, time, depth, wavelength)
    - A_tzw : Absorbed photons through time, depth and wv. (shape: lat, lon, time, depth, wavelength)
    - E_tz : Irradiance through time and  depth. (shape: lat, lon, time, depth)
    - A_tz : Absorbed photons through time and depth. (shape: lat, lon, time, depth)
    - AP_z : Absorbed photons through depth. (shape: lat, lon, depth)
    - AP : Absorbed photons [mol m-2 day-1]. (shape: lat, lon)
    - Eu : conversion factor that converts downwelling planar irradiance to scalar irradiance. (shape: lat, lon)
    
    - IML : Median irradiance in the mixed layer [umol m-2 day-1]. (shape: lat, lon)
    - Ek : Photoacclimation parameter through depth [mol photons m^-2 day^-1] (Behrenfeld et al. 2016). (shape: lat, lon, depth)
    - Eg_mld : Growth irradiance when mld < zeu. (shape: lat, lon, depth)
    - Eg : Growth irradiance when mld < zeu & zseq > mld. (shape: lat, lon, depth)

    - mean_aphi : Spectrally averaged absorption due to phytoplankton. (shape: lat, lon)
    - KPUR : Spectrally explicit Ek. (shape: lat, lon, depth)
    
    - phimax : Maximum quantum yield of net carbon fixation [mol C (mol photons)^-1].
    - phirange : Range of phimax values. [0.018, 0.030]
    - Ekrange : Relates phimax to Ek. [150*0.086400, 10*0.0864]
    - slope : Relates phimax to Ek.
    
    - aphi_fact : Parameter to increase aphi beneath the MLD.
    
    - NPP : Net primary production [mg C m^-2 day^-1]. (shape: lat, lon)

Note:
    For further understanding of individual calculations and steps, refer to the provided reference.
"""

import numpy as np
import xarray as xr

PI = np.pi

def betasw_ZHH2009(lambda_nm, S, Tc):
    """
    Calculate backscattering of pure seawater as a function of wavelength, salinity, and temperature.

    Parameters:
    - lambda_nm: Wavelength in nanometers (nm)
    - S: Salinity (must be scalar)
    - Tc: Temperature in degrees Celsius (must be scalar)

    Returns:
    - bsw: Total scattering coefficient
    """

    # Constants
    Na = 6.0221417930e23  # Avogadro's constant
    Kbz = 1.3806503e-23   # Boltzmann constant
    Tk = Tc + 273.15      # Absolute temperature in Kelvin
    M0 = 18e-3            # Molecular weight of water in kg/mol
    delta = 0.039         # Depolarization ratio

    # Define angles in radians for scattering calculations
    rad = np.arange(0, 181) * PI / 180  # 0 to 180 degrees in radians

    # Refractive index of air from Ciddor (1996)
    lambda_um = lambda_nm / 1e3  # Convert wavelength to micrometers
    n_air = 1.0 + (5792105.0 / (238.0185 - 1 / (lambda_um**2)) + 167917.0 / (57.362 - 1 / (lambda_um**2))) / 1e8

    # Refractive index of seawater from Quan and Fry (1994)
    n0, n1, n2, n3, n4 = 1.31405, 1.779e-4, -1.05e-6, 1.6e-8, -2.02e-6
    n5, n6, n7, n8, n9 = 15.868, 0.01155, -0.00423, -4382, 1.1455e6

    nsw = n0 + (n1 + n2 * Tc + n3 * (Tc**2)) * S + n4 * (Tc**2) + (n5 + n6 * S + n7 * Tc) / lambda_nm + n8 / (lambda_nm**2) + n9 / (lambda_nm**3)
    nsw *= n_air
    dnswds = (n1 + n2 * Tc + n3 * (Tc**2) + n6 / lambda_nm) * n_air

    # Isothermal compressibility from Lepple & Millero (1971)
    kw = 19652.21 + 148.4206 * Tc - 2.327105 * (Tc**2) + 1.360477e-2 * (Tc**3) - 5.155288e-5 * (Tc**4)
    Btw_cal = 1 / kw

    # Seawater secant bulk from Kell sound measurement in pure water
    Btw = (50.88630 + 0.717582 * Tc + 0.7819867e-3 * (Tc**2) + 31.62214e-6 * (Tc**3) - 0.1323594e-6 * (Tc**4) + 0.634575e-9 * (Tc**5)) / (1 + 21.65928e-3 * Tc) * 1e-6

    # Seawater secant bulk modulus
    g0 = 54.6746 - 0.603459 * Tc + 1.09987e-2 * (Tc**2) - 6.167e-5 * (Tc**3)
    g1= 7.944e-2 + 1.6483e-2 * Tc - 5.3009e-4 * (Tc**2)
    Ks = kw + g0 * S + g1 * (S**1.5)
    
    #calculate seawater isothermal compressibility from the secant bulk
    IsoComp = 1 / Ks * 1e-5  # units Pa

    # Density of pure water and seawater from UNESCO 1981, kg/m3
    a0, a1, a2, a3, a4 = 8.24493e-1, -4.0899e-3, 7.6438e-5, -8.2467e-7, 5.3875e-9
    a5, a6, a7, a8 = -5.72466e-3, 1.02270e-4, -1.6546e-6, 4.8314e-4
    b0, b1, b2, b3, b4, b5 = 999.842594, 6.793952e-2, -9.09529e-3, 1.001685e-4, -1.120083e-6, 6.536332e-9

    # density for pure water
    density = b0 + b1 * Tc + b2 * (Tc**2) + b3 * (Tc**3) + b4 * (Tc**4) + b5 * (Tc**5)
    #density for pure seawater 
    density += (a0 + a1 * Tc + a2 * (Tc**2) + a3 * (Tc**3) + a4 * (Tc**4)) * S + (a5 + a6 * Tc + a7 * (Tc**2)) * (S**1.5) + a8 * (S**2)

    # Water activity of seawater from Millero and Leung (1976)
    dlnawds = (-5.58651e-4 + 2.40452e-7 * Tc - 3.12165e-9 * (Tc**2) + 2.40808e-11 * (Tc**3)) + 1.5 * (1.79613e-5 - 9.9422e-8 * Tc + 2.08919e-9 * (Tc**2) - 1.39872e-11 * (Tc**3)) * (S**0.5) + 2 * (-2.31065e-6 - 1.37674e-9 * Tc - 1.93316e-11 * (Tc**2)) * S

    # Density derivative of refractive index from PMH model
    n_wat2 = nsw**2
    base = (nsw / 3 - (1.0 / 3.0) / nsw)
    DFRI = (n_wat2 - 1) * (1 + (2.0 / 3.0) * (n_wat2 + 2) * (base ** 2))

    # Volume scattering at 90 degrees due to density fluctuation
    beta_df = PI**2 / 2 * ((lambda_nm * 1e-9)**-4) * Kbz * Tk * IsoComp * (DFRI**2) * (6 + 6 * delta) / (6 - 7 * delta)

    # Volume scattering at 90 degrees due to concentration fluctuation
    flu_con = S * M0 * dnswds**2 / (density * -dlnawds) / Na
    beta_cf = 2 * PI**2 * ((lambda_nm * 1e-9)**-4) * (nsw**2) * flu_con * (6 + 6 * delta) / (6 - 7 * delta)

    # Total volume scattering at 90 degrees and total scattering coefficient
    beta90sw = beta_df + beta_cf
    bsw = 8 * PI / 3 * beta90sw * (2 + delta) / (1 + delta)

    return bsw

# -----------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------
# Main function for CAFE model's Net Primary Production (NPP) calculation
# -----------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------

def pycafe(PAR, chl, mld, lat_grid, aph_443, adg_443, bbp_443, bbp_s, sst, yd, lat_chunk, lon_chunk):
    """
    Computes daily net primary production following the Carbon, Absorption, Fluorescence Euphotic-resolving (CAFE) model.
    """

    # STEP 1: Define constants outside of the function to avoid redundant recalculation
    wavelengths = np.array([400, 410, 420, 430, 440, 450, 460, 470, 480, 490, 500, 510, 520, 530, 540, 550, 560, 570, 580, 590, 600, 610, 620, 630, 640, 650, 660, 670, 680, 690, 700])
    n_wavelengths = len(wavelengths)
    
    # Constants for absorption and scattering from the provided data
    aw = np.array([0.00663, 0.00473, 0.00454, 0.00495, 0.00635, 0.00922, 0.00979, 0.0106, 0.0127,
        0.015, 0.0204, 0.0325, 0.0409, 0.0434, 0.0474, 0.0565, 0.0619, 0.0695, 0.0896,
        0.1351, 0.2224, 0.2644, 0.2755, 0.2916, 0.3108, 0.34, 0.41, 0.439, 0.465, 0.516, 0.624])
    A_Bricaud = np.array([0.0241, 0.0287, 0.0328, 0.0359, 0.0378, 0.0350, 0.0328, 0.0309, 0.0281, 
        0.0254, 0.0210, 0.0162, 0.0126, 0.0103, 0.0085, 0.0070, 0.0057, 0.0050,
        0.0051, 0.0054, 0.0052, 0.0055, 0.0061, 0.0066, 0.0071, 0.0078, 0.0108, 
        0.0174, 0.0161, 0.0069, 0.0025])
    E_Bricaud = np.array([0.6877, 0.6834, 0.6664, 0.6478, 0.6266, 0.5993, 0.5961, 0.5970, 0.5890, 
        0.6074, 0.6529, 0.7212, 0.7939, 0.8500, 0.9036, 0.9312, 0.9345, 0.9298,
        0.8933, 0.8589, 0.8410, 0.8548, 0.8704, 0.8638, 0.8524, 0.8155, 0.8233,
        0.8138, 0.8284, 0.9255, 1.0286])
    PAR_spectrum = np.array([0.00227, 0.00218, 0.00239, 0.00189, 0.00297, 0.00348, 0.00345, 
                                 0.00344, 0.00373, 0.00377, 0.00362, 0.00364, 0.00360, 0.00367, 
                                 0.00354, 0.00368, 0.00354, 0.00357, 0.00363, 0.00332, 0.00358, 
                                 0.00357, 0.00359, 0.00340, 0.00350, 0.00332, 0.00342, 0.00347,
                             0.00342, 0.00290, 0.00314])


    # -----------------------------------------------------------------------------------------------
    # Step 2: Derive inherent optical properties (IOPs) at 10 nm increments from 400 to 700 nm
    # -----------------------------------------------------------------------------------------------

    # Define shapes for (lat, lon, wavelength) arrays
    grid_shape = (len(lat_chunk), len(lon_chunk), n_wavelengths)

    # Initialize 3D arrays for inherent optical properties
    aphi = np.zeros(grid_shape)
    a = np.zeros(grid_shape)
    bb = np.zeros(grid_shape)
    bbw = np.zeros(grid_shape)
    
    wavelengths = np.array(wavelengths)  # Convert wavelengths to NumPy array
    aw = np.array(aw)                    # Convert aw to NumPy array
    A_Bricaud = np.array(A_Bricaud)      # Convert A_Bricaud to NumPy array
    E_Bricaud = np.array(E_Bricaud)      # Convert E_Bricaud to NumPy array
    
    # Expand wavelength-related arrays for broadcasting
    wavelengths_expanded = wavelengths[np.newaxis, np.newaxis, :]  # Shape: (1, 1, n_wavelengths)
    aw_expanded = aw[np.newaxis, np.newaxis, :]                    # Shape: (1, 1, n_wavelengths)
    A_Bricaud_expanded = A_Bricaud[np.newaxis, np.newaxis, :]      # Shape: (1, 1, n_wavelengths)
    E_Bricaud_expanded = E_Bricaud[np.newaxis, np.newaxis, :]      # Shape: (1, 1, n_wavelengths)
    
    # Compute bbw (broadcast over wavelengths)
    bbw = betasw_ZHH2009(wavelengths_expanded, 32.5, sst.values[:,:,np.newaxis]) / 2  # Shape: (lat, lon, n_wavelengths)
    
    # Compute aphi
    aphi = (aph_443.values[:, :, np.newaxis] * A_Bricaud_expanded * (chl.values[:, :, np.newaxis] ** E_Bricaud_expanded) / (0.03711 * (chl.values[:, :, np.newaxis] ** 0.61479)))
    
    # Compute total absorption a
    a = (aw_expanded + aphi + adg_443.values[:, :, np.newaxis] * np.exp(-0.018 * (wavelengths_expanded - 443)))
    
    # Compute backscattering bb
    bb = (bbw + bbp_443.values[:, :, np.newaxis] * ((443 / wavelengths_expanded) ** bbp_s.values[:, :, np.newaxis]))

    # -----------------------------------------------------------------------------------------------
    # Step 3: Calculate the amount of energy absorbed by phytoplankton in the euphotic zone
    # -----------------------------------------------------------------------------------------------

    # Vectorized calculation of absorbed_photons
    absorbed_photons = 0.5 * 10 * np.sum(
        (PAR_spectrum[1:] * aphi[:, :, 1:] / a[:, :, 1:]) +
        (PAR_spectrum[:-1] * aphi[:, :, :-1] / a[:, :, :-1]),
        axis=2) # Sum over the wavelength dimension
    
    # Final scaling factor applied to the entire 2D absorbed_photons array
    absorbed_photons *= PAR.values * 0.95

    # -----------------------------------------------------------------------------------------------
    # Step 4: Derive spectral attenuation coefficient (kd) and attenuation coefficient of PAR (kdpar)
    # -----------------------------------------------------------------------------------------------

    #Transform latitudinal degrees to radians
    lat_rad = np.radians(lat_grid)

    # Compute declination angle (remains scalar as it depends only on day of the year)
    decl = 23.5 * np.cos(2 * PI * (yd - 172) / 365) * PI / 180
    
    # Daylength (DL) calculation based on latitude
    DL = -1 * np.tan(lat_rad) * np.tan(decl)
    DL = np.clip(DL, -1, 1)
    DL = np.arccos(DL) / PI  # Daylength in days for each latitude
    
    solzen = 90 - np.degrees(np.arcsin(np.sin(lat_rad) * np.sin(decl) - np.cos(lat_rad) * np.cos(decl) * np.cos(PI)))
    
    # Adjusted optical path length (m0) as a 2D array
    m0 = np.sqrt((1 + 0.005 * solzen) ** 2)  # absolute value
    
    # Vectorized calculation of kd
    kd = m0[..., None] * a + 4.18 * (1 - 0.52 * np.exp(-10.8 * a)) * bb
    
    # Calculate kdpar as a 2D array for (lat, lon) using the kd value at the 9th wavelength index (around 490 nm)
    kdpar = 0.0665 + (0.874 * kd[:, :, 9]) - (0.00121 / kd[:, :, 9])
    
    # Calculate the euphotic depth (zeu) as a 2D array
    zeu = -1 * np.log(0.1 / (PAR * 0.95)) / kdpar

    # -----------------------------------------------------------------------------------------------
    # Step 5: Derive conversion factor (Eu) to convert downwelling planar irradiance to scalar irradiance
    # -----------------------------------------------------------------------------------------------

    # Initialize time and depth sequences
    tseq = np.linspace(0, 1, 51)  # Fractional time of day for 51 time steps
    zseq = np.linspace(0, 1, 101) * np.ceil(zeu.values[:,:,None]) # Depth sequence divided into 101 increments
    delz= zseq[:,:,1] - zseq[:,:,0]
    
    # Vectorized computation of PAR_noon
    PAR_noon = (PI / 2) * PAR.values[:, :, None] * 0.95 * PAR_spectrum  # Shape: (len(lat_chunk), len(lon_chunk), 31)
    
    # Initialize 4D arrays for E_tzw and AP_tzw with shape
    E_tzw = np.zeros((len(lat_chunk), len(lon_chunk), 51, 101, 31))
    AP_tzw = np.zeros((len(lat_chunk), len(lon_chunk), 51, 101, 31))
    
    # Precompute for E_tzw
    sin_t = np.sin(PI * tseq)
    exp_zw = np.exp(-1* kd[:, :, None, None, :] * zseq[:, :, None,:, None])  # Shape: (lat, lon, depth, 1, wavelength)
    
    # Compute E_tzw (Irradiance through time, depth and wavelength)
    E_tzw = PAR_noon[:, :, None, None, :] * sin_t[None, None, :, None, None] * exp_zw  # Shape: (lat, lon, time, depth, wavelength)
    
    # Compute AP_tzw (Absorbed Photons through time, depth and wavelength))
    AP_tzw = E_tzw * aphi[:, :, None, None, :]
    
    # Integrate E and AP through wavelengths (using trapezoidal integration)
    E_tz = np.zeros((len(lat_chunk), len(lon_chunk), 51, 101))
    AP_tz = np.zeros((len(lat_chunk), len(lon_chunk), 51, 101))
    
    E_tz = np.trapz(E_tzw, dx=10, axis=4) #across wavelength dimension
    AP_tz = np.trapz(AP_tzw, dx=10, axis=4)
    
    # Integrate AP through time (using trapezoidal integration)
    AP_z = np.zeros((len(lat_chunk), len(lon_chunk), 101))
    AP_z = np.trapz(AP_tz, dx=0.02, axis=2) #across time dimension
    
    # Broadcast `delz` to match the shape of `AP_z` for depth integration
    delz_broadcasted = delz[:,:,np.newaxis]  # Add an extra axis to make it (lat, lon, depth)
    
    # Integrate AP through depth to get total absorbed photons `AP`
    AP = np.trapz(AP_z, dx=delz_broadcasted, axis=2)  # across depth dimension
    
    # Derive Eu across the grid
    Eu = absorbed_photons / AP  # Shape (len(lat_chunk), len(lon_chunk))
    
    # Apply the operation across all time and depth dimensions
    E_tz *= Eu[:, :, None, None] # Shape: (lat, lon, time, depth)
    AP_tz *= Eu[:, :, None, None]  # Shape: (lat, lon, time, depth)

    # -----------------------------------------------------------------------------------------------
    # Step 6: Calculate Ek through depth
    # -----------------------------------------------------------------------------------------------
 
    # Pre-compute constants and initializations
    IML1 = (PAR.values * 0.95 / (DL * 24))  # Shape: (len(lat_chunk), len(lon_chunk))
    IML2 = np.exp(-0.5 * kdpar * mld.values)  # Shape: (len(lat_chunk), len(lon_chunk))
    
    # Compute the initial irradiance term IML as a 2D array (len(lat_chunk), len(lon_chunk))
    IML = IML1 * IML2  # Shape: (len(lat_chunk), len(lon_chunk))
    
    # Compute the initial Ek for all depths (vectorized over all depths)
    Ek = 19 * np.exp(0.038 * (IML1 ** 0.45) / kdpar)[:, :, None]  # Initial Ek for all depths
    Ek = np.broadcast_to(Ek, (len(lat_chunk), len(lon_chunk), 101))
    Ek_ini = np.maximum(Ek, 10)  # Ensure Ek >= 10 for all depths
    
    # Modify Ek beneath the MLD (vectorized over all depths)
    mask_mld_zeu = mld.values < zeu.values  # Mask where MLD < Zeu, shape (lat, lon)
    
    # Compute Eg_mld, which has the same shape as mld
    Eg_mld = (PAR.values / DL) * np.exp(-kdpar * mld.values)  # Shape: (lat, lon)
    
    # Compute the factor for each depth (vectorized)
    factor = (1 + np.exp(-0.15 * IML1)) / (1 + np.exp(-3 * IML))  # Shape: (lat, lon)
    
    # Use broadcasting directly without flattening
    mask_mld_zeu_3d = np.broadcast_to(mask_mld_zeu[:, :, None], Ek.shape)
    Ek_zeu = np.copy(Ek_ini)
    Ek_zeu = np.where(mask_mld_zeu_3d, Ek_zeu * factor[:,:,None], Ek_ini)  # Apply the factor only where the mask is True
    
    # Create a mask for when zseq > mld, vectorized over depth (this gives a 3D mask)
    mask_zseq_mld = zseq > mld.values[:, :, None]  # Shape (lat, lon, 101)
    mask_combined = mask_mld_zeu[:,:,None] & mask_zseq_mld[:,:,:] # Shape (lat, lon, depth)
    
    # Compute Eg for each depth (vectorized over depth)
    Eg = (PAR.values[:, :, None] / DL[:, :, None]) * np.exp(-1*kdpar[:, :, None] * zseq)  # Shape: (lat, lon, 101)
    
    # Apply the condition where zseq > mld and adjust Ek
    Ek = np.where(mask_combined, 
                   10 + (Ek_zeu - 10) / (Eg_mld[:, :, None] - 0.1) * (Eg - 0.1), 
                   Ek_zeu)  # Modify Ek where zseq > mld
    
    # Ensure Ek remains above 10
    Ek = np.maximum(Ek, 10)
    
    # Convert all Ek to mol photons/m2/day
    Ek *= 0.0864  # Apply the conversion to mol photons/m2/day

    # -----------------------------------------------------------------------------------------------
    # Step 7: Make Ek spectrally explicit (KPUR)
    # -----------------------------------------------------------------------------------------------

    # Calculate mean aphi over all wavelengths (31 wavelengths)
    mean_aphi = np.mean(aphi, axis=2)  # Shape: (len(lat_chunk), len(lon_chunk))
    
    # Compute numerator and denominator for all depths and wavelengths
    AP_avg = 10 * 0.5 * (AP_tzw[:, :, 24, :, :-1] + AP_tzw[:, :, 24, :, 1:])  # Shape: (len(lat_chunk), len(lon_chunk), 101, 30)
    E_avg = 10 * 0.5 * (E_tzw[:, :, 24, :, :-1] + E_tzw[:, :, 24, :, 1:])    # Shape: (len(lat_chunk), len(lon_chunk), 101, 30)
    
    # Sum over wavelengths
    numerator = np.sum(AP_avg, axis=3)  # Shape: (len(lat_chunk), len(lon_chunk), 101)
    denominator = np.sum(E_avg, axis=3)  # Shape: (len(lat_chunk), len(lon_chunk), 101)
    
    # Compute KPUR for all depths
    KPUR = Ek / ((numerator / (denominator * mean_aphi[:, :, None])) / 1.3)  # Shape: (len(lat_chunk), len(lon_chunk), 101)

    # -----------------------------------------------------------------------------------------------
    # Step 8: Calculate phimax as a function of Ek
    # -----------------------------------------------------------------------------------------------
    
    # Range of phimax values from C code
    phirange = [0.018, 0.030]  # Range of phimax values
    
    # Range of Ek values (scaled)
    Ekrange = [150 * 0.0864, 10 * 0.0864]  # Relates phimax to Ek
    
    # Compute the slope
    slope = (phirange[1] - phirange[0]) / (Ekrange[1] - Ekrange[0])
    
    # Vectorized computation of phimax across all depths
    phimax = phirange[1] + (Ek - Ekrange[1]) * slope  # Shape: (len(lat_chunk), len(lon_chunk), 101)
    
    # Apply range constraints
    phimax = np.clip(phimax, phirange[0], phirange[1])  # Ensures phimax is within [phirange[0], phirange[1]]

    # -----------------------------------------------------------------------------------------------
    # Step 9: Scale aphi beneath the MLD to Ek
    # -----------------------------------------------------------------------------------------------
    
    # Initialize aphi_fact with ones holaa
    aphi_fact = np.ones((len(lat_chunk), len(lon_chunk), 101))
    
    # Update aphi_fact where the condition is true
    aphi_fact = np.where(mask_combined, 1 + Ek[:, :, 0][:, :, None] / Ek * 0.15, 1.0)
    
    # Modify Irradiance and absorbed energy to account for any change to aphi
    modified_a = (aw_expanded[:, :, None, :] + aphi[:, :, None, :] * aphi_fact[:, :, :, None] 
                  + adg_443.values[:, :, None, None] * np.exp(-0.018 * (wavelengths_expanded[:, :, None, :] - 443)))
    
    # Calculate `kd_modified` (attenuation coefficient), which is based on the modified absorption coefficients
    kd_modified = m0[:, :, None, None] * modified_a + 4.18 * (1 - 0.52 * np.exp(-10.8 * modified_a)) * bb[:, :, None, :]
    
    # Precompute the exponential decay factor for all z levels
    exp_tzw = np.exp(-kd_modified * delz[:, :, None, None])
    
    # Use broadcasting to compute E_tzw for all z levels at once
    E_tzw[:, :, :, 1:, :] = E_tzw[:, :, :, 0, None, :] * np.cumprod(exp_tzw[:, :, None, 1:, :], axis=3)
    
    # Initialize AP_tzw with the first layer
    AP_tzw[:, :, :, 0, :] = E_tzw[:, :, :, 0, :] * aphi[:, :, None, :]
    
    # Update the absorbed energy (AP_tzw) based on the modified irradiance
    AP_tzw = E_tzw * aphi[:, :, None, None, :] * aphi_fact[:, :, None, :, None]
    
    # Integrate over wavelengths using trapezoidal rule
    AP_tz2 = np.zeros((len(lat_chunk), len(lon_chunk), 51, 101))
    AP_tz2 = np.trapz(AP_tzw, wavelengths, axis=-1)
    AP_tz2 *= Eu[:,:,None, None]
    
    # Handle case where MLD >= Zeu (use the value directly from AP_tz if MLD >= Zeu)
    AP_tz2[~mask_mld_zeu] = AP_tz[~mask_mld_zeu]

    # -----------------------------------------------------------------------------------------------
    # Step 10: Calculate NPP
    # -----------------------------------------------------------------------------------------------
    # Net primary production (NPP) without loops
    # Create a mask for zero values in E_tz to avoid division by zero
    E_tz_zero_mask = E_tz == 0  # Shape: (lat_chunk, lon_chunk, 51, 101)
    
    # Safely compute NPP_tz
    with np.errstate(divide='ignore', invalid='ignore'):  # Handle division warnings
        NPP_tz = np.where(
            E_tz_zero_mask,
            0,  # If E_tz is zero, set NPP_tz to zero
            phimax[:, :, None, :] * AP_tz2 * np.tanh(KPUR[:, :, None, :] / E_tz) * 12000) # Shape: (lat_chunk, lon_chunk, 51, 101)
    
    # Integrate NPP through time
    # Using trapezoidal rule for time integration (50 intervals)
    NPP_z = np.trapz(NPP_tz, dx=0.02, axis=2)  # Shape: (lat_chunk, lon_chunk, 101)
    
    # Integrate NPP through depth
    # Using trapezoidal rule for depth integration (100 intervals)
    NPP = np.trapz(NPP_z, dx=delz[:, :, None], axis=2)  # Shape: (lat_chunk, lon_chunk)

    ds = xr.Dataset({"NPP": (("lat", "lon"), NPP),},
        coords={"lat": lat_chunk, "lon": lon_chunk},)
    
    return ds

