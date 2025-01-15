# PYCAFE

'PyCAFE' model computes monthly net primary production following the Carbon, Absorption, Fluorescence Euphotic-resolving (CAFE) model. It was rewritten in Python to perform computations across latitude and longitude chunks for efficient processing of spatial data.

- For a full description of the original model, please see:
  Silsbe, G.M., M.J. Behrenfeld, K.H. Halsey, A.J. Milligan, and T.K. Westberry. 2016.  
  The CAFE model. A net production model for global ocean phytoplankton. Global Biogeochemical Cycles. doi: 10.1002/2016GB005521.

## Model Inputs:
- **PAR** (shape: lat, lon)     : Daily photosynthetic active radiation [mol photons m^-2 day^-1].
- **chl** (shape: lat, lon)     : Chlorophyll concentration, NASA OCI algorithm [mg m^-3].
- **mld** (shape: lat, lon)     : Mixed layer depth [m].
- **lat_grid** (shape: lat, lon): Latitude [degrees, north positive].
- **yd** (int)  : Day of the year.
- **aph_443** (shape: lat, lon) : Absorption due to phytoplankton at 443 nm, NASA GIOP algorithm [m^-1].
- **adg_443** (shape: lat, lon) : Absorption due to gelbstoff and detrital material at 443 nm, NASA GIOP model [m^-1].
- **bbp_443** (shape: lat, lon) : Particulate backscatter at 443 nm, NASA GIOP model [m^-1].
- **bbp_s** (shape: lat, lon)   : Backscattering spectral parameter for GIOP model [dimensionless].
- **sst** (shape: lat, lon)     : Sea surface temperature [degrees Celsius]. Only used to compute backscattering of pure seawater in function betasw_ZHH2009.

## Modeling Steps:
1. Declare all local variables.
2. Derive inherent optical properties (IOPs) at 10 nm increments from 400 to 700 nm.
3. Calculate the amount of energy in the euphotic zone absorbed by phytoplankton.
4. Derive the spectral attenuation coefficient and the attenuation coefficient of PAR.
5. Derive conversion factor (Eu) that converts downwelling planar irradiance to scalar irradiance.
6. Calculate the photoacclimation parameter (Ek) through depth.
7. Scale Ek to a spectrally explicit parameter (KPUR).
8. Model the maximum quantum efficiency of net carbon fixation (phi_max).
9. If the mixed layer depth (MLD) is shallower than the euphotic depth, apply a scaling factor to phytoplankton absorption beneath the MLD and adjust the energy absorbed by phytoplankton.
10. Derive net phytoplankton production (NPP).

## Defined Constants and Parameters:
- **Wavelengths** (shape: 31): Range from 400 to 700 nm in 10 nm increments.
- **Absorption due to pure water (aw)**: Predefined constants for absorption at each wavelength.
- **Spectral shape of phytoplankton absorption (A_Bricaud and E_Bricaud)**: Predefined arrays.
- **Spectral shape of PAR (PAR_spectrum)**: Predefined distribution of PAR across wavelengths.

## Computed Variables and Parameters:
- **aphi** : Absorption due to phytoplankton. (shape: lat, lon, wavelength)
- **bbw**  : Backscattering of pure water. (shape: lat, lon, wavelength)
- **a**    : Total absorption. (shape: lat, lon, wavelength)
- **bb**   : Total backscattering. (shape: lat, lon, wavelength)

- **decl** : Solar declination [radians]. 
- **DL**   : Day length [fraction of day]. (shape: lat, lon)
- **solzen**: Solar zenith angle [degrees]. (shape: lat, lon)
- **m0**   : Coefficient to calculate kd (Lee et al. 2005) [m^-1].
- **kd**   : Downwelling attenuation coefficient of irradiance [m^-1]. (shape: lat, lon, wavelength)
- **kdpar**: Downwelling attenuation coefficient of PAR [m^-1]. (shape: lat, lon)
- **zeu**  : Euphotic depth [m]. (shape: lat, lon)

- **tseq** : Fractional time of day - 51 increments. (shape: time) 
- **zseq** : Euphotic depth divided into 101 increments. (shape: lat, lon, depth)
- **delz** : Depth of zseq. (shape: lat, lon)
- **absorbed_photons** : Energy absorbed by phytoplankton [mol photons m^-2 day^-1]. (shape: lat, lon)
- **PAR_noon** : PAR at solar noon [mol photons m^-2 day^-1]. (shape: lat, lon, wavelength)
- **E_tzw** : Irradiance through time, depth and wavelength. (shape: lat, lon, time, depth, wavelength)
- **A_tzw** : Absorbed photons through time, depth and wavelength. (shape: lat, lon, time, depth, wavelength)
- **E_tz** : Irradiance through time and depth. (shape: lat, lon, time, depth)
- **A_tz** : Absorbed photons through time and depth. (shape: lat, lon, time, depth)
- **AP_z** : Absorbed photons through depth. (shape: lat, lon, depth)
- **AP** : Absorbed photons [mol m^-2 day^-1]. (shape: lat, lon)
- **Eu** : Conversion factor that converts downwelling planar irradiance to scalar irradiance. (shape: lat, lon)

- **IML** : Median irradiance in the mixed layer [umol m^-2 day^-1]. (shape: lat, lon)
- **Ek** : Photoacclimation parameter through depth [mol photons m^-2 day^-1] (Behrenfeld et al. 2016). (shape: lat, lon, depth)
- **Eg_mld** : Growth irradiance when mixed layer depth (mld) < euphotic depth (zeu). (shape: lat, lon, depth)
- **Eg** : Growth irradiance when mld < zeu & zseq > mld. (shape: lat, lon, depth)

- **mean_aphi** : Spectrally averaged absorption due to phytoplankton. (shape: lat, lon)
- **KPUR** : Spectrally explicit Ek. (shape: lat, lon, depth)

- **phimax** : Maximum quantum yield of net carbon fixation [mol C (mol photons)^-1].
- **phirange** : Range of phimax values. [0.018, 0.030]
- **Ekrange** : Relates phimax to Ek. [150*0.086400, 10*0.0864]
- **slope** : Relates phimax to Ek.

- **aphi_fact** : Parameter to increase aphi beneath the mixed layer depth (MLD).

- **NPP** : Net primary production [mg C m^-2 day^-1]. (shape: lat, lon)

**Note**:  
For further understanding of individual calculations and steps, refer to the provided reference.
