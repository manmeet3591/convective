import numpy as np
import xarray as xr

def compute_srh(ua, va, plev, ps, storm_motion=None, z_levels=None):
    """
    Compute SRH (Storm-Relative Helicity) from CMIP6-style 6hrLev data.
    
    Parameters:
    ua (xr.DataArray): Zonal wind (m/s), dimensions [time, lev, lat, lon]
    va (xr.DataArray): Meridional wind (m/s), dimensions [time, lev, lat, lon]
    plev (xr.DataArray): Pressure levels (Pa), 1D array
    ps (xr.DataArray): Surface pressure (Pa), dimensions [time, lat, lon]
    storm_motion (tuple): (u_c, v_c), storm motion components (m/s). If None, Bunkers method is used.
    z_levels (list): Optional, heights corresponding to pressure levels (m). If None, standard atmosphere is assumed.

    Returns:
    srh01 (xr.DataArray): 0–1 km SRH (m²/s²)
    srh03 (xr.DataArray): 0–3 km SRH (m²/s²)
    """
    
    # Constants
    Rd = 287.05  # J/(kg*K)
    g0 = 9.80665  # m/s²
    T0 = 288.15  # K (assume standard)
    
    # If z_levels are not provided, estimate height (hydrostatic approx)
    if z_levels is None:
        z_levels = (Rd * T0 / g0) * np.log(ps.values[..., np.newaxis] / plev.values)
    
    # Compute mean winds for storm motion if needed
    if storm_motion is None:
        # Mean wind from surface to 6 km approximation (~500 hPa)
        u_mean = ua.sel(lev=slice(plev.max(), 50000)).mean(dim="lev")
        v_mean = va.sel(lev=slice(plev.max(), 50000)).mean(dim="lev")
        
        # Shear vector (surface - 6km)
        u_sfc = ua.sel(lev=plev.max(), method="nearest")
        v_sfc = va.sel(lev=plev.max(), method="nearest")
        u_6km = ua.sel(lev=50000, method="nearest")
        v_6km = va.sel(lev=50000, method="nearest")
        
        delta_u = u_6km - u_sfc
        delta_v = v_6km - v_sfc
        mag_shear = np.sqrt(delta_u**2 + delta_v**2) + 1e-8  # avoid division by zero
        
        # Right-moving supercell storm motion (Bunkers method)
        u_c = u_mean + 7.5 * (-delta_v / mag_shear)
        v_c = v_mean + 7.5 * (delta_u / mag_shear)
    else:
        u_c, v_c = storm_motion
    
    # Now calculate SRH
    def integrate_srh(z_bottom, z_top):
        """
        Helper to integrate SRH between z_bottom and z_top (e.g., 0–1 km or 0–3 km).
        """
        srh = xr.zeros_like(ps)
        
        # Loop over levels
        for k in range(len(plev) - 1):
            # Mid-layer height
            z1 = z_levels[..., k]
            z2 = z_levels[..., k+1]
            dz = np.abs(z2 - z1)
            
            mid_z = (z1 + z2) / 2
            
            # Only consider layers within desired height range
            mask = (mid_z >= z_bottom) & (mid_z <= z_top)
            
            if not np.any(mask):
                continue
            
            # Wind components at levels
            u1 = ua.isel(lev=k)
            u2 = ua.isel(lev=k+1)
            v1 = va.isel(lev=k)
            v2 = va.isel(lev=k+1)
            
            du = u2 - u1
            dv = v2 - v1
            
            # Streamwise vorticity contribution
            srh_layer = ( (u1 - u_c) * dv - (v1 - v_c) * du ) * dz
            srh = srh.where(~mask, srh + srh_layer)
        
        return srh
    
    # Compute SRH between 0–1 km and 0–3 km
    srh01 = integrate_srh(0, 1000)
    srh03 = integrate_srh(0, 3000)
    
    return srh01, srh03
# # Compute SRH
# srh01, srh03 = compute_srh(ua, va, plev, ps)
