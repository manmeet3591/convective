import metpy.calc as mpcalc
from metpy.units import units
import numpy as np

# Assume data_at_point is already selected as you did
ta = data_at_point['Temperature_isobaric']  # should have units attached
hus = data_at_point['Specific_humidity_isobaric']  # specific humidity (dimensionless or kg/kg)
p = data_at_point['isobaric3']  # pressure levels

# 1. Compute mixing ratio w
w = hus / (1 - hus)

# 2. Compute vapor pressure e
e = (w * p) / (0.622 + w)  # e in same units as p

# 3. Compute saturation vapor pressure es
# Important: temperature must be in degrees Celsius
ta_celsius = ta.to('degC')

es = 6.112 * np.exp((17.67 * ta_celsius) / (ta_celsius + 243.5)) * units.hPa

# 4. Compute relative humidity RH
RH = e / es  # RH as fraction

# 5. Dewpoint
dewpoint = mpcalc.dewpoint_from_relative_humidity(ta, RH)

# 6. CAPE and CIN
cape, cin = mpcalc.surface_based_cape_cin(
    p[::-1],
    ta[::-1],
    dewpoint[::-1]
)
cape
