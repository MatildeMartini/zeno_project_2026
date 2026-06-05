import math

# Earth's radius in meters
R = 6378137.0

# WGS-84 flattening
f = 1.0 / 298.257223563  

def get_ellipsoid_radii(lat_rad):
    """
    Compute the radius of curvature in the prime vertical and meridian
    at a given latitude using the WGS-84 ellipsoid parameters.
    Args:
        lat_rad (float): Latitude in radians.
    Returns:
        tuple: Radius of curvature in the prime vertical (Rn) and meridian (Rm).
    """

    # Compute the radius of curvature in the prime vertical and meridian
    # using the WGS-84 ellipsoid parameters
    Rn = R / math.sqrt(1 - f * (2 - f) * math.sin(lat_rad)**2)
    Rm = Rn * (1 - f * (2 - f)) / (1 - (2 * f - f * f) * math.sin(lat_rad)**2)

    return Rn, Rm

def ll2ne(ll0, ll):
    """Convert latitude and longitude to northing and easting.

    Args:
        ll0 (tuple): Latitude and longitude of the origin point.
        ll (tuple): Latitude and longitude of the point to convert.

    Returns:
        tuple: Northing and easting coordinates.
    """
    if len(ll0) != 2 or len(ll) != 2:
        raise ValueError("Input coordinates must be tuples of length 2.")

    lat0, lon0  = ll0
    lat, lon    = ll

    # Convert degrees to radians
    lat0_rad    = math.radians(lat0)
    lon0_rad    = math.radians(lon0)
    lat_rad     = math.radians(lat)
    lon_rad     = math.radians(lon)

    # Latitude and longitude differences
    dlat = lat_rad - lat0_rad
    dlon = lon_rad - lon0_rad

    # Calculate the radius of curvature in the prime vertical and meridian
    # using the WGS-84 ellipsoid parameters
    Rn, Rm = get_ellipsoid_radii(lat0_rad)

    # Calculate the northing and easting using the radii of curvature
    # and the latitude and longitude differences
    n = dlat / math.atan2(1, Rm)
    e = dlon / math.atan2(1, Rn * math.cos(lat0_rad))
    ne = [n, e]

    return ne

def ne2ll(ll0, ne):
    """Convert northing and easting to latitude and longitude.
    Args:
        ll0 (tuple): Latitude and longitude of the origin point.
        ne (tuple): Northing and easting coordinates to convert.
    Returns:
        tuple: Latitude and longitude coordinates.
    """

    if len(ll0) != 2 or len(ne) != 2:
        raise ValueError("Input coordinates must be tuples of length 2.")
    
    lat0, lon0  = ll0
    lat0 = math.radians(lat0)
    lon0 = math.radians(lon0)
    
    Rn, Rm = get_ellipsoid_radii(lat0)

    lat = (lat0 + ne[0] * math.atan2(1, Rm)) * 180 / math.pi
    lon = (lon0 + ne[1] * math.atan2(1, Rn * math.cos(lat0))) * 180 / math.pi

    ll = [lat, lon]

    return ll

def lld2ned(ll0d, lld):
    """Convert latitude, longitude, and depth to northing, easting, and depth.
    Args:
        ll0d (tuple): Latitude, longitude, and depth of the origin point.
        lld (tuple): Latitude, longitude, and depth of the point to convert.
    Returns:
        tuple: Northing, easting, and depth coordinates.
    """

    if len(ll0d) != 3 or len(lld) != 3:
        raise ValueError("Input coordinates must be tuples of length 3.")
    ne = ll2ne(ll0d[:2], lld[:2])
    d = lld[2] - ll0d[2]
    ned = [ne[0], ne[1], d]

    return ned

def ned2lld(ll0d, ned):
    """Convert northing, easting, and depth to latitude, longitude, and depth.
    Args:
        ll0d (tuple): Latitude, longitude, and depth of the origin point.
        ned (tuple): Northing, easting, and depth coordinates to convert.
    Returns:
        tuple: Latitude, longitude, and depth coordinates.
    """

    if len(ll0d) != 3 or len(ned) != 3:
        raise ValueError("Input coordinates must be tuples of length 3.")
    ll = ne2ll(ll0d[:2], ned[:2])
    d = ll0d[2] + ned[2]
    lld = [ll[0], ll[1], d]

    return lld

def lld2direction(lld1, lld2):
    """Calculate the 2D direction from one point to another defined in 3D space.
    Args:
        lld1 (tuple): Latitude, longitude, and depth of the first point.
        lld2 (tuple): Latitude, longitude, and depth of the second point.
    Returns:
        float: Direction in radians w.r.t the North (no depth involved)
    """
    if len(lld1) != 3 or len(lld2) != 3:
        raise ValueError("Input coordinates must be tuples of length 3.")
    
    tmp_ned = lld2ned(lld1, lld2)
    direction = math.atan2(tmp_ned[1], tmp_ned[0])

    return direction

def ll2direction(ll1, ll2):
    """Calculate the 2D direction from one point to another defined in 2D space.
    Args:
        ll1 (tuple): Latitude and longitude of the first point.
        ll2 (tuple): Latitude and longitude of the second point.
    Returns:
        float: Direction in radians w.r.t the North (no depth involved)
    """
    if len(ll1) != 2 or len(ll2) != 2:
        raise ValueError("Input coordinates must be tuples of length 2.")
    
    tmp_ned = ll2ne(ll1, ll2)
    direction = math.atan2(tmp_ned[1], tmp_ned[0])

    return direction

def lld2distance(ll1, ll2):
    """Calculate the 3D distance from one point to another defined in 3D space.
    Args:
        ll1 (tuple): Latitude, longitude, and depth of the first point.
        ll2 (tuple): Latitude, longitude, and depth of the second point.
    Returns:
        float: Distance in meters (depth involved).
    """

    if len(ll1) != 3 or len(ll2) != 3:
        raise ValueError("Input coordinates must be tuples of length 3.")
    
    tmp_ned = lld2ned(ll1, ll2)
    distance = math.sqrt(tmp_ned[0]**2 + tmp_ned[1]**2 + tmp_ned[2]**2)
    return distance

def ll2distance(ll1, ll2):
    """Calculate the 2D distance from one point to another defined in 2D space.
    Args:
        ll1 (tuple): Latitude and longitude of the first point.
        ll2 (tuple): Latitude and longitude of the second point.
    Returns:
        float: Distance in meters.
    """

    if len(ll1) != 2 or len(ll2) != 2:
        raise ValueError("Input coordinates must be tuples of length 2.")
    
    tmp_ned = ll2ne(ll1, ll2)
    distance = math.sqrt(tmp_ned[0]**2 + tmp_ned[1]**2)
    return distance