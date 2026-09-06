import warnings

import pandas as pd
import numpy as np
import geopandas as gpd
import h3
from shapely.geometry import Polygon, MultiPolygon
from sklearn.neighbors import NearestNeighbors
import networkx as nx
import folium

from . import validate as v

################
## H3 version-compatibility shims ##
################
# h3-py's public API changed names across major versions (v3 -> v4). These
# small wrappers try the current API first and fall back for older
# installs, so hexseg works across the range declared in pyproject.toml
# rather than only against whatever version happened to be installed
# when this was last tested.


def _h3_polyfill(geo_interface, resolution):
    if hasattr(h3, "geo_to_cells"):
        return set(h3.geo_to_cells(geo_interface, resolution))
    if hasattr(h3, "polygon_to_cells"):
        try:
            return set(h3.polygon_to_cells(geo_interface, resolution, geo_json_conformant=True))
        except TypeError:
            # h3-py v4's polygon_to_cells takes a LatLngPoly/LatLngMultiPoly,
            # not a raw __geo_interface__ dict, and has no
            # geo_json_conformant kwarg.
            coords = geo_interface["coordinates"]
            rings = [
                [(lat, lng) for lng, lat in ring]
                for ring in coords
            ]
            poly = h3.LatLngPoly(rings[0], *rings[1:])
            return set(h3.polygon_to_cells(poly, resolution))
    return set(h3.polyfill(geo_interface, resolution, geo_json_conformant=True))


def _h3_cell_boundary(cell):
    if hasattr(h3, "cell_to_boundary"):
        return h3.cell_to_boundary(cell)
    return h3.h3_to_geo_boundary(cell)


def _h3_grid_disk(cell, k):
    if hasattr(h3, "grid_disk"):
        return set(h3.grid_disk(cell, k))
    return set(h3.k_ring(cell, k))


################
## Function 1 ##
################

def get_hexagons(gdf_polygons: gpd.GeoDataFrame,
                  name_col: str,
                  resolution: int = 9) -> gpd.GeoDataFrame:
    """
    For each polygon in `gdf_polygons`, generate all H3 hexagons at `resolution`,
    then assign each hex to the polygon with which it has the largest intersection.
    Returns a GeoDataFrame with columns ['hex_id', 'geo_boundary', 'geometry']
    in the same CRS as `gdf_polygons`.

    Parameters:
    ----------
    gdf_polygons : GeoDataFrame
        Polygon or MultiPolygon study-area boundaries.
    name_col : str
        Police force, district or other geography name.
    resolution : int
        Uber hexagon resolution, see: https://h3geo.org/docs/3.x/core-library/restable/

    Example:
    --------
    hexes = get_hexagons(gdf_districts, name_col="lad21nm", resolution=9)
    """
    v.require_geom_type(gdf_polygons, {"Polygon", "MultiPolygon"}, "gdf_polygons")
    v.require_crs(gdf_polygons, "gdf_polygons")
    v.require_column(gdf_polygons, name_col, "gdf_polygons")

    # Validate and reproject to WGS84 for H3
    wgs = gdf_polygons.to_crs("EPSG:4326")

    # Real-world boundary data (especially anything hand-digitised or
    # simplified) is often topologically invalid -- self-intersections,
    # bowties, etc. h3 either errors or silently mis-tiles on those. Repair
    # rather than fail, but tell the caller it happened.
    invalid_mask = ~wgs.geometry.is_valid
    if invalid_mask.any():
        n_invalid = int(invalid_mask.sum())
        warnings.warn(
            f"{n_invalid} of {len(wgs)} polygon(s) in gdf_polygons were "
            f"topologically invalid (e.g. self-intersecting) and were "
            f"repaired automatically before tiling.",
            RuntimeWarning,
        )
        geom_col = wgs.geometry.name
        if hasattr(wgs.geometry, "make_valid"):
            wgs.loc[invalid_mask, geom_col] = wgs.loc[invalid_mask].geometry.make_valid()
        else:
            wgs.loc[invalid_mask, geom_col] = wgs.loc[invalid_mask].geometry.buffer(0)

    records = []
    # Loop over features, handle Polygons & MultiPolygons
    for _, row in wgs.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        boundary_name = row[name_col]
        parts = geom.geoms if isinstance(geom, MultiPolygon) else [geom]

        # collect all hex IDs
        hex_ids = set()
        for part in parts:
            hex_ids.update(_h3_polyfill(part.__geo_interface__, resolution))

        for h in hex_ids:
            # get the hex boundary as a list of (lat, lon)
            coords = _h3_cell_boundary(h)
            # flip to (x, y) = (lon, lat)
            pts = [(lng, lat) for lat, lng in coords]
            records.append((h, boundary_name, Polygon(pts)))

    if not records:
        raise ValueError(
            "No H3 hexagons were generated for any polygon in gdf_polygons. "
            "Check that the geometries are non-empty and that `resolution` "
            "is fine enough relative to their size."
        )

    # Assemble and reproject back to the original CRS
    out = (
        gpd.GeoDataFrame(
            records,
            columns=["hex_id", "geo_boundary", "geometry"],
            crs="EPSG:4326"
        )
        .to_crs(gdf_polygons.crs)
    )

    # Spatial‐join purely to get index_right for overlap area
    join_polys = gdf_polygons[["geometry"]]
    joined = gpd.sjoin(out, join_polys, how="left", predicate="intersects")

    # Map back to original polygon geometry and compute overlap
    joined["poly_geom"] = joined["index_right"].map(gdf_polygons.geometry)
    joined["overlap"] = joined.geometry.intersection(joined["poly_geom"]).area

    # Pick the best overlap per hex_id. Ties (a hex split exactly evenly
    # between two districts) are broken arbitrarily by sort stability --
    # rare in practice, but worth knowing about if a hex ends up assigned
    # to a district you didn't expect right on a shared boundary.
    best = (
        joined
          .sort_values("overlap", ascending=False)
          .drop_duplicates("hex_id")
          .loc[:, ["hex_id", "geo_boundary", "geometry"]]
          .reset_index(drop=True)
    )

    # Return as GeoDataFrame in original CRS
    return gpd.GeoDataFrame(best, geometry="geometry", crs=gdf_polygons.crs)

################
## Function 2 ##
################

def summarise_by_hex(hexes_gdf: gpd.GeoDataFrame,
                      crimes_gdf: gpd.GeoDataFrame,
                      count: bool = None,
                      weight_col: str = None,
                      count_col: str = None) -> gpd.GeoDataFrame:
    """
    Spatially join crime points to H3 hexagons and summarise by hex.

    Parameters:
    ----------
    hexes_gdf : GeoDataFrame
        Must contain 'hex_id' and geometry.
    crimes_gdf : GeoDataFrame
        Point (or MultiPoint) GeoDataFrame. If CRS differs, it will be
        reprojected to match hexes_gdf.
    count : bool, optional
        If True, counts all points in each hex. Outputs 'crime_count'.
    weight_col : str, optional
        If not None, sums this field for points in each hex.
        Outputs 'crime_weight'.
    count_col : str, optional
        .. deprecated:: 0.2.0
            Use `count=True` instead. In the original API this was a
            string whose value was ignored (only whether it was None
            mattered), which was confusing -- any non-None value here is
            treated as `count=True`.

    Returns:
    -------
    GeoDataFrame
        Copy of hexes_gdf with added 'crime_count' and/or 'crime_weight'.

    Notes:
    -----
    Points that fall exactly on a hex boundary edge belong to neither
    hex's interior and are therefore excluded (a `predicate='within'`
    join). This is usually a vanishingly small edge case, but a warning
    is raised if any points are dropped for this or any other reason
    (e.g. falling outside all hexes).

    Example:
    --------
    hex_both = summarise_by_hex(
        hexes_gdf=hexes,
        crimes_gdf=gdf_crimes,
        count=True,
        weight_col='pseudo_harm'
    )
    """
    if count_col is not None:
        warnings.warn(
            "count_col is deprecated and will be removed in a future "
            "release; use count=True instead. The value passed to "
            "count_col is ignored either way.",
            DeprecationWarning,
            stacklevel=2,
        )
        if count is None:
            count = True

    v.require_geodataframe(hexes_gdf, "hexes_gdf")
    v.require_column(hexes_gdf, "hex_id", "hexes_gdf")
    v.require_geom_type(crimes_gdf, {"Point", "MultiPoint"}, "crimes_gdf")
    v.require_crs(crimes_gdf, "crimes_gdf")
    if weight_col is not None:
        v.require_column(crimes_gdf, weight_col, "crimes_gdf")

    if not count and weight_col is None:
        raise ValueError("Must pass count=True and/or weight_col")

    # Ensure both are in the same CRS
    if crimes_gdf.crs != hexes_gdf.crs:
        crimes = crimes_gdf.to_crs(hexes_gdf.crs)
    else:
        crimes = crimes_gdf

    # Spatial join points to hexes (brings in hex_id on each crime)
    joined = gpd.sjoin(
        crimes,
        hexes_gdf[['hex_id', 'geometry']],
        how='inner',
        predicate='within'
    )
    # joined now has a 'hex_id' column for each crime

    n_total = len(crimes)
    n_matched = joined.index.nunique()
    if n_matched < n_total:
        warnings.warn(
            f"{n_total - n_matched} of {n_total} points did not fall "
            f"within any hex (commonly points exactly on a hex boundary "
            f"edge, or outside the hex coverage area) and were excluded "
            f"from the counts/weights.",
            RuntimeWarning,
        )

    # Prepare output
    out = hexes_gdf.copy()

    # Count crimes if requested
    if count:
        counts = (
            joined
            .groupby('hex_id')
            .size()
            .rename('crime_count')
        )
        out = out.merge(counts, on='hex_id', how='left')
        out['crime_count'] = out['crime_count'].fillna(0).astype(int)

    # Sum weights if requested
    if weight_col is not None:
        weights = (
            joined
            .groupby('hex_id')[weight_col]
            .sum()
            .rename('crime_weight')
        )
        out = out.merge(weights, on='hex_id', how='left')
        out['crime_weight'] = out['crime_weight'].fillna(0)

    return out

################
## Function 3 ##
################

def add_spatial_lag(hexes_gdf: gpd.GeoDataFrame,
                     count_col: str = None,
                     weight_col: str = None,
                     k: int = 1,
                     method: str = "h3") -> gpd.GeoDataFrame:
    """
    Given a GeoDataFrame of hexagons (with 'hex_id' and geometry),
    add spatial-lag features (sum/mean of neighbours).

    Parameters:
    ----------
    hexes_gdf : GeoDataFrame
        Must contain 'hex_id' and geometry.
    count_col : str, optional
        If provided, name of the integer column to lag. Adds:
          - 'lag_sum_count', 'lag_mean_count'
          - 'count_plus_sum', 'count_plus_mean'
    weight_col : str, optional
        If provided, name of the numeric column to lag. Adds:
          - 'lag_sum_weight', 'lag_mean_weight'
          - 'weight_plus_sum_sqrt', 'weight_plus_mean_sqrt'
    k : int
        method='h3': number of H3 grid rings out (k=1 is the immediate
        6-hex "donut", matching the historical default behaviour).
        method='knn': number of nearest neighbours by centroid distance.
    method : {'h3', 'knn'}
        'h3' (default): true H3 grid adjacency via `h3.grid_disk`. Purely
        topological -- no CRS/units requirement. Hexes with no neighbours
        present in `hexes_gdf` within the ring distance (e.g. at the edge
        of the study area) get a lag of 0 over however many neighbours
        *are* present, and a warning is raised summarising how many hexes
        that affected.
        'knn' (legacy): finds the k nearest hex centroids by Euclidean
        distance in `hexes_gdf`'s CRS. Requires a projected CRS. This can
        pick geometrically-nearby hexes that are not true H3 neighbours
        (e.g. across a gap between two districts), so 'h3' is preferred
        for anything meant to represent real adjacency.

    Returns:
    -------
    GeoDataFrame
        A copy of `hexes_gdf` with the new lag columns appended.

    .. versionchanged:: 0.2.0
        Default changed from an approximate `k=6` nearest-centroid lookup
        (`method='knn'`) to true H3 ring adjacency (`method='h3'`,
        `k=1`), which is what the original docstring already claimed to
        do ("typically a 'donut' around hexagons") but did not actually
        implement. Pass `method='knn', k=6` to reproduce the exact old
        behaviour.

    Example:
    -------
    hex_lagged = add_spatial_lag(
        hexes_gdf=hex_both,
        count_col='crime_count',
        weight_col='crime_weight',
    )
    """
    v.require_geodataframe(hexes_gdf, "hexes_gdf")
    v.require_column(hexes_gdf, "hex_id", "hexes_gdf")
    if count_col:
        v.require_column(hexes_gdf, count_col, "hexes_gdf")
    if weight_col:
        v.require_column(hexes_gdf, weight_col, "hexes_gdf")

    if not count_col and not weight_col:
        raise ValueError("Must specify at least one of count_col or weight_col")

    out = hexes_gdf.copy()

    if method == "h3":
        hex_ids = out["hex_id"].to_numpy()
        pos_by_id = {h: i for i, h in enumerate(hex_ids)}

        nbr_positions = []
        for h in hex_ids:
            ring = _h3_grid_disk(h, k)
            ring.discard(h)
            nbr_positions.append([pos_by_id[n] for n in ring if n in pos_by_id])

        n_isolated = sum(1 for idx in nbr_positions if len(idx) == 0)
        if n_isolated:
            warnings.warn(
                f"{n_isolated} of {len(out)} hexes have no neighbours "
                f"present in hexes_gdf within k={k} ring(s) (likely at "
                f"the edge of the study area). Their lag values are 0.",
                RuntimeWarning,
            )

        def _lag(values):
            values = np.asarray(values, dtype=float)
            sum_nb = np.array([values[idx].sum() if idx else 0.0 for idx in nbr_positions])
            n_nb = np.array([len(idx) for idx in nbr_positions], dtype=float)
            mean_nb = np.divide(sum_nb, n_nb, out=np.zeros_like(sum_nb), where=n_nb > 0)
            return sum_nb, mean_nb

        if count_col:
            sum_nb, mean_nb = _lag(out[count_col])
            out['lag_sum_count'] = sum_nb
            out['lag_mean_count'] = mean_nb
            out['count_plus_sum'] = out[count_col] + sum_nb
            out['count_plus_mean'] = out[count_col] + mean_nb

        if weight_col:
            sum_nb, mean_nb = _lag(out[weight_col])
            out['lag_sum_weight'] = sum_nb
            out['lag_mean_weight'] = mean_nb
            out['weight_plus_sum_sqrt'] = np.sqrt(np.clip(out[weight_col] + sum_nb, 0, None))
            out['weight_plus_mean_sqrt'] = np.sqrt(np.clip(out[weight_col] + mean_nb, 0, None))

        return out

    elif method == "knn":
        v.require_projected_crs(out, "hexes_gdf")
        v.require_min_rows(out, k + 1, "hexes_gdf")

        # Build centroid coordinate array
        pts = np.array([
            (geom.centroid.x, geom.centroid.y)
            for geom in out.geometry
        ])

        # Fit KNN (including self at position 0)
        knn = NearestNeighbors(n_neighbors=k + 1, algorithm='auto').fit(pts)
        _, nbrs = knn.kneighbors(pts)
        nbrs = nbrs[:, 1:]  # drop self

        # Spatial lag for counts
        if count_col:
            counts = out[count_col].to_numpy()
            sum_nb = np.array([counts[ids].sum() for ids in nbrs])
            out['lag_sum_count'] = sum_nb
            out['lag_mean_count'] = sum_nb / k
            out['count_plus_sum'] = out[count_col] + sum_nb
            out['count_plus_mean'] = out[count_col] + (sum_nb / k)

        # Spatial lag for weights
        if weight_col:
            weights = out[weight_col].to_numpy()
            wsum_nb = np.array([weights[ids].sum() for ids in nbrs])
            out['lag_sum_weight'] = wsum_nb
            out['lag_mean_weight'] = wsum_nb / k
            out['weight_plus_sum_sqrt'] = np.sqrt(np.clip(out[weight_col] + wsum_nb, 0, None))
            out['weight_plus_mean_sqrt'] = np.sqrt(np.clip(out[weight_col] + (wsum_nb / k), 0, None))

        return out

    else:
        raise ValueError(f"method must be 'h3' or 'knn', got {method!r}")

################
## Function 4 ##
################

def add_spatial_stats(hex_gdf: gpd.GeoDataFrame,
                       col: str,
                       group_col: str) -> gpd.GeoDataFrame:
    """
    Given a GeoDataFrame with numeric column `col` and a grouping column `group_col`,
    add four new columns:
      - '{col}_zscore'                  : global z-score of col
      - '{col}_rank'                    : global rank (1 = highest)
      - '{col}_zscore_by_{group_col}'   : z-score within each group
      - '{col}_rank_by_{group_col}'     : rank within each group (1 = highest)

    Returns a new GeoDataFrame with these columns appended.

    Example:
    -------
    hex_stats = add_spatial_stats(hex_lagged, col='weight_plus_mean_sqrt', group_col='geo_boundary')
    """
    v.require_geodataframe(hex_gdf, "hex_gdf")
    v.require_column(hex_gdf, col, "hex_gdf")
    v.require_column(hex_gdf, group_col, "hex_gdf")
    if not pd.api.types.is_numeric_dtype(hex_gdf[col]):
        raise TypeError(f"'{col}' must be numeric, got dtype {hex_gdf[col].dtype}")

    df = hex_gdf.copy()

    # Global z-score and rank
    mean_all = df[col].mean()
    std_all = df[col].std(ddof=0) if df[col].std(ddof=0) != 0 else 1
    df[f"{col}_zscore"] = (df[col] - mean_all) / std_all
    df[f"{col}_rank"] = df[col].rank(ascending=False, method='min').astype(int)

    # Grouped z-score and rank
    df[f"{col}_zscore_by_{group_col}"] = df.groupby(group_col)[col] \
        .transform(lambda x: (x - x.mean()) / (x.std(ddof=0) if x.std(ddof=0) != 0 else 1))
    df[f"{col}_rank_by_{group_col}"] = df.groupby(group_col)[col] \
        .transform(lambda x: x.rank(ascending=False, method='min').astype(int))

    return gpd.GeoDataFrame(df, geometry=hex_gdf.geometry.name, crs=hex_gdf.crs)

################
## Function 5 ##
################

def count_crimes_by_nearest_road(crimes_gdf: gpd.GeoDataFrame,
                                  roads_gdf: gpd.GeoDataFrame,
                                  max_dist: float = 75.0) -> gpd.GeoDataFrame:
    """
    Snap each crime to the nearest road segment within max_dist via sjoin_nearest,
    then count how many crimes fell on each segment.

    Parameters:
    ----------
    crimes_gdf : GeoDataFrame (points)
        Crime locations.
    roads_gdf : GeoDataFrame (lines)
        Road segments. Must be in a projected CRS, since `max_dist` is
        interpreted in that CRS's linear units (e.g. metres). crimes_gdf
        is reprojected to match if its CRS differs.
    max_dist : float
        Maximum snapping distance in CRS units (default 75, e.g. metres).

    Returns:
    -------
    GeoDataFrame
        Copy of roads_gdf with new column 'crime_count' (int). The number
        of crimes that couldn't be matched within max_dist, and the total
        considered, are attached as `.attrs['n_unmatched']` and
        `.attrs['n_total']`, and a warning is raised if any were dropped.
        Points tied for equally-nearest between two segments are matched
        to just one (not double-counted).

    Example:
    -------
    roads_with_counts = count_crimes_by_nearest_road(
        crimes_gdf=gdf_crimes,
        roads_gdf=gdf_roads,
        max_dist=75
    )
    """
    v.require_geom_type(crimes_gdf, {"Point", "MultiPoint"}, "crimes_gdf")
    v.require_geom_type(roads_gdf, {"LineString", "MultiLineString"}, "roads_gdf")
    v.require_crs(crimes_gdf, "crimes_gdf")
    v.require_projected_crs(roads_gdf, "roads_gdf")

    # Ensure same CRS
    if crimes_gdf.crs != roads_gdf.crs:
        crimes = crimes_gdf.to_crs(roads_gdf.crs)
    else:
        crimes = crimes_gdf

    # Tag roads with an explicit ID
    roads = roads_gdf.copy()
    roads["road_id"] = roads.index

    # Nearest join: each crime gets the 'road_id' of its nearest road.
    # how='left' (rather than 'inner') keeps every crime row, with NaN
    # road_id for anything beyond max_dist, so unmatched points can be
    # counted instead of silently vanishing.
    joined = gpd.sjoin_nearest(
        crimes,
        roads[["road_id", "geometry"]],
        how="left",
        max_distance=max_dist,
        distance_col="_dist",
    )

    # sjoin_nearest emits one row per (point, tied-nearest-road) pair, so
    # an exact tie between two equally-close segments produces two rows
    # for the same crime. Keep just one per crime so it isn't double-counted.
    joined = joined.sort_values("_dist").loc[~joined.index.duplicated(keep="first")]

    n_total = len(crimes)
    n_unmatched = int(joined["road_id"].isna().sum())
    if n_unmatched:
        warnings.warn(
            f"{n_unmatched} of {n_total} crime points were further than "
            f"max_dist={max_dist} from any road and were not counted. "
            f"Increase max_dist if that's not intended.",
            RuntimeWarning,
        )
    joined = joined.dropna(subset=["road_id"])

    # Count crimes per road_id
    counts = (
        joined
        .groupby("road_id")
        .size()
        .rename("crime_count")
    )

    # Merge counts back onto the original roads GeoDataFrame
    out = roads_gdf.copy()
    out["crime_count"] = out.index.map(counts).fillna(0).astype(int)
    out.attrs["n_unmatched"] = n_unmatched
    out.attrs["n_total"] = n_total

    return out

################
## Function 6 ##
################

def build_adj_graph(roads_gdf: gpd.GeoDataFrame,
                     fid_col: str = None,
                     crime_count_col: str = 'crime_count',
                     snap_tolerance: float = 0.0):
    """
    Build a contiguous adjacency graph of road segments.

    Parameters:
    ----------
    roads_gdf : GeoDataFrame
        Must contain:
          - geometry: LineString/MultiLineString segments
          - crime_count_col: numeric attribute on each segment
    fid_col : str or None
        Name of the unique ID column (default None). If None or not
        found, a new integer 'fid' column will be created.
    crime_count_col : str
        Name of the crime count column (default 'crime_count').
    snap_tolerance : float, default 0.0
        Distance (in roads_gdf's CRS linear units) within which two
        segments are treated as adjacent, in addition to exact
        topological touching. Real-world, commercially-sourced road
        centreline data very often has small digitising gaps or
        overshoots at junctions that `touches()` alone will miss; a
        value of a metre or two is usually enough to bridge those without
        over-connecting unrelated streets. 0.0 reproduces the original
        exact-touching-only behaviour.

    Returns:
    -------
    (G, df) : tuple
        G  : networkx.Graph. Nodes are segment IDs (from fid_col or
             generated), each with attribute 'crime_count'. Edges connect
             segments that touch (or are within snap_tolerance).
        df : the GeoDataFrame actually used to build G, containing a
             column named fid_col whose values are exactly G's node IDs.

             **Always pass this same df on to `clusters_to_gdf`, not your
             original roads_gdf.**

    .. versionchanged:: 0.2.0
        Now returns `(G, df)` instead of just `G`. Previously, when
        `fid_col=None`, node IDs were generated from a positional reset
        index *inside* this function and never handed back; if the
        caller's own DataFrame didn't already have a clean 0..n-1 index
        (e.g. after any filtering), `clusters_to_gdf` would silently pair
        each cluster with the wrong row's geometry. Returning the exact
        `df` used to build `G` removes that failure mode.

    Example:
    -------
    G, roads_indexed = build_adj_graph(
        roads_with_counts,
        fid_col=None,
        crime_count_col='crime_count',
    )
    """
    v.require_geom_type(roads_gdf, {"LineString", "MultiLineString"}, "roads_gdf")
    v.require_column(roads_gdf, crime_count_col, "roads_gdf")
    if snap_tolerance < 0:
        raise ValueError(f"snap_tolerance must be >= 0, got {snap_tolerance}")

    # Copy to avoid modifying original
    df = roads_gdf.copy()

    # If no fid_col provided or missing, generate one
    if fid_col is None or fid_col not in df.columns:
        df = df.reset_index(drop=True)
        df['fid'] = df.index.astype(int)
        fid_col = 'fid'
    elif df[fid_col].duplicated().any():
        n_dupe = int(df[fid_col].duplicated().sum())
        raise ValueError(
            f"fid_col '{fid_col}' must be unique; found {n_dupe} duplicate "
            f"value(s). Pass fid_col=None to generate a fresh unique ID."
        )

    # Prepare DataFrame indexed by fid
    df_idx = df[[fid_col, 'geometry', crime_count_col]].set_index(fid_col)

    # Spatial index for quick bbox queries
    sindex = df_idx.sindex

    # Initialise graph and add nodes
    G = nx.Graph()
    for fid in df_idx.index:
        G.add_node(fid, crime_count=df_idx.at[fid, crime_count_col])

    # Add edges between touching (or near-touching) segments
    for fid, geom in df_idx.geometry.items():
        minx, miny, maxx, maxy = geom.bounds
        if snap_tolerance > 0:
            minx, miny = minx - snap_tolerance, miny - snap_tolerance
            maxx, maxy = maxx + snap_tolerance, maxy + snap_tolerance
        candidate_pos = list(sindex.intersection((minx, miny, maxx, maxy)))
        candidate_fids = df_idx.iloc[candidate_pos].index
        for nbr in candidate_fids:
            if nbr == fid or G.has_edge(fid, nbr):
                continue
            other = df_idx.at[nbr, 'geometry']
            if snap_tolerance > 0:
                connected = geom.distance(other) <= snap_tolerance
            else:
                connected = geom.touches(other)
            if connected:
                G.add_edge(fid, nbr)

    return G, df

################
## Function 7 ##
################

def segment_clusters(G, min_size=2, max_size=10, min_crimes=10):
    """
    G: NetworkX graph with node attribute 'crime_count' (as returned by
       build_adj_graph).
    Returns a list of dicts:
      - 'cluster_id': sequential ID
      - 'nodes': set of node-IDs (fids)
      - 'crime_sum': total crime_count in the cluster

    Note this is a greedy heuristic (highest-crime segments seeded first,
    growing by adding the highest-crime touching neighbour), not an
    optimal solution -- it's fast and produces reasonable, if not
    guaranteed-optimal, clusters. It also doesn't optimise for cluster
    *shape*: nothing stops a cluster from growing into a long, thin,
    hard-to-patrol chain of segments rather than a compact area. If that
    matters for your use case, treat the output as a starting point and
    inspect cluster geometries (e.g. via clusters_to_gdf) before using
    them operationally.

    Example:
    -------
    clusters = segment_clusters(G, min_size=2, max_size=10, min_crimes=24)
    """
    if G.number_of_nodes() == 0:
        raise ValueError(
            "G has no nodes. Check that the roads_gdf passed to "
            "build_adj_graph wasn't empty."
        )
    if min_size > max_size:
        raise ValueError(f"min_size ({min_size}) cannot exceed max_size ({max_size})")
    missing = [n for n in G.nodes if 'crime_count' not in G.nodes[n]]
    if missing:
        raise ValueError(
            "G nodes are missing the 'crime_count' attribute. Build G "
            "with build_adj_graph rather than constructing it directly."
        )

    seeds = sorted(G.nodes, key=lambda n: G.nodes[n]['crime_count'], reverse=True)
    used = set()
    clusters = []
    cluster_id = 1

    for seed in seeds:
        if seed in used:
            continue

        cluster = {seed}
        frontier = set(G.neighbors(seed))

        while frontier and len(cluster) < max_size:
            nxt = max(frontier, key=lambda n: G.nodes[n]['crime_count'])
            frontier.remove(nxt)
            if nxt in cluster:
                continue
            cluster.add(nxt)
            used.add(nxt)
            frontier |= set(G.neighbors(nxt)) - cluster

        total = sum(G.nodes[n]['crime_count'] for n in cluster)
        if len(cluster) >= min_size and total >= min_crimes:
            clusters.append({
                'cluster_id': cluster_id,
                'nodes': cluster,
                'crime_sum': total
            })
            used |= cluster
            cluster_id += 1

    return clusters

################
## Function 8 ##
################

def clusters_to_gdf(clusters, G, df, fid_col='fid', crime_count_col='crime_count', crs=None):
    """
    Convert cluster dicts into a GeoDataFrame.

    Parameters:
    ----------
    clusters : list of dict
        Each dict from segment_clusters must have keys:
        - 'cluster_id': int
        - 'nodes': iterable of fid values
        - 'crime_sum': total crime count for the cluster
    G : networkx.Graph
        Graph used to generate clusters, with node attribute crime_count.
    df : GeoDataFrame
        **The exact `df` returned by `build_adj_graph`** (its second
        return value), so that fid values in `clusters`/`G` line up with
        the correct row. Passing your original, pre-`build_adj_graph`
        roads_gdf here is the single most common way to get silently
        mismatched geometries -- this function will raise a clear
        `KeyError` instead if a node's fid can't be found, but it can't
        detect a *different* row's fid coincidentally matching.
    fid_col : str
        Column name in df matching nodes in clusters.
    crime_count_col : str
        Name of crime count attribute in G and/or df.
    crs : dict or string, optional
        Coordinate reference system for the output. Defaults to df.crs.

    Returns:
    -------
    GeoDataFrame
        Each row is one segment in a cluster, with columns:
        - cluster_id, fid, cluster_crime_sum, crime_count, geometry
        Empty (but correctly-shaped) if `clusters` is empty.

    Example:
    ------
    gdf_clusters = clusters_to_gdf(clusters, G, roads_indexed, fid_col='fid')
    """
    out_crs = crs if crs is not None else getattr(df, "crs", None)

    if not clusters:
        warnings.warn("clusters is empty; returning an empty GeoDataFrame.", RuntimeWarning)
        return gpd.GeoDataFrame(
            columns=['cluster_id', 'fid', 'cluster_crime_sum', 'crime_count', 'geometry'],
            geometry='geometry',
            crs=out_crs,
        )

    v.require_column(df, fid_col, "df")
    v.require_column(df, 'geometry', "df")

    df_idx = df.set_index(fid_col) if df.index.name != fid_col else df

    records = []
    for cl in clusters:
        cid = cl['cluster_id']
        total = cl['crime_sum']
        for fid in cl['nodes']:
            if fid not in df_idx.index:
                raise KeyError(
                    f"fid {fid!r} from clusters/G was not found in df's "
                    f"'{fid_col}' column. Make sure `df` is the same "
                    f"object build_adj_graph returned alongside G, not "
                    f"your original roads_gdf."
                )
            records.append({
                'cluster_id': cid,
                'fid': fid,
                'cluster_crime_sum': total,
                'crime_count': G.nodes[fid][crime_count_col],
                'geometry': df_idx.at[fid, 'geometry'],
            })

    return gpd.GeoDataFrame(records, crs=out_crs)


################
## Function 9 ##
################

def create_folium_map(hex_gdf=None, hex_query=None, hex_color_col=None,
                       seg_gdf=None, seg_query=None, seg_color_col=None,
                       district_gdf=None, district_query=None,
                       zoom_start=12):
    """
    Create a Folium map with optional layers:
      - hexagons (filtered by hex_query, optionally choropleth-coloured)
      - road segments (filtered by seg_query, optionally choropleth-coloured)
      - district boundaries (outline only, filtered by district_query)

    Parameters:
    ----------
      hex_gdf: GeoDataFrame of hex polygons
      hex_query: string query for hex_gdf (e.g. "rank <= 100")
      hex_color_col: numeric column to choropleth-colour hexes by (e.g. a
        rank or z-score column). If omitted, hexes are drawn as plain
        black outlines as before.
      seg_gdf: GeoDataFrame of line segments
      seg_query: string query for seg_gdf (e.g. "cluster_crime_sum > 50")
      seg_color_col: numeric column to choropleth-colour segments by.
      district_gdf: GeoDataFrame of polygons
      district_query: string query for district_gdf
      zoom_start: initial zoom level (default 12)

    Returns:
    -------
      folium.Map object with layer control, OSM/Positron basemaps, and
      hover tooltips on every layer showing that feature's attributes.

    Notes:
    -----
    If a `*_query` filters a layer down to zero rows, that layer is
    skipped with a warning rather than raising (e.g. an empty map centred
    at (0, 0) from a failed `.centroid` call), so one overly-strict query
    doesn't take down the whole map.

    Example:
    -------
      m = create_folium_map(
          hex_gdf=hex_stats,
          hex_query="weight_plus_mean_sqrt_rank_by_geo_boundary <= 20",
          hex_color_col="weight_plus_mean_sqrt_rank_by_geo_boundary",
          seg_gdf=gdf_clusters,
          seg_query="cluster_crime_sum > 50",
          district_gdf=gdf_districts)
      m
    """
    if hex_gdf is None and seg_gdf is None and district_gdf is None:
        raise ValueError("Provide at least one of hex_gdf, seg_gdf, district_gdf")

    # Determine map centre from first non-empty layer (projected to WGS84)
    center = [0, 0]
    for gdf in (hex_gdf, seg_gdf, district_gdf):
        if gdf is not None and not gdf.empty:
            wgs = gdf.to_crs("EPSG:4326")
            merged = wgs.geometry.union_all()
            ctr = merged.centroid
            center = [ctr.y, ctr.x]
            break

    # Initialise Folium map with no default tiles
    m = folium.Map(location=center, zoom_start=zoom_start, tiles=None)

    # Add base layers
    folium.TileLayer('OpenStreetMap', name='OSM', control=True).add_to(m)
    folium.TileLayer('CartoDB Positron', name='CartoDB Positron', control=True).add_to(m)

    def _add_layer(gdf, query, layer_name, show, color_col, filled):
        if gdf is None:
            return
        sel = gdf.query(query) if query else gdf
        if sel.empty:
            warnings.warn(
                f"{layer_name} layer: query {query!r} matched no rows; "
                f"layer skipped.",
                RuntimeWarning,
            )
            return

        wgs = sel.to_crs("EPSG:4326")
        fg = folium.FeatureGroup(name=layer_name, show=show)
        geom_col = wgs.geometry.name
        tooltip_fields = [c for c in wgs.columns if c != geom_col][:8]
        tooltip = folium.GeoJsonTooltip(fields=tooltip_fields) if tooltip_fields else None

        if color_col:
            v.require_column(wgs, color_col, layer_name)
            from branca.colormap import linear
            vmin, vmax = wgs[color_col].min(), wgs[color_col].max()
            if vmax > vmin:
                colormap = linear.YlOrRd_09.scale(vmin, vmax)
            else:
                colormap = lambda _v: '#800026'  # noqa: E731 -- single uniform colour when there's no spread

            def style_function(feature, colormap=colormap, color_col=color_col, filled=filled):
                return {
                    'fillColor': colormap(feature['properties'][color_col]),
                    'fillOpacity': 0.7 if filled else 0,
                    'color': 'black',
                    'weight': 1 if filled else 2,
                }
        else:
            def style_function(feature, filled=filled):
                return {
                    'fillOpacity': 0.4 if filled else 0,
                    'color': 'black',
                    'weight': 1 if filled else 2,
                }

        folium.GeoJson(wgs, style_function=style_function, tooltip=tooltip).add_to(fg)
        fg.add_to(m)

        if color_col and vmax > vmin:
            colormap.caption = f"{layer_name}: {color_col}"
            colormap.add_to(m)

    _add_layer(hex_gdf, hex_query, 'Hexagons', True, hex_color_col, filled=True)
    _add_layer(seg_gdf, seg_query, 'Segments', False, seg_color_col, filled=bool(seg_color_col))
    _add_layer(district_gdf, district_query, 'District Boundary', False, None, filled=False)

    # Layer control
    folium.LayerControl(collapsed=False).add_to(m)

    return m
