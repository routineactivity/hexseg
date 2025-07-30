import pandas as pd
import numpy as np
import math
import geopandas as gpd
import h3
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
from sklearn.neighbors import NearestNeighbors
import networkx as nx
import folium

################
# Troubleshoot #
################

## using this space to test and troubleshoot functions

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
        
    name_col : str
        Police force, district or other geography name
    resolution : int
        Uber hexagon resolution, see: https://h3geo.org/docs/3.x/core-library/restable/

    Example:
    --------
    hexes = get_hexagons(gdf_districts, name_col="lad21nm", resolution=9)
    """

    # Validate and reproject to WGS84 for H3
    assert gdf_polygons.crs, "Input must have a valid CRS"
    wgs = gdf_polygons.to_crs("EPSG:4326")

    records = []
    # Loop over features, handle Polygons & MultiPolygons
    for _, row in wgs.iterrows():
        geom = row.geometry
        boundary_name = row[name_col]
        parts = geom.geoms if isinstance(geom, MultiPolygon) else [geom]

        # collect all hex IDs
        hex_ids = set()
        for part in parts:
            geoif = part.__geo_interface__
            if hasattr(h3, "geo_to_cells"):
                ids = h3.geo_to_cells(geoif, resolution)
            elif hasattr(h3, "polygon_to_cells"):
                ids = h3.polygon_to_cells(geoif, resolution, geo_json_conformant=True)
            else:
                ids = h3.polyfill(geoif, resolution, geo_json_conformant=True)
            hex_ids.update(ids)

        # build polygons: **always** use lat/lon tuples flipped to lon/lat
        #for h in hex_ids:
        #    # this returns a list of (lat, lon) tuples
        #    coords = h3.h3_to_geo_boundary(h)
        #    # flip to (x, y) = (lon, lat)
        #    pts = [(lng, lat) for lat, lng in coords]
        #    records.append((h, boundary_name, Polygon(pts)))

        for h in hex_ids:
            # get the hex boundary as a list of (lat, lon)
            if hasattr(h3, "h3_to_geo_boundary"):
                coords = h3.h3_to_geo_boundary(h)
            elif hasattr(h3, "cell_to_boundary"):
                coords = h3.cell_to_boundary(h)
            else:
                raise AttributeError("h3 module has no cell boundary function")

            pts = [(lng, lat) for lat, lng in coords]
            records.append((h, boundary_name, Polygon(pts)))

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

    # Pick the best overlap per hex_id
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
                     count_col: str = None,
                     weight_col: str = None) -> gpd.GeoDataFrame:
    """
    Spatially join crime points to H3 hexagons and summarise by hex.

    Parameters:
    ----------
    hexes_gdf : GeoDataFrame
        Must contain 'hex_id' and geometry.
    crimes_gdf : GeoDataFrame
        Point GeoDataFrame. If CRS differs, it will be reprojected.
    count_col : str, optional
        If not None, counts all points in each hex (column name is ignored).
        Outputs 'crime_count'.
    weight_col : str, optional
        If not None, sums this field for points in each hex.
        Outputs 'crime_weight'.

    Returns:
    -------
    GeoDataFrame
        Copy of hexes_gdf with added 'crime_count' and/or 'crime_weight'.

    Example:
    --------
    hex_both = summarise_by_hex(
    hexes_gdf=hexes,
    crimes_gdf=gdf_crimes,
    count_col='any',
    weight_col='pseudo_harm'
    )
    """
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

    # Prepare output
    out = hexes_gdf.copy()

    # Count crimes if requested
    if count_col is not None:
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
        if weight_col not in crimes.columns:
            raise KeyError(f"Weight column '{weight_col}' not found in crimes_gdf")
        weights = (
            joined
            .groupby('hex_id')[weight_col]
            .sum()
            .rename('crime_weight')
        )
        out = out.merge(weights, on='hex_id', how='left')
        out['crime_weight'] = out['crime_weight'].fillna(0)

    # Require at least one
    if count_col is None and weight_col is None:
        raise ValueError("Must pass at least one of count_col or weight_col")

    return out

################
## Function 3 ##
################   

def add_spatial_lag(hexes_gdf: gpd.GeoDataFrame,
                    count_col: str = None,
                    weight_col: str = None,
                    k: int = 6) -> gpd.GeoDataFrame:
    """
    Given a GeoDataFrame of hexagons (with 'hex_id' and geometry),
    computes K-nearest neighbours (by centroid) and adds lag features.

    Parameters:
    ----------
    hexes_gdf : GeoDataFrame
        Must be in a projected CRS (so distances are planar).
    count_col : str, optional
        If provided, name of the integer column to count. Adds:
          - 'lag_sum_count'
          - 'lag_mean_count'
          - 'count_plus_sum'
          - 'count_plus_mean'
    weight_col : str, optional
        If provided, name of the numeric column to sum. Adds:
          - 'lag_sum_weight'
          - 'lag_mean_weight'
          - 'weight_plus_sum_sqrt'
          - 'weight_plus_mean_sqrt'
    k : int
        Number of neighbours (default = 6).

    Returns:
    -------
    GeoDataFrame
        A copy of `hexes_gdf` with the new lag columns appended.

    Example:
    -------

    hex_lagged = add_spatial_lag(
    hexes_gdf=hex_both,
    count_col='crime_count',
    weight_col='crime_weight',
    k=6
    )
    """
    # Build centroid coordinate array
    pts = np.array([
        (geom.centroid.x, geom.centroid.y)
        for geom in hexes_gdf.geometry
    ])

    # Fit KNN (including self at position 0)
    knn = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(pts)
    _, nbrs = knn.kneighbors(pts)
    nbrs = nbrs[:, 1:]  # drop self

    out = hexes_gdf.copy()

    # Spatial lag for counts
    if count_col:
        counts = out[count_col].to_numpy()
        sum_nb = np.array([counts[ids].sum() for ids in nbrs])
        out['lag_sum_count']  = sum_nb
        out['lag_mean_count'] = sum_nb / k
        out['count_plus_sum'] = out[count_col] + sum_nb
        out['count_plus_mean']= out[count_col] + (sum_nb / k)

    # Spatial lag for weights
    if weight_col:
        weights = out[weight_col].to_numpy()
        wsum_nb = np.array([weights[ids].sum() for ids in nbrs])
        out['lag_sum_weight']       = wsum_nb
        out['lag_mean_weight']      = wsum_nb / k
        out['weight_plus_sum_sqrt'] = np.sqrt(out[weight_col] + wsum_nb)
        out['weight_plus_mean_sqrt']= np.sqrt(out[weight_col] + (wsum_nb / k))

    if not (count_col or weight_col):
        raise ValueError("Must specify at least one of count_col or weight_col")

    return out

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
    hex_stats = add_spatial_stats(hex_lagged, col='weight_plus_mean_sqrt', group_col='name')
    
    """
    df = hex_gdf.copy()

    # Global z-score and rank
    mean_all = df[col].mean()
    std_all  = df[col].std(ddof=0) if df[col].std(ddof=0) != 0 else 1
    df[f"{col}_zscore"] = (df[col] - mean_all) / std_all
    df[f"{col}_rank"]   = df[col].rank(ascending=False, method='min').astype(int)

    # Grouped z-score and rank
    # Z-score within group
    df[f"{col}_zscore_by_{group_col}"] = df.groupby(group_col)[col] \
                                            .transform(lambda x: (x - x.mean()) / (x.std(ddof=0) if x.std(ddof=0) != 0 else 1))
    # Rank within group
    df[f"{col}_rank_by_{group_col}"] = df.groupby(group_col)[col] \
                                          .transform(lambda x: x.rank(ascending=False, method='min').astype(int))

    return gpd.GeoDataFrame(df, geometry=hex_gdf.geometry, crs=hex_gdf.crs)

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
        Crime locations. Must be in a projected CRS.
    roads_gdf : GeoDataFrame (lines)
        Road segments, same CRS as crimes_gdf.
    max_dist : float
        Maximum snapping distance in CRS units (default 75 m).

    Returns:
    -------
    GeoDataFrame
        Copy of roads_gdf with new column 'crime_count' (int).

    Example:
    -------

    roads_with_counts = count_crimes_by_nearest_road(
    crimes_gdf=gdf_crimes,
    roads_gdf=gdf_roads,
    max_dist=75
    )
    """
    # Ensure same CRS
    if crimes_gdf.crs != roads_gdf.crs:
        crimes = crimes_gdf.to_crs(roads_gdf.crs)
    else:
        crimes = crimes_gdf

    # Tag roads with an explicit ID
    roads = roads_gdf.copy()
    roads["road_id"] = roads.index

    # Nearest join: each crime gets the 'road_id' of its nearest road (within max_dist)
    joined = gpd.sjoin_nearest(
        crimes,
        roads[["road_id", "geometry"]],
        how="inner",
        max_distance=max_dist
    )
    # joined now has a 'road_id' column

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

    return out

################
## Function 6 ##
################ 

def build_adj_graph(roads_gdf: gpd.GeoDataFrame,
                          fid_col: str = None,
                          crime_count_col: str = 'crime_count') -> nx.Graph:
    """
    Build a contiguous adjacency graph of road segments.

    Parameters:
    ----------
    roads_gdf : GeoDataFrame
        Must contain:
          - geometry: LineString segments
          - crime_count_col: numeric attribute on each segment
    fid_col : str or None
        Name of the unique ID column (default None). If None or not found,
        a new integer 'fid' index will be created.
    crime_count_col : str
        Name of the crime count column (default 'crime_count').

    Returns:
    -------
    G : networkx.Graph
        Undirected graph where:
          - nodes are segment IDs (from fid_col or generated), each with attribute 'crime_count'
          - edges connect segments whose geometries touch

    Example:
    -------
    G = build_adj_graph(roads_with_counts,
                          fid_col=None,
                          crime_count_col='crime_count')
    """
    # Copy to avoid modifying original
    df = roads_gdf.copy()

    # If no fid_col provided or missing, generate one
    if fid_col is None or fid_col not in df.columns:
        df = df.reset_index(drop=True)
        df['fid'] = df.index.astype(int)
        fid_col_internal = 'fid'
    else:
        fid_col_internal = fid_col

    # Ensure crime_count exists
    if crime_count_col not in df.columns:
        raise KeyError(f"Crime count column '{crime_count_col}' not found")

    # Prepare DataFrame indexed by fid
    df_idx = df[[fid_col_internal, 'geometry', crime_count_col]].set_index(fid_col_internal)
    
    # Spatial index for quick bbox queries
    sindex = df_idx.sindex
    
    # Initialise graph and add nodes
    G = nx.Graph()
    for fid in df_idx.index:
        G.add_node(fid, crime_count=df_idx.at[fid, crime_count_col])
    
    # Add edges between touching segments
    for fid, geom in df_idx.geometry.items():
        candidate_pos = list(sindex.intersection(geom.bounds))
        candidate_fids = df_idx.iloc[candidate_pos].index
        for nbr in candidate_fids:
            if nbr == fid:
                continue
            if geom.touches(df_idx.at[nbr, 'geometry']):
                G.add_edge(fid, nbr)
            
    return G

################
## Function 7 ##
################ 

def segment_clusters(G, min_size=2, max_size=10, min_crimes=10):
    """
    G: NetworkX graph with node attribute 'crime_count'.
    Returns a list of dicts:
      - 'cluster_id': sequential ID
      - 'nodes': set of node-IDs (fids)
      - 'crime_sum': total crime_count in the cluster

    Example:
    -------
    clusters = segment_clusters(G, min_size=2, max_size=10, min_crimes=24)
    """
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
        Each dict from greedy_clusters must have keys:
        - 'cluster_id': int
        - 'nodes': iterable of fid values
        - 'crime_sum': total crime count for the cluster
    G : networkx.Graph
        Graph used to generate clusters, with node attribute crime_count_col.
    df : GeoDataFrame
        Original roads GeoDataFrame indexed by fid_col or containing fid_col.
    fid_col : str
        Column name or index in df that matches nodes in clusters.
    crime_count_col : str
        Name of crime count attribute in G and/or df.
    crs : dict or string, optional
        Coordinate reference system to set on the output GeoDataFrame.

    Returns:
    -------
    GeoDataFrame
        Each row is one segment in a cluster, with columns:
        - cluster_id
        - fid
        - cluster_crime_sum
        - crime_count
        - geometry

    Example:
    ------
    gdf_clusters = clusters_to_gdf(clusters, G, roads_with_counts, fid_col='fid', crime_count_col='crime_count')
    
    """
    # Prepare df index
    #df_orig = df.copy()
    #if fid_col in df_orig.columns:
    #    df_idx = df_orig.set_index(fid_col)
    #    id_name = fid_col
    #else:
    #    df_idx = df_orig.copy()
    #    df_idx.index.name = fid_col
    #    id_name = fid_col

    # Build record dicts
    records = []
    for cl in clusters:
        cid   = cl['cluster_id']
        total = cl['crime_sum']
        for fid in cl['nodes']:
        #    if fid not in df_idx.index:
        #        continue
        #    geom  = df_idx.geometry.loc[fid]
        #    count = G.nodes[fid].get(crime_count_col, 0)
            records.append({
                'cluster_id'       : cid,
                'fid'            : fid,
                'cluster_crime_sum': total,
                'crime_count'    : G.nodes[fid]['crime_count'],
                'geometry'         : df.at[fid, 'geometry']
            })

    # Turn into a GeoDataFrame, explicitly setting the geometry column
    gdf = gpd.GeoDataFrame(
        records,
        crs=df.crs
    )
    return gdf


################
## Function 9 ##
################ 

def create_folium_map(hex_gdf=None, hex_query=None,
                      seg_gdf=None, seg_query=None,
                      district_gdf=None, district_query=None,
                      zoom_start=12):
    """
    Create a Folium map with optional layers:
      - hexagons (outline only, filtered by hex_query)
      - road segments (filtered by seg_query)
      - district boundaries (outline only, filtered by district_query)
    
    Parameters:
    ----------
      hex_gdf: GeoDataFrame of hex polygons
      hex_query: string query for hex_gdf (e.g. "rank <= 100")
      seg_gdf: GeoDataFrame of line segments
      seg_query: string query for seg_gdf (e.g. "cluster_crime_sum > 50")
      district_gdf: GeoDataFrame of polygons
      district_query: string query for district_gdf
      zoom_start: initial zoom level (default 12)
    
    Returns:
    -------
      folium.Map object with layer control and OSM/Positron basemaps.
      
    Example:
    -------
      m = create_folium_map(
      hex_gdf=hex_lagged,
      hex_query="rank <= 100",
      seg_gdf=gdf_clusters,
      seg_query="cluster_crime_sum > 50",
      district_gdf=gdf_districts,
      district_query=None) 
      
      m
    """
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
    m = folium.Map(location=center,
                   zoom_start=zoom_start,
                   tiles=None)

    # Add base layers
    folium.TileLayer('OpenStreetMap', name='OSM', control=True).add_to(m)
    folium.TileLayer('CartoDB Positron', name='CartoDB Positron', control=True).add_to(m)

    # Hexagon outlines layer
    if hex_gdf is not None:
        hex_sel = hex_gdf.query(hex_query) if hex_query else hex_gdf
        hex_wgs = hex_sel.to_crs("EPSG:4326")
        fg_hex = folium.FeatureGroup(name='Hexagons', show=True)
        folium.GeoJson(
            hex_wgs,
            style_function=lambda f: {'fillOpacity': 0, 'color': 'black', 'weight': 1}
        ).add_to(fg_hex)
        fg_hex.add_to(m)

    # Segment clusters layer
    if seg_gdf is not None:
        seg_sel = seg_gdf.query(seg_query) if seg_query else seg_gdf
        seg_wgs = seg_sel.to_crs("EPSG:4326")
        fg_seg = folium.FeatureGroup(name='Segments', show=False)
        folium.GeoJson(
            seg_wgs,
            style_function=lambda f: {'color': 'black', 'weight': 2}
        ).add_to(fg_seg)
        fg_seg.add_to(m)

    # District boundary layer
    if district_gdf is not None:
        dist_sel = district_gdf.query(district_query) if district_query else district_gdf
        dist_wgs = dist_sel.to_crs("EPSG:4326")
        fg_dist = folium.FeatureGroup(name='District Boundary', show=False)
        folium.GeoJson(
            dist_wgs,
            style_function=lambda f: {'fillOpacity': 0, 'color': 'black', 'weight': 2}
        ).add_to(fg_dist)
        fg_dist.add_to(m)

    # Layer control
    folium.LayerControl(collapsed=False).add_to(m)

    return m
