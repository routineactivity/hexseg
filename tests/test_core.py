import warnings

import geopandas as gpd
import networkx as nx
import pytest
from shapely.geometry import LineString, Point, Polygon

import hexseg as HS


# ---------------------------------------------------------------------------
# Fixtures: small, hand-built geometries so tests run instantly and are easy
# to reason about, rather than depending on the (large) bundled sample data.
# ---------------------------------------------------------------------------

@pytest.fixture
def district():
    return gpd.GeoDataFrame(
        {"district": ["A"], "geometry": [Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])]},
        crs="EPSG:4326",
    )


@pytest.fixture
def hexes(district):
    return HS.get_hexagons(district, name_col="district", resolution=7)


@pytest.fixture
def crimes(hexes):
    # One point at the centroid of every hex, plus a weight column, so every
    # hex gets a non-zero count/weight and grouping/ranking is well-defined.
    centroids = hexes.geometry.centroid
    return gpd.GeoDataFrame(
        {"pseudo_harm": [1.0] * len(hexes), "geometry": list(centroids)},
        crs=hexes.crs,
    )


@pytest.fixture
def projected_lines():
    # A little chain of 4 touching segments, in a projected (metric) CRS.
    return gpd.GeoDataFrame(
        {
            "crime_count": [10, 5, 20, 1],
            "geometry": [
                LineString([(0, 0), (10, 0)]),
                LineString([(10, 0), (20, 0)]),
                LineString([(20, 0), (30, 0)]),
                LineString([(30, 0), (40, 0)]),
            ],
        },
        crs="EPSG:27700",
    )


# ---------------------------------------------------------------------------
# get_hexagons
# ---------------------------------------------------------------------------

def test_get_hexagons_basic(hexes):
    assert "hex_id" in hexes.columns
    assert not hexes.empty


def test_get_hexagons_rejects_non_polygon_input():
    points = gpd.GeoDataFrame({"geometry": [Point(0, 0)]}, crs="EPSG:4326")
    with pytest.raises(TypeError):
        HS.get_hexagons(points, name_col="x")


def test_get_hexagons_requires_crs():
    poly = gpd.GeoDataFrame(
        {"geometry": [Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])], "d": ["A"]}
    )
    with pytest.raises(ValueError):
        HS.get_hexagons(poly, name_col="d")


def test_get_hexagons_requires_name_col(district):
    with pytest.raises(KeyError):
        HS.get_hexagons(district, name_col="does_not_exist")


# ---------------------------------------------------------------------------
# summarise_by_hex
# ---------------------------------------------------------------------------

def test_summarise_by_hex_count_and_weight(hexes, crimes):
    out = HS.summarise_by_hex(hexes, crimes, count=True, weight_col="pseudo_harm")
    assert "crime_count" in out.columns
    assert "crime_weight" in out.columns
    assert out["crime_count"].sum() == len(crimes)


def test_summarise_by_hex_requires_count_or_weight(hexes, crimes):
    with pytest.raises(ValueError):
        HS.summarise_by_hex(hexes, crimes)


def test_summarise_by_hex_rejects_non_point_crimes(hexes, district):
    with pytest.raises(TypeError):
        HS.summarise_by_hex(hexes, district, count=True)


def test_summarise_by_hex_deprecated_count_col_still_works(hexes, crimes):
    with pytest.deprecated_call():
        out = HS.summarise_by_hex(hexes, crimes, count_col="any")
    assert "crime_count" in out.columns


# ---------------------------------------------------------------------------
# add_spatial_lag
# ---------------------------------------------------------------------------

def test_add_spatial_lag_h3_default(hexes, crimes):
    hex_both = HS.summarise_by_hex(hexes, crimes, count=True, weight_col="pseudo_harm")
    lagged = HS.add_spatial_lag(hex_both, count_col="crime_count", weight_col="crime_weight")
    for col in ("lag_sum_count", "lag_mean_count", "lag_sum_weight", "weight_plus_mean_sqrt"):
        assert col in lagged.columns
    # h3 method needs no projected CRS at all -- WGS84 input is fine.
    assert lagged.crs == hexes.crs


def test_add_spatial_lag_knn_legacy_requires_projected_crs(hexes, crimes):
    hex_both = HS.summarise_by_hex(hexes, crimes, count=True, weight_col="pseudo_harm")
    with pytest.raises(ValueError):
        HS.add_spatial_lag(hex_both, count_col="crime_count", method="knn", k=3)


def test_add_spatial_lag_requires_a_column(hexes):
    with pytest.raises(ValueError):
        HS.add_spatial_lag(hexes)


def test_add_spatial_lag_unknown_method(hexes, crimes):
    hex_both = HS.summarise_by_hex(hexes, crimes, count=True)
    with pytest.raises(ValueError):
        HS.add_spatial_lag(hex_both, count_col="crime_count", method="bogus")


# ---------------------------------------------------------------------------
# add_spatial_stats
# ---------------------------------------------------------------------------

def test_add_spatial_stats(hexes, crimes):
    hex_both = HS.summarise_by_hex(hexes, crimes, count=True, weight_col="pseudo_harm")
    stats = HS.add_spatial_stats(hex_both, col="crime_weight", group_col="geo_boundary")
    assert "crime_weight_zscore" in stats.columns
    assert "crime_weight_rank" in stats.columns
    assert stats["crime_weight_rank"].min() == 1


def test_add_spatial_stats_rejects_non_numeric_col(hexes, crimes):
    hex_both = HS.summarise_by_hex(hexes, crimes, count=True)
    with pytest.raises(TypeError):
        HS.add_spatial_stats(hex_both, col="hex_id", group_col="geo_boundary")


# ---------------------------------------------------------------------------
# count_crimes_by_nearest_road
# ---------------------------------------------------------------------------

def test_count_crimes_by_nearest_road_basic(projected_lines):
    crimes = gpd.GeoDataFrame(
        {"geometry": [Point(1, 0.1), Point(1, 0.1), Point(35, 0.1)]},
        crs=projected_lines.crs,
    )
    out = HS.count_crimes_by_nearest_road(crimes, projected_lines, max_dist=5)
    assert out["crime_count"].sum() == 3
    assert out.attrs["n_unmatched"] == 0


def test_count_crimes_by_nearest_road_reports_unmatched(projected_lines):
    crimes = gpd.GeoDataFrame(
        {"geometry": [Point(1, 0.1), Point(1000, 1000)]}, crs=projected_lines.crs
    )
    with pytest.warns(RuntimeWarning, match="unmatched|further than|not counted"):
        out = HS.count_crimes_by_nearest_road(crimes, projected_lines, max_dist=5)
    assert out.attrs["n_unmatched"] == 1
    assert out["crime_count"].sum() == 1


def test_count_crimes_by_nearest_road_rejects_geographic_crs(projected_lines):
    lines_wgs = projected_lines.to_crs("EPSG:4326")
    crimes = gpd.GeoDataFrame({"geometry": [Point(0, 0)]}, crs="EPSG:4326")
    with pytest.raises(ValueError):
        HS.count_crimes_by_nearest_road(crimes, lines_wgs, max_dist=5)


# ---------------------------------------------------------------------------
# build_adj_graph / clusters_to_gdf -- the id-alignment regression
# ---------------------------------------------------------------------------

def test_build_adj_graph_basic_chain(projected_lines):
    G, df = HS.build_adj_graph(projected_lines, fid_col=None, crime_count_col="crime_count")
    assert G.number_of_nodes() == 4
    # a 4-segment chain has 3 touching pairs
    assert G.number_of_edges() == 3
    assert set(df["fid"]) == set(G.nodes)


def test_build_adj_graph_snap_tolerance_bridges_gap():
    # Two segments with a small real-world digitising gap between them --
    # exact touches() sees no adjacency; a small snap_tolerance should.
    gapped = gpd.GeoDataFrame(
        {
            "crime_count": [1, 1],
            "geometry": [LineString([(0, 0), (10, 0)]), LineString([(10.3, 0), (20, 0)])],
        },
        crs="EPSG:27700",
    )
    G0, _ = HS.build_adj_graph(gapped, crime_count_col="crime_count", snap_tolerance=0.0)
    assert G0.number_of_edges() == 0

    G1, _ = HS.build_adj_graph(gapped, crime_count_col="crime_count", snap_tolerance=0.5)
    assert G1.number_of_edges() == 1


def test_build_adj_graph_rejects_duplicate_fid_col(projected_lines):
    bad = projected_lines.copy()
    bad["seg_id"] = [1, 1, 2, 3]
    with pytest.raises(ValueError):
        HS.build_adj_graph(bad, fid_col="seg_id", crime_count_col="crime_count")


def test_clusters_to_gdf_survives_filtered_non_contiguous_index(projected_lines):
    """
    Regression test for the original bug: build_adj_graph used to generate
    node IDs from a positional reset index *internally* and never hand
    that frame back, so if the caller's DataFrame had a non-contiguous
    index (e.g. after filtering), clusters_to_gdf's `df.at[fid, ...]`
    lookup against the caller's *original* DataFrame silently returned
    the wrong row's geometry. Passing back (G, df) from build_adj_graph
    and requiring that same df in clusters_to_gdf removes the failure
    mode -- this test would have failed under the old implementation.
    """
    filtered = projected_lines.iloc[[1, 2, 3]].copy()  # index is now [1, 2, 3], not [0, 1, 2]
    assert list(filtered.index) != list(range(len(filtered)))

    G, df = HS.build_adj_graph(filtered, fid_col=None, crime_count_col="crime_count")
    clusters = HS.segment_clusters(G, min_size=1, max_size=10, min_crimes=0)
    gdf_clusters = HS.clusters_to_gdf(clusters, G, df, fid_col="fid")

    # Every geometry attached to a cluster must equal the geometry that
    # build_adj_graph's own returned df associates with that fid.
    df_by_fid = df.set_index("fid")
    for _, row in gdf_clusters.iterrows():
        assert row["geometry"].equals(df_by_fid.loc[row["fid"], "geometry"])


def test_clusters_to_gdf_raises_on_unknown_fid(projected_lines):
    G, df = HS.build_adj_graph(projected_lines, crime_count_col="crime_count")
    clusters = [{"cluster_id": 1, "nodes": {999}, "crime_sum": 1}]
    with pytest.raises(KeyError):
        HS.clusters_to_gdf(clusters, G, df, fid_col="fid")


def test_clusters_to_gdf_empty_clusters_returns_empty_gdf(projected_lines):
    G, df = HS.build_adj_graph(projected_lines, crime_count_col="crime_count")
    with pytest.warns(RuntimeWarning):
        out = HS.clusters_to_gdf([], G, df, fid_col="fid")
    assert out.empty
    assert list(out.columns) == ["cluster_id", "fid", "cluster_crime_sum", "crime_count", "geometry"]


# ---------------------------------------------------------------------------
# segment_clusters
# ---------------------------------------------------------------------------

def test_segment_clusters_empty_graph_raises():
    with pytest.raises(ValueError):
        HS.segment_clusters(nx.Graph())


def test_segment_clusters_min_greater_than_max_raises(projected_lines):
    G, _ = HS.build_adj_graph(projected_lines, crime_count_col="crime_count")
    with pytest.raises(ValueError):
        HS.segment_clusters(G, min_size=10, max_size=2)


# ---------------------------------------------------------------------------
# create_folium_map
# ---------------------------------------------------------------------------

def test_create_folium_map_requires_a_layer():
    with pytest.raises(ValueError):
        HS.create_folium_map()


def test_create_folium_map_basic(hexes, crimes):
    hex_both = HS.summarise_by_hex(hexes, crimes, count=True)
    m = HS.create_folium_map(hex_gdf=hex_both, hex_color_col="crime_count")
    assert m is not None


def test_create_folium_map_empty_query_warns_and_skips(hexes, crimes):
    hex_both = HS.summarise_by_hex(hexes, crimes, count=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        m = HS.create_folium_map(hex_gdf=hex_both, hex_query="crime_count > 999999")
    assert any("matched no rows" in str(w.message) for w in caught)
    assert m is not None
