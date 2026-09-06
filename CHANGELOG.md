# Changelog

## 0.2.0

### Fixed
- **`build_adj_graph` / `clusters_to_gdf` id-alignment bug.** When `fid_col=None`, `build_adj_graph` generated node IDs from a positional reset index *internal* to the function and never returned that frame. If the caller's own DataFrame didn't already have a clean `0..n-1` index (e.g. after any filtering), `clusters_to_gdf`'s `df.at[fid, 'geometry']` lookup against the caller's *original* DataFrame silently returned the wrong row's geometry -- crime counts could end up mapped to the wrong street segment with no error raised. `build_adj_graph` now returns `(G, df)`, and `clusters_to_gdf` requires that same `df`, removing the failure mode. **This is a breaking change to `build_adj_graph`'s return signature** -- see Migration below.

### Added
- Input validation across every public function: geometry type, CRS presence, projected-vs-geographic CRS (for anything that measures distance in CRS units), required columns, and empty/duplicate-ID checks. Errors are now specific (`TypeError`/`ValueError`/`KeyError`) and point at what to fix, rather than surfacing later as an opaque error inside geopandas/shapely/h3.
- `build_adj_graph(..., snap_tolerance=...)`: an optional distance tolerance (in the CRS's linear units) for treating nearby-but-not-quite-touching segments as adjacent. Real-world, commercially-sourced road centreline data often has small digitising gaps/overshoots at junctions that exact `touches()` misses. Default `0.0` reproduces the old exact-touching-only behaviour.
- `count_crimes_by_nearest_road` and `summarise_by_hex` now warn when points are silently excluded (too far from any road / outside the hex coverage area / on a hex boundary edge), and `count_crimes_by_nearest_road`'s return value carries `.attrs['n_unmatched']` / `.attrs['n_total']` diagnostics. Points tied for equally-nearest between two road segments are also matched to just one, rather than double-counted.
- `get_hexagons` now repairs topologically invalid input polygons (self-intersections etc., common in real boundary data) automatically, with a warning, instead of failing or silently mis-tiling.
- `create_folium_map(..., hex_color_col=..., seg_color_col=...)`: optional choropleth colouring of a layer by any numeric column (e.g. rank or z-score), plus hover tooltips showing each feature's attributes on every layer. A query that matches zero rows now skips that layer with a warning instead of crashing the whole map.
- A real pytest suite (`tests/test_core.py`) covering all nine public functions, including a regression test for the id-alignment bug above.
- `hexseg.__version__`.

### Changed
- **`add_spatial_lag` default behaviour**: now uses true H3 grid adjacency (`method='h3'`, default `k=1` ring) instead of approximate k-nearest-centroid lookup. This is what the original docstring already claimed ("typically a 'donut' around hexagons") but did not actually implement -- the old KNN approach could treat geometrically-nearby-but-topologically-unrelated hexes (e.g. across a gap between two districts) as neighbours. `method='h3'` is purely topological and needs no CRS/units assumption at all. Pass `method='knn', k=6` to reproduce the exact 0.1.x default.
- `summarise_by_hex(count_col='any', ...)` deprecated in favour of `summarise_by_hex(count=True, ...)`. The old parameter was a string whose *value* was ignored (only `is not None` mattered), which was confusing; `count_col` still works but raises a `DeprecationWarning`.

### Migration from 0.1.x
```python
# before
G = HS.build_adj_graph(roads_with_counts, fid_col=None, crime_count_col='crime_count')
gdf_clusters = HS.clusters_to_gdf(clusters, G, roads_with_counts, fid_col='fid')

# after
G, roads_indexed = HS.build_adj_graph(roads_with_counts, fid_col=None, crime_count_col='crime_count')
gdf_clusters = HS.clusters_to_gdf(clusters, G, roads_indexed, fid_col='fid')
```
```python
# before
hex_lagged = HS.add_spatial_lag(hex_both, count_col='crime_count', weight_col='crime_weight', k=6)

# after (recommended: true adjacency)
hex_lagged = HS.add_spatial_lag(hex_both, count_col='crime_count', weight_col='crime_weight')

# after (exact old behaviour, if you specifically want it)
hex_lagged = HS.add_spatial_lag(hex_both, count_col='crime_count', weight_col='crime_weight', method='knn', k=6)
```
```python
# before
hex_both = HS.summarise_by_hex(hexes, crimes, count_col='any', weight_col='pseudo_harm')

# after
hex_both = HS.summarise_by_hex(hexes, crimes, count=True, weight_col='pseudo_harm')
```

## 0.1.3 and earlier
See PyPI release history.
