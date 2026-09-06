"""
Shared input-validation helpers for hexseg.

Every public function in `core.py` calls into these near the top, so that
bad input (wrong geometry type, missing CRS, geographic CRS where a
projected one is required, missing columns, empty data) raises a clear,
specific error immediately -- instead of surfacing later as an opaque
error from deep inside geopandas / shapely / h3, or worse, silently
producing wrong numbers.

These are intentionally simple and dependency-free (just geopandas).
"""

from __future__ import annotations

import geopandas as gpd


def require_geodataframe(gdf, name: str) -> None:
    """Check `gdf` is a non-empty GeoDataFrame."""
    if not isinstance(gdf, gpd.GeoDataFrame):
        raise TypeError(f"{name} must be a GeoDataFrame, got {type(gdf).__name__}")
    if gdf.empty:
        raise ValueError(f"{name} is empty")


def require_geom_type(gdf: gpd.GeoDataFrame, allowed, name: str) -> None:
    """Check `gdf` is a GeoDataFrame containing only geometry types in `allowed`.

    `allowed` is an iterable of shapely geom_type strings, e.g.
    {"Point", "MultiPoint"} or {"LineString", "MultiLineString"}.
    """
    require_geodataframe(gdf, name)
    if gdf.geometry.isna().any():
        n = int(gdf.geometry.isna().sum())
        raise ValueError(f"{name} contains {n} missing/null geometr{'y' if n == 1 else 'ies'}")
    allowed = set(allowed)
    present = set(gdf.geom_type.unique())
    bad = present - allowed
    if bad:
        raise TypeError(
            f"{name} must contain only {sorted(allowed)} geometries, "
            f"found {sorted(bad)}. If you have a mix, split or explode the "
            f"GeoDataFrame first."
        )


def require_crs(gdf: gpd.GeoDataFrame, name: str) -> None:
    """Check `gdf` has a CRS set at all."""
    require_geodataframe(gdf, name)
    if gdf.crs is None:
        raise ValueError(
            f"{name} has no CRS set. Set one with "
            f"`{name} = {name}.set_crs(<EPSG code>)` before calling this function "
            f"(or reproject with `.to_crs(...)` if you know the correct EPSG code)."
        )


def require_projected_crs(gdf: gpd.GeoDataFrame, name: str) -> None:
    """Check `gdf` is in a projected (metric) CRS, not lat/lon degrees.

    Several functions measure distance directly in the GeoDataFrame's CRS
    units (e.g. `max_dist` in metres, or nearest-neighbour centroid
    distances). Passing geographic (WGS84 lat/lon) data into those silently
    produces meaningless results rather than an error, which is worse than
    failing loudly here.
    """
    require_crs(gdf, name)
    if gdf.crs.is_geographic:
        raise ValueError(
            f"{name} is in a geographic CRS ({gdf.crs.name}), but this function "
            f"measures distance in the CRS's linear units and needs a projected "
            f"(metric) CRS. Reproject first, e.g. "
            f"`{name} = {name}.to_crs(<local projected EPSG code>)` "
            f"(e.g. a national grid or UTM zone)."
        )


def require_column(gdf, col: str, name: str) -> None:
    """Check `col` exists in `gdf`, with a helpful message listing what is there."""
    if col not in gdf.columns:
        raise KeyError(
            f"'{col}' not found in {name}. Available columns: {list(gdf.columns)}"
        )


def require_min_rows(gdf, n: int, name: str) -> None:
    """Check `gdf` has at least `n` rows."""
    if len(gdf) < n:
        raise ValueError(f"{name} must have at least {n} row(s), got {len(gdf)}")
