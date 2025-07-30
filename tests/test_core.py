import geopandas as gpd
from shapely.geometry import Polygon
from hexseg import get_hexagons

def test_get_hexagons_empty():
    # minimal GeoDataFrame with a single small polygon
    gdf = gpd.GeoDataFrame({
        'district': [1],
        'geometry': [Polygon([(0,0),(0,1),(1,1),(1,0)])]
    }, crs="EPSG:4326")
    hexes = get_hexagons(gdf, name_col="district", resolution=7)
    assert "hex_id" in hexes.columns
    assert not hexes.empty
