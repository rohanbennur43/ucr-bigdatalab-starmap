import folium
import geopandas as gpd

gdf = gpd.read_file("./out_dir/tiles.geojson")
m = folium.Map(tiles="CartoDB positron")

# fit map to tiles
m.fit_bounds([[gdf.total_bounds[1], gdf.total_bounds[0]],
              [gdf.total_bounds[3], gdf.total_bounds[2]]])

# style: outline only
style = lambda _: {"fillOpacity": 0.05, "weight": 1}

# add tiles with popup = id
folium.GeoJson(
    gdf.to_json(),
    style_function=style,
    tooltip=folium.GeoJsonTooltip(fields=["id"])
).add_to(m)

m.save("tiles_map.html")
print("Open tiles_map.html in your browser")
