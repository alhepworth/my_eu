POSTCODES_OUTPUTS := \
  postcodes/output/postcode-area-boundaries.geo.json \
	postcodes/output/postcode-area-boundaries-simplified.geo.json
OUTPUTS += $(POSTCODES_OUTPUTS)

postcodes/output/postcode-area-boundaries.geo.json: postcodes/input/postcode-area-boundaries.kml
	node utilities/kml_to_geojson.js <$< >$@

postcodes/output/postcode-area-boundaries-simplified.geo.json: postcodes/output/postcode-area-boundaries.geo.json
	node_modules/.bin/mapshaper -i $< -simplify 0.45 -drop fields="styleUrl,styleHash,styleMapHash,stroke,stroke-opacity,stroke-width,fill-opacity" -o precision=0.0001 $@

postcodes/output/postcode-area-boundaries-centroids.geo.json: postcodes/output/postcode-area-boundaries.geo.json
	node_modules/.bin/mapshaper -i $< -points centroid -drop fields="styleUrl,styleHash,styleMapHash,stroke,stroke-opacity,stroke-width,fill-opacity" -o precision=0.0001 $@
