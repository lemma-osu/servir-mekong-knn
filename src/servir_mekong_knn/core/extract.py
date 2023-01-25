import ee

ee.Initialize()

from covariates import Covariates


def extract_signatures(
    fc: ee.FeatureCollection, img: ee.Image, radius: float, scale: float = 30.0
) -> ee.FeatureCollection:
    # Create footprints based on plot locations
    footprints = ee.FeatureCollection(fc).map(lambda f: f.buffer(radius))

    # Sample these locations from the environmental data
    return img.reduceRegions(
        collection=footprints,
        reducer=ee.Reducer.mean(),
        scale=scale,
        tileScale=16,
    )


def extract_matching_year_signatures(
    fc: ee.FeatureCollection,
    covariates: Covariates,
    year_attr: str,
    radius: float,
    scale: float = 30.0,
) -> ee.FeatureCollection:
    # Get all years represented
    years = fc.aggregate_array(year_attr).distinct().sort()

    # Get the mean covariate value associated with each year
    covariates_by_year = years.map(
        lambda year: covariates.get_realizations_for_year(year, 1, "mean")
    )

    # Zip these together and extract the spatial data on a year by year basis
    def process_year(t):
        t = ee.List(t)
        year = ee.Number(t.get(0))
        img = ee.ImageCollection(t.get(1)).first()
        sub_fc = fc.filter(ee.Filter.eq(year_attr, year))
        return extract_signatures(sub_fc, img, radius, scale)

    return ee.FeatureCollection(
        years.zip(covariates_by_year).map(process_year)
    ).flatten()


if __name__ == "__main__":
    import json
    from models import Config

    with open(
        "D:/code/gee-repos/python/servir-mekong-knn/src/servir_mekong_knn/"
        "examples/config-training.json"
    ) as fh:
        config = Config.parse_obj(json.load(fh))
        covariates = Covariates(config)

        # fc: ee.FeatureCollection,
        # covariates: Covariates,
        # year_attr: str,
        # radius: float,
        # scale: float = 30.0,

        fc = ee.FeatureCollection(config.plots)
        year_attr = config.year_field
        radius = 15.0
        foo = extract_matching_year_signatures(
            fc, covariates, year_attr, radius
        )
        print(foo.first().getInfo())
