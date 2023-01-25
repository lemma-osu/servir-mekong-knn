from datetime import datetime, timezone
import ee
from models import Config

ee.Initialize()


def now() -> int:
    return int(datetime.now(tz=timezone.utc).timestamp() * 1000)


def _filter_by_year(
    collection: ee.ImageCollection, year: int
) -> ee.ImageCollection:
    collection = ee.ImageCollection(collection)
    year = ee.Number(year)
    start_date = ee.Date.fromYMD(year, 1, 1)
    end_date = ee.Date.fromYMD(year.add(1), 1, 1)
    return collection.filterDate(start_date, end_date)


def _get_random_idx(
    collection: ee.ImageCollection, seed: int = None
) -> ee.Number:
    collection = ee.ImageCollection(collection)
    seed = ee.Number(seed or now())
    lst = ee.List.sequence(0, collection.size().subtract(1)).shuffle(seed)
    return ee.Number(lst.get(0))


def _get_image_at_idx(collection: ee.ImageCollection, idx: int) -> ee.Image:
    return ee.Image(collection.toList(1, idx).get(0))


class RealizationCollection:
    def __init__(self, collection: set, name: str):
        self.collection = ee.ImageCollection(collection)
        self.name = name

    def filter_by_year(self, year: int) -> ee.ImageCollection:
        return _filter_by_year(self.collection, year)

    def random_image_from_year(self, year: int) -> ee.Image:
        fltr_coll = _filter_by_year(self.collection, year)
        idx = _get_random_idx(fltr_coll)
        img = _get_image_at_idx(fltr_coll, idx).rename(self.name)
        d = ee.Dictionary(
            {
                "id": img.get("system:id"),
                "idx": idx,
            }
        )
        return img.set({"src": d})

    def static_image_from_year(self, year: int) -> ee.Image:
        img = _filter_by_year(self.collection, year).first().rename(self.name)
        d = ee.Dictionary(
            {
                "id": img.get("system:id"),
                "idx": 0,
            }
        )
        return img.set({"src": d})


class UncertaintyCollection:
    def __init__(self, collection: str, uncertainty_collection: str, name: str):
        self.collection = ee.ImageCollection(collection)
        self.uncertainty_collection = ee.ImageCollection(uncertainty_collection)
        self.name = name

    def filter_collection_by_year(self, year: int) -> ee.ImageCollection:
        return _filter_by_year(self.collection, year)

    def filter_uncertainty_collection_by_year(
        self, year: int
    ) -> ee.ImageCollection:
        return _filter_by_year(self.uncertainty_collection, year)

    def random_image_from_year(self, year: int, seed: int = None) -> ee.Image:
        seed = ee.Number(seed or now())
        img = self.filter_collection_by_year(year).first()
        uncertainty_img = self.filter_uncertainty_collection_by_year(
            year
        ).first()
        rand_img = ee.Image.random(seed, "normal")
        out_img = (
            img.add(uncertainty_img.multiply(rand_img))
            .clamp(0, 100)
            .rename(self.name)
        )
        d = ee.Dictionary(
            {
                "id": img.get("system:id"),
                "uncertainty_id": uncertainty_img.get("system:id"),
            }
        )
        return out_img.set({"src": d})

    def static_image_from_year(self, year: int) -> ee.Image:
        out_img = self.filter_collection_by_year(year).first().rename(self.name)
        d = ee.Dictionary(
            {
                "id": out_img.get("system:id"),
                "uncertainty_id": None,
            }
        )
        return out_img.set({"src": d})


class YearlyCollection:
    def __init__(self, collection: str, name: str):
        self.collection = ee.ImageCollection(collection)
        self.name = name

    def filter_by_year(self, year: int) -> ee.ImageCollection:
        return _filter_by_year(self.collection, year)

    def static_image_from_year(self, year: int) -> ee.Image:
        img = _filter_by_year(self.collection, year).first().rename(self.name)
        d = ee.Dictionary({"id": img.get("system:id")})
        return img.set({"src": d})


class CovariateCombinationFactory:
    def __init__(self, config: Config, with_location: bool = False):
        self.config = config
        self.with_location = with_location

    def random_image_for_year(self, year: int) -> ee.Image:
        return self.to_bands(self.get_covariate_images(year, "random"))

    def static_image_for_year(self, year: int) -> ee.Image:
        return self.to_bands(self.get_covariate_images(year, "static"))

    def to_bands(self, collection: ee.ImageCollection) -> ee.Image:
        band_names = collection.toList(collection.size()).map(
            lambda img: (ee.Image(img).bandNames().get(0))
        )
        band_srcs = collection.toList(collection.size()).map(
            lambda img: (ee.Image(img).get("src"))
        )
        image = (
            collection.toBands()
            .rename(band_names)
            .set({"band_sources": band_srcs})
        )

        # Handle location bands if requested
        location_src = ee.List.repeat({"id": "ee.Image.pixelLonLat"}, 2)
        if self.with_location:
            return image.addBands(ee.Image.pixelLonLat()).set(
                {
                    "band_sources": ee.List(image.get("band_sources")).cat(
                        location_src
                    )
                }
            )
        return image

    def get_covariate_images(
        self, year: int, method: str
    ) -> ee.ImageCollection:
        covariates = self.config.covariates

        def get_realization_image(obj):
            c = RealizationCollection(obj.collection, obj.name)
            if method == "random":
                return c.random_image_from_year(year)
            return c.static_image_from_year(year)

        realization_coll = list(
            map(get_realization_image, covariates.realization_collections)
        )

        def get_uncertainty_image(obj):
            c = UncertaintyCollection(
                obj.collection, obj.uncertainty_collection, obj.name
            )
            if method == "random":
                return c.random_image_from_year(year)
            return c.static_image_from_year(year)

        uncertainty_coll = list(
            map(get_uncertainty_image, covariates.uncertainty_collections)
        )

        def get_yearly_image(obj):
            c = YearlyCollection(obj.collection, obj.name)
            return c.static_image_from_year(year)

        yearly_coll = list(map(get_yearly_image, covariates.yearly_collections))

        def get_static_image(obj):
            img = ee.Image(obj.image).rename(obj.name)
            d = ee.Dictionary(
                {
                    "id": img.get("system:id"),
                }
            )
            return img.set({"src": d})

        static_coll = list(map(get_static_image, covariates.static_images))

        # Merge and return
        return (
            ee.ImageCollection(realization_coll)
            .merge(ee.ImageCollection(uncertainty_coll))
            .merge(ee.ImageCollection(yearly_coll))
            .merge(ee.ImageCollection(static_coll))
        )


class Covariates:
    def __init__(self, config: Config):
        self.config = config

    def _sequential_client_array(self, num: int) -> ee.List:
        return ee.List(list(range(num)))

    def get_realizations_for_year(
        self, year: int, num: int, return_random: bool = True
    ) -> ee.ImageCollection:
        def get_realization_for_year(idx: int) -> ee.Image:
            idx = ee.Number(idx)
            ccf = CovariateCombinationFactory(self.config, True)
            img = (
                ccf.random_image_for_year(year)
                if return_random
                else ccf.static_image_for_year(year)
            )
            return img.set({"year": year, "realization": idx.add(1)})

        return ee.ImageCollection(
            self._sequential_client_array(num).map(get_realization_for_year)
        )


if __name__ == "__main__":
    import json
    from pprint import pprint

    with open(
        "D:/code/gee-repos/python/servir-mekong-knn/src/servir_mekong_knn/"
        "examples/config-training.json"
    ) as fh:
        config = Config.parse_obj(json.load(fh))
        foo = Covariates(config)
        first_realization = foo.get_realizations_for_year(2017, 1).first()
        pprint(first_realization.get("band_sources").getInfo())
