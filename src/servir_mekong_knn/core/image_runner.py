import ee

from models import Config
from covariates import Covariates
from plots import Plots, PlotsWithCovariates
from wrapper import Wrapper

ee.Initialize()


class ImageRunner:
    def __init__(self, config: Config) -> None:
        self.config = config

    def get_covariates(self, model_year: int) -> ee.ImageCollection:
        return Covariates(self.config).get_realizations_for_year(
            model_year, self.config.p, "random"
        )

    def get_plots(self) -> Plots:
        return Plots(self.config)

    def get_plots_with_covariates(self) -> PlotsWithCovariates:
        return PlotsWithCovariates(self.get_plots(), Covariates(self.config))

    def get_wrapper(self, model_year: int) -> Wrapper:
        return Wrapper(
            self.get_plots_with_covariates(),
            self.get_covariates(model_year),
            self.config.methods,
            self.config.k,
        )

    def run_models(self) -> ee.ImageCollection:
        coll = ee.FeatureCollection(
            list(
                map(
                    lambda model_year: self.get_wrapper(
                        model_year
                    ).run_models(),
                    self.config.model_years,
                )
            )
        )
        return ee.ImageCollection(coll).flatten()

    # Need to fix this for multiyear
    # def run_models_fc(self) -> ee.List:
    #     coll = ee.FeatureCollection(self.config.model_years.map(
    #         lambda model_year: self.get_wrapper(model_year).run_models_fc()
    #     ))

    def export_models(self, region: ee.Geometry):
        nn_coll = self.run_models()
        idx = 0
        for year in self.config.model_years:
            for p in range(1, self.config.p + 1):
                for method in self.config.methods:
                    method = method.lower()
                    img = ee.Image(nn_coll.toList(1, idx).get(0)).clip(region)
                    name = f"{method}_realization{p}_{year}"
                    task = ee.batch.Export.image.toAsset(
                        image=img,
                        description=name,
                        assetId=f"{self.config.output_collection}/{name}",
                        region=region,
                        scale=30,
                        pyramidingPolicy={".default": "mode"},
                        maxPixels=1e13,
                    )
                    task.start()
                    idx += 1


if __name__ == "__main__":
    import sys
    import json
    from models import Config

    # with open(
    #     "D:/code/gee-repos/python/servir-mekong-knn/src/servir_mekong_knn/"
    #     "examples/config-training.json"
    # ) as fh:

    with open(sys.argv[1]) as fh:
        config = Config.parse_obj(json.load(fh))
        runner = ImageRunner(config)
        rect = ee.Geometry.Rectangle([104.528, 12.235, 104.554, 12.262])
        runner.export_models(rect)
