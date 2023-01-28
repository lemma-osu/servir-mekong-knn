from typing import Any, List

import ee
from geeknn import ordination
from geeknn.ordination.utils import Colocation

from plots import Plots, PlotsWithCovariates

ee.Initialize()


# Create a dictionary of model keyword to instantiation
ORDINATION_METHODS = {
    "RAW": ordination.Raw,
    "EUC": ordination.Euclidean,
    "MAH": ordination.Mahalanobis,
    "MSN": ordination.MSN,
    "GNN": ordination.GNN,
}


class Wrapper:
    def __init__(
        self,
        plots: PlotsWithCovariates,
        covariate_coll: ee.ImageCollection,
        methods: List[str],
        k: int,
    ):
        self.plots = plots
        self.covariate_coll = covariate_coll
        self.methods = methods
        self.k = k
        self.p = ee.Number(covariate_coll.size())

    def get_model(self, method: str) -> Any:
        if method == "GNN":
            return ORDINATION_METHODS[method](
                k=self.k, spp_transform="NONE", num_cca_axes=8
            )
        else:
            return ORDINATION_METHODS[method](k=self.k)

    def create_colocation_object(self) -> Colocation:
        # Create a colocation object to filter out self-assignment
        # TODO: Consider whether this should be an argument
        colocation_fc = self.plots.fc.map(
            lambda f: f.set("LOC_ID", f.get(self.plots.nn_id_field))
        )
        return Colocation(
            fc=colocation_fc,
            location_field="LOC_ID",
            plot_field=self.plots.nn_id_field,
        )

    def train_model(
        self, model: str, env_fc: ee.FeatureCollection, env_columns: List[str]
    ) -> Any:
        spp_columns = self.plots.species_fields
        return model.train(
            fc=env_fc,
            id_field=self.plots.nn_id_field,
            spp_columns=spp_columns,
            env_columns=env_columns,
        )

    def run_model(
        self, model: str, env_fc: ee.FeatureCollection, img: ee.Image
    ) -> ee.Image:
        trained = self.train_model(model, env_fc, img.bandNames())
        return trained.predict(img)

    def run_realization(self, img: ee.Image) -> ee.ImageCollection:
        def run_method(method):
            model = self.get_model(method)
            return (
                self.run_model(model, self.plots.env_fc, img)
                .copyProperties(img)
                .set("ordination_method", method)
            )

        return ee.ImageCollection(list(map(run_method, self.methods)))

    def run_model_fc(
        self,
        model: str,
        env_fc: ee.FeatureCollection,
        img: ee.Image,
        colocation_obj: Colocation = None,
    ) -> Any: # want ee.Array
        trained = self.train_model(model, env_fc, img.bandNames())
        return trained.predict_fc(
            fc=env_fc,
            colocation_obj=colocation_obj,
        )

    def run_realization_fc(
        self, img: ee.Image, colocation_obj: Colocation = None
    ) -> ee.List:
        # TODO: There may be plots that have null signatures
        # after running extract.  We need to drop these plots before
        # running models.  Is this the best place to do this? Also,
        # right now, I'm only testing the first band - need to do this
        # with all bands.  We probably need to have a "pre-flight".
        bn = img.bandNames()
        env_fc = self.plots.env_fc.filter(
            ee.Filter.neq(ee.String(bn.get(0)), None)
        )

        def run_method(method: str) -> ee.FeatureCollection:
            model = self.get_model(method)
            return self.run_model_fc(model, env_fc, img, colocation_obj)

        return ee.List(list(map(run_method, self.methods)))

    def run_first_model(self) -> ee.ImageCollection:
        first = self.covariate_coll.first()
        return self.run_realization(first).copyProperties(first)

    def run_models(self) -> ee.ImageCollection:
        return ee.ImageCollection(
            self.covariate_coll.map(lambda img: self.run_realization(img))
        ).flatten()

    def run_first_model_fc(self) -> ee.List:
        colocation_obj = self.create_colocation_object()
        img = self.covariate_coll.first()
        return self.run_realization_fc(img, colocation_obj)

    def run_models_fc(self) -> ee.List:
        colocation_obj = self.create_colocation_object()
        realizations = ee.List.sequence(
            0, self.covariate_coll.size().subtract(1)
        )

        def run_realization(idx) -> ee.List:
            img = ee.Image(self.covariate_coll.toList(1, idx).get(0))
            return self.run_realization_fc(img, colocation_obj)

        return realizations.map(run_realization)
