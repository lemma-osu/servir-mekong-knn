import ee
from models import Config
from covariates import Covariates
from extract import extract_matching_year_signatures

ee.Initialize()


class Plots:
    def __init__(self, config: Config):
        self.fc = ee.FeatureCollection(config.plots)
        self.nn_id_field = config.nn_id_field
        self.year_field = config.year_field
        self.species_fields = config.species_fields


class PlotsWithCovariates:
    def __init__(self, plots: Plots, covariates: Covariates):
        self.fc = plots.fc
        self.nn_id_field = plots.nn_id_field
        self.year_field = plots.year_field
        self.species_fields = plots.species_fields
        self.env_fc = self.match_by_year(covariates)

    def match_by_year(self, covariates: Covariates) -> ee.FeatureCollection:
        return extract_matching_year_signatures(
            self.fc, covariates, self.year_field, 15.0
        )
