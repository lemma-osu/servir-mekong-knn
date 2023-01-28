from typing import List

from pydantic import BaseModel


# TODO: We have Collection models here and then again
# TODO: in covariates.py.  Possibly combine the functionality. I'm
# TODO: unclear how "clean" we should keep the pydantic models


class RealizationCollection(BaseModel):
    name: str
    collection: str


class UncertaintyCollection(BaseModel):
    name: str
    collection: str
    uncertainty_collection: str


class YearlyCollection(BaseModel):
    name: str
    collection: str


class StaticImage(BaseModel):
    name: str
    image: str


class Covariates(BaseModel):
    realization_collections: List[RealizationCollection]
    uncertainty_collections: List[UncertaintyCollection]
    yearly_collections: List[YearlyCollection]
    static_images: List[StaticImage]


class Config(BaseModel):
    k: int
    p: int
    methods: List[str]
    covariates: Covariates
    species_fields: List[str]
    categorical_fields: List[str]
    plots: str
    model_years: List[int]
    nn_id_field: str
    year_field: str
    output_collection: str
    output_fc_collection: str
    output_accuracy_statistics: str
