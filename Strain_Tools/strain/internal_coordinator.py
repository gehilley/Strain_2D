"""
Driver program for strain calculation
"""
import importlib
from . import input_manager, output_manager


def get_model(model_name):
    """
    Return an instance of the model model_name.
    Dynamic module discovery.
    """
    module_name = 'Strain_2D.Strain_Tools.strain.models.strain_' + model_name.lower()
    model_module = importlib.import_module(module_name)
    obj = getattr(model_module, model_name)
    return obj


def strain_coordinator(MyParams):
    """
    Main function for strain computation
    """
    velField = input_manager.inputs(MyParams);
    strain_model = get_model(MyParams.strain_method);  # getting an object of type that inherits from Strain_2d
    constructed_object = strain_model(MyParams);   # calling the constructor, building strain model from our params
    [lons, lats, rot, exx, exy, eyy] = constructed_object.compute(velField);  # computing strain
    output_manager.outputs_2d(lons, lats, rot, exx, exy, eyy, MyParams, velField);  # 2D grid output format
    return
