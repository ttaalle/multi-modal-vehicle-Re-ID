# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
#from .cuhk03 import CUHK03
from .dukemtmcreid import DukeMTMCreID
from .market1501 import Market1501
from .msmt17 import MSMT17
from .dataset_loader import ImageDataset
from .veri import Veri
from .vehicleid import Vehicleid
from .cityflow import Cityflow
from .irour    import Irour
from .rgbour    import Rgbour
from .rgbir    import Rgbir

__factory = {
    'market1501': Market1501,
  #  'cuhk03': CUHK03,
    'dukemtmc': DukeMTMCreID,
    'msmt17': MSMT17,
    'veri': Veri,
    'vehicleid': Vehicleid,
    'cityflow': Cityflow,
    'irour':Irour,
    'rgbour':Rgbour,
    'rgbir':Rgbir,
}


def get_names():
    return __factory.keys()


def init_dataset(name, *args, **kwargs):
    if name not in __factory.keys():
        raise KeyError("Unknown datasets: {}".format(name))
    return __factory[name](*args, **kwargs)
