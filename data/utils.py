import h5py 
import numpy as np
from functools import lru_cache
from data.HighLevelFeatures import HighLevelFeatures

def read_hdf5(data_path, fields: list[str], indices=None):
    with h5py.File(data_path, 'r') as f:
        if indices is None:
            data = [f[field][:].astype(np.float32) for field in fields]
        else:
            data = [f[field][indices].astype(np.float32) for field in fields]
    return data

def lookup_hdf5(data_path, field: str="showers"):
    with h5py.File(data_path, 'r') as f:
        num_entries = f[field].shape[0]
    return num_entries

def scale_shower(X, cond):

    cond = cond.reshape(-1, 1)
    X /= cond
    cond /= 1000. # convert to GeV

    return X, cond

@lru_cache(maxsize=8)
def _get_hlf_extractor(particle: str, xml_path: str) -> HighLevelFeatures:
    return HighLevelFeatures(particle, xml_path)

def get_high_level_features(X, cond, xml_path, particle):
    hlf = _get_hlf_extractor(particle, xml_path)
    setattr(hlf, "Einc", cond)
    hlf.CalculateFeatures(X)
    hlf_features = [
        hlf.GetEtot(),
        *hlf.GetElayers().values(),
        *hlf.GetECEtas().values(),
        *hlf.GetECPhis().values(),
        *hlf.GetWidthEtas().values(),
        *hlf.GetWidthPhis().values(),
        # *hlf.GetSparsity().values()
    ]

    hlf_features = np.stack(hlf_features, axis=1, dtype=np.float32)

    return hlf_features

def prepare_gen_latent(cond, X_dim):
    num_entries = cond.shape[0]
    gen_latent = np.random.normal(size=(num_entries, X_dim)).astype(np.float32)
    return gen_latent

