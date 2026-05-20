import glob as _glob_module
_orig_glob = _glob_module.glob
def _normalized_glob(*args, **kwargs):
    return [p.replace('\\', '/') for p in _orig_glob(*args, **kwargs)]
_glob_module.glob = _normalized_glob
