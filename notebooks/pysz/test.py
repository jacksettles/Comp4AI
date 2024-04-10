import numpy as np
from pathlib import Path
from pysz import SZ
import sys

# prepare your data in numpy array format
#HOME = str(Path.home())
HOME="/lustre/orion/csc143/proj-shared/jwang/Comp4AI"
print(HOME)
data = np.fromfile(HOME + '/data/baryon_density.f32', dtype=np.float32)
data = np.reshape(data, (512, 512, 512))

# init SZ (both SZ2 and SZ3 are supported)
# Please change the path to the SZ dynamic library file in your system
lib_extention = {
    "darwin": "libSZ3c.dylib",
    "windows": "SZ3c.dll",
}.get(sys.platform, "libSZ3c.so")

sz = SZ("/lustre/orion/csc143/proj-shared/jwang/Comp4AI/ExternalDependencies/SZ3/install/lib64/{}".format(lib_extention))

# compress, both input and output data are numpy array
data_cmpr, cmpr_ratio = sz.compress(data, 0, 1e-3, 0, 0)
print("compression ratio = {:5G}".format(cmpr_ratio))

# decompress, both input and output data are numpy array
data_dec = sz.decompress(data_cmpr, data.shape, data.dtype)

# verify
sz.verify(data, data_dec)
