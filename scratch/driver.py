import holoviews as hv
import numpy as np

ds = hv.Dataset({"a": [1, 2, 3], "b": ["A", "BB", "CCC"]})
print(ds, ds.kdims, ds.vdims, ds.interface.driver)
# hv.Dataset({"a": [1, 2, 3], "b": ["A", "BB", "CCC"]}, kdims=["a"], vdims=["b"])

# image
img = hv.Image(np.array([[1, 2, 3], [4, 5, 6]]))
print(img, img.kdims, img.vdims, img.interface, img.interface.driver)

# Polygon
def rectangle(x=0, y=0, width=.05, height=.05):
    return np.array([(x,y), (x+width, y), (x+width, y+height), (x, y+height)])

polys = hv.Polygons([{('x', 'y'): rectangle(x, y), 'level': z}
                     for x, y, z in np.random.rand(100, 3)], vdims='level')

print(polys, polys.kdims, polys.vdims, polys.interface, polys.interface.driver)
