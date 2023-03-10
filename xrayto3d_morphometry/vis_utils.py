import vedo
from typing import List

def get_custom_axes(center, axis1,axis2,axis3,scale=20) ->List[vedo.Mesh]:
    a = vedo.Arrow(center,center + axis1*scale,c='r')
    b = vedo.Arrow(center,center + axis2*scale,c='g')
    c = vedo.Arrow(center,center + axis3*scale,c='b')
    return [a,b,c]

def get_custom_axes_from_ellipsoid(ellipsoid:vedo.Ellipsoid)-> List[vedo.Mesh]:
    return get_custom_axes(ellipsoid.center,ellipsoid.axis1,
                           ellipsoid.axis2,
                           ellipsoid.axis3)