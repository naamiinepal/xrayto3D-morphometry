import vedo
from typing import List

def get_oriented_camera(mesh_obj:vedo.Mesh,axis,camera_dist=200):
    """return a camera dict with 
    focal point: mesh_obj center of mass
    """
    x0,x1,y0,y1,z0,z1 = mesh_obj.bounds()
    focal_point = mesh_obj.center_of_mass()
    
    position = list(focal_point).copy()
    position[axis] += camera_dist
    
    distance = camera_dist
    clipping_range = (x0,x1) if axis == 0 else (y0,y1) if axis == 1 else (z0,z1) 
    viewup = (0,1,0) if axis == 2 else (0,0,1) if axis == 1 else (0,0,1)
    return {
        'position':position,
        'focal_point':focal_point,
        'viewup':viewup,
        'distance':distance,
        'clipping_range':clipping_range
    }


def get_custom_axes(center, axis1,axis2,axis3,scale=20) ->List[vedo.Mesh]:
    a = vedo.Arrow(center,center + axis1*scale,c='r')
    b = vedo.Arrow(center,center + axis2*scale,c='g')
    c = vedo.Arrow(center,center + axis3*scale,c='b')
    return [a,b,c]

def get_custom_axes_from_ellipsoid(ellipsoid:vedo.Ellipsoid)-> List[vedo.Mesh]:
    return get_custom_axes(ellipsoid.center,ellipsoid.axis1,
                           ellipsoid.axis2,
                           ellipsoid.axis3)