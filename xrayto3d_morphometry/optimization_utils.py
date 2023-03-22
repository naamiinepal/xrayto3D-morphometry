import vedo
import cma
import sys
import numpy as np
from typing import Sequence,Tuple

def get_cross_section_area(mesh_obj:vedo.Mesh,plane_origin: Sequence[float],plane_normal:Sequence[float]):
    sliced_mesh = mesh_obj.clone().cut_with_plane(origin=plane_origin, normal=plane_normal)
    return sliced_mesh.boundaries().triangulate().area()

def cma_es_search_candidate_cut_plane(mesh_obj: vedo.Mesh, init_plane_origin: Sequence[float],init_plane_normal: Sequence[float],init_std=1.0,verbose=False)->cma.CMAEvolutionStrategy:
    es = cma.CMAEvolutionStrategy(np.asarray(init_plane_normal),init_std,{'bounds': [-1.5,1.5 ]})
    es.opts.set({'tolfunhist': 1e-1})# stop when the cross section area does not reduce by 0.1

    while not es.stop():
        solutions = es.ask()
        loss_values = [ get_cross_section_area(mesh_obj,
                                               plane_normal=tuple(x),
                                               plane_origin=init_plane_origin)
                       for x in solutions]
        es.tell(solutions, loss_values)
        if verbose:
            es.disp()
    if verbose:
        es.result_pretty()
    return es


def grid_search_candidate_cut_plane(mesh_obj: vedo.Mesh, init_plane_origin:Sequence[float],init_plane_normal:Sequence[float],num_cuts=5,range_min=-0.5,range_max=0.5,verbose=False)->Tuple[Sequence[float],float]:
    """grid search through candidate cut plane normals at given position to find the one with smallest cross-sectional area"""
    best_cut_plane_normal = tuple()
    smallest_csarea = sys.float_info.max
    for incr_k in np.linspace(range_min,range_max,num_cuts):
        for incr_i in np.linspace(range_min,range_max,num_cuts):
            for incr_j in np.linspace(range_min,range_max,num_cuts):
                candidate_cut_plane_normal =  tuple( v+inc for v,inc in zip(init_plane_normal,(incr_i,incr_j,incr_k)))
                csa = get_cross_section_area(mesh_obj,
                                             plane_normal=candidate_cut_plane_normal,
                                             plane_origin=init_plane_origin)
                if csa <= smallest_csarea:
                    # additional sanity check: is the cut plane actually circular
                    # Cross section consistency check:
                    #1. Circular fitting
                    # boundary_points = mesh_obj.clone().cut_with_plane(origin=init_plane_origin,normal=candidate_cut_plane_normal).boundaries().points()
                    # c,R,n = vedo.fit_circle(boundary_points)
                    
                    if verbose:
                        print(f'found better candidate with cs-area {csa:.3f}')
                    smallest_csarea = csa
                    best_cut_plane_normal = candidate_cut_plane_normal
    return best_cut_plane_normal,smallest_csarea