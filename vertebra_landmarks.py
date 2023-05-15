from typing import Sequence
from pathlib import Path
import vedo
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.linear_model import QuantileRegressor
from xrayto3d_morphometry import (
    get_mesh_from_segmentation,
    move_to_origin,
    align_along_principal_axes,
    get_principal_axis,
    add_tuple, subtract_tuple, multiply_tuple_scalar,
    get_closest_points_between_point_clouds,
    brute_force_search_get_closest_points_between_point_clouds,
    get_vector_from_points,
    get_angle_between_vectors,
    get_farthest_point_from_line_segment,
    project_points_onto_line,
    get_distance_between_points,
    get_distance2_between_points,
    lerp,
    get_points_along_directions,
    get_oriented_camera
    
)

def fit_l1(data, alpha=0.2):
    """fit a L1 regularized linear model
    data: (N,3) where N is number of data points
    alpha: regularization strength
    """
    quantile_reg = QuantileRegressor(solver='interior-point', alpha=alpha).fit(data[:, :2], data[:, 2])
    fitted_data = []
    for datum in data:
        x, y, z = datum
        z_new, = quantile_reg.predict([[x, y],]).flatten().tolist()
        fitted_data.append([x, y, z_new])
    return fitted_data


def get_axis_lines(origin, boundary_points, scale=40):
    vb_axes = vedo.pca_ellipsoid(boundary_points)
    # print(vb_axes.va, vb_axes.vb)
    vb_ax1_p0 = add_tuple(tuple(origin), multiply_tuple_scalar(tuple(vb_axes.axis1), scale))
    vb_ax1_p1 = subtract_tuple(tuple(origin), multiply_tuple_scalar(tuple(vb_axes.axis1), scale))
    vb_ax2_p0 = add_tuple(tuple(origin), multiply_tuple_scalar(tuple(vb_axes.axis2), scale))
    vb_ax2_p1 = subtract_tuple(tuple(origin), multiply_tuple_scalar(tuple(vb_axes.axis2), scale))
    axis1_line = (vb_ax1_p0, vb_ax1_p1)
    axis2_line = (vb_ax2_p0, vb_ax2_p1)
    return axis1_line, axis2_line


def get_slope_intercept_from_two_points_z_y(p0: Sequence[float], p1: Sequence[float]):
    "z is the independent dimension, y is the dependent dimension"
    p0_x, p0_y, p0_z = p0
    p1_x, p1_y, p1_z = p1
    # y = mz+c, m = y2 - y1 / z2 - z1, c = y1 - m*z1
    m = (p1_y - p0_y) / (p1_z - p0_z)
    c = p1_y - m * p1_z
    return m, c


def get_slope_intercept_from_two_points_y_z(p0: Sequence[float], p1: Sequence[float]):
    "y is the independent dimension, z is the dependent dimension"
    p0_x, p0_y, p0_z = p0
    p1_x, p1_y, p1_z = p1
    # z = my+c, m = z2 - z1 / y2 - y1, c = z1 - m*y1
    m = (p1_z - p0_z) / (p1_y - p0_y)
    c = p1_z - m * p1_y
    return m, c

def get_symmetry_plane(vert_mesh):
    mirrored_vert_mesh = vert_mesh.clone(deep=True, transformed=True).mirror('x')
    mirrored_vert_points = vedo.Points(mirrored_vert_mesh.points())
    vert_mesh_points = vedo.Points(vert_mesh.clone(deep=True, transformed=True).points())
    aligned_pts1 = mirrored_vert_points.clone().align_to(vert_mesh_points, invert=False)

    # draw arrows to see where points end up
    rand_idx = np.random.randint(0, len(vert_mesh.points()), 100)
    sampled_vmp = vert_mesh.points()[rand_idx]
    sampled_apts1 = aligned_pts1.points()[rand_idx]
    avg_points = [lerp(a, b, 0.5) for a, b in zip(sampled_vmp, sampled_apts1)]
    sym_plane = vedo.fit_plane(avg_points, signed=True)
    return sym_plane


def get_fitted_line_along_y(ap_or_lat_line: vedo.Line, boundary_points: vedo.Points):
    """update fitted line to avoid longer lines than required"""
    vb_anterior_proj = project_points_onto_line(boundary_points, *ap_or_lat_line.points())
    anterior_up_proj = [(x,y,z) for x,y,z in vb_anterior_proj if y < ap_or_lat_line.center[1]]
    anterior_down_proj = [(x,y,z) for x,y,z in vb_anterior_proj if y > ap_or_lat_line.center[1]]

    anterior_up_most_proj_id = np.argmax([get_distance2_between_points(ap_or_lat_line.center, p) for p in anterior_up_proj])
    anterior_up_most_proj = anterior_up_proj[anterior_up_most_proj_id]

    anterior_down_most_proj_id = np.argmax([get_distance2_between_points(ap_or_lat_line.center, p) for p in anterior_down_proj])
    anterior_down_most_proj = anterior_down_proj[anterior_down_most_proj_id]
    # update line
    ap_or_lat_line = vedo.Line(anterior_down_most_proj, anterior_up_most_proj)
    return ap_or_lat_line


def get_fitted_line_along_z(sup_or_inf_line: vedo.Line, boundary_points: vedo.Points):
    """update fitted line to avoid longer lines than required"""
    vb_anterior_proj = project_points_onto_line(boundary_points, *sup_or_inf_line.points())
    anterior_up_proj = [(x,y,z) for x,y,z in vb_anterior_proj if z < sup_or_inf_line.center[2]]
    anterior_down_proj = [(x,y,z) for x,y,z in vb_anterior_proj if z > sup_or_inf_line.center[2]]

    anterior_up_most_proj_id = np.argmax([get_distance2_between_points(sup_or_inf_line.center, p) for p in anterior_up_proj])
    anterior_up_most_proj = anterior_up_proj[anterior_up_most_proj_id]

    anterior_down_most_proj_id = np.argmax([get_distance2_between_points(sup_or_inf_line.center, p) for p in anterior_down_proj])
    anterior_down_most_proj = anterior_down_proj[anterior_down_most_proj_id]
    # update line
    sup_or_inf_line = vedo.Line(anterior_down_most_proj, anterior_up_most_proj)
    return sup_or_inf_line



import argparse
parser = argparse.ArgumentParser()
parser.add_argument('nifti_file')
parser.add_argument('--offscreen',default=False,action='store_true')
args = parser.parse_args()

sample_file = args.nifti_file

vert_mesh = get_mesh_from_segmentation(sample_file,largest_component=True,reorient=True,orientation='ASL')
move_to_origin(vert_mesh)
# initial orientation
vert_ctd = vert_mesh.center_of_mass()
vert_mesh.compute_normals()

# setup symmetry plane: mirroring and registration
sym_plane = get_symmetry_plane(vert_mesh)
# arrows = vedo.Arrows(sampled_vmp, sampled_apts1, s=0.7, c='black', alpha=0.2)

# plt = vedo.Plotter(N=2, axes=1)


# plt.at(0).show(vert_mesh.c('white',0.5), mirrored_vert_mesh.c('green',0.05), sym_plane.c('red'),
#         #   vedo.Points(avg_points,c='red',r=8),
#           axes=1)
# plt.at(1).show(aligned_pts1.c('white', 0.02), arrows, vert_mesh_points.c('green',0.02),vedo.Points(avg_points,c='red'),sym_plane.c('red'))
# plt.interactive().close()
# sys.exit()

cut_mesh = vert_mesh.clone(transformed=True).cut_with_plane(normal=(sym_plane.normal))
sym_plane_boundaries = cut_mesh.boundaries()

sym_plane_points = sym_plane_boundaries.points().tolist()
# use kmeans to sepearte the vertebral body and spinous process boundary points
kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto").fit(sym_plane_points)
c0_x, c0_y, c0_z = kmeans.cluster_centers_[0]
c1_x, c1_y, c1_z = kmeans.cluster_centers_[1]
vb_label = 1 if c1_z < c0_z else 0
sp_label = int(not bool(vb_label))
vertebral_body_points = [p for p in sym_plane_points if kmeans.predict([p,])[0] == vb_label]
# smooth vertebral body points
vertebral_body_points = [np.mean(sym_plane_boundaries.closest_point(p, n=10), axis=0).tolist() for p in vertebral_body_points]
spinous_process_points = [p for p in sym_plane_points if kmeans.predict([p,])[0] == sp_label]

vbc = np.mean(vertebral_body_points, axis=0)
spc = np.mean(spinous_process_points, axis=0)
v0, s0, vcl = brute_force_search_get_closest_points_between_point_clouds(vertebral_body_points, spinous_process_points)
pq_unit_vec = get_vector_from_points(v0, s0)
vb_axis1, vb_axis2 = get_axis_lines(vbc, vertebral_body_points)
sp_axis1, sp_axis2 = get_axis_lines(spc, spinous_process_points)

# calculate spinous process morphometry: spa, spl
project_spp = project_points_onto_line(spinous_process_points, *sp_axis1)
anterior_project_spp = [(x,y,z) for x,y,z in project_spp if z < spc[2]]
posterior_project_spp = [(x,y,z) for x,y,z in project_spp if z > spc[2]]
posterior_most_spp_id = np.argmax([get_distance_between_points(spc, p) for p in posterior_project_spp])
posterior_most_spp = posterior_project_spp[posterior_most_spp_id]
anterior_most_spp_id = np.argmax([get_distance_between_points(spc, p) for p in anterior_project_spp])
anterior_most_spp = anterior_project_spp[anterior_most_spp_id]
spl = get_distance_between_points(anterior_most_spp, posterior_most_spp)

# find upper endplate and lower endplate points
vert_normals = vert_mesh.normals(recompute=False)
vb_normals = [vert_normals[vert_mesh.closest_point(p,return_point_id=True)] for p in vertebral_body_points]
# calculate dot product of vertebral body normals wrt vertebra foramen
vbn_projections = [np.dot(vbn, pq_unit_vec) for vbn in vb_normals]
vb_endplate = [vertebral_body_points[i] for i, vbnp in enumerate(vbn_projections) if ((vbnp < 0.5) and (vbnp > -0.5))]
vb_anteroposterior = [vertebral_body_points[i] for i, vbnp in enumerate(vbn_projections) if ((vbnp > 0.5) or (vbnp < -0.5))]

# separate endplates
m, c = get_slope_intercept_from_two_points_z_y(*vb_axis1)
vb_endplate_label = [1 if (z*m+c) > y else 0 for x, y, z in vb_endplate]
vb_endplate_pcl = vedo.Points(vb_endplate)

# smoothed_vb_endplate_label = []
# for p in vb_endplate:
#     ids = vb_endplate_pcl.closest_point(p, n=50, return_point_id=True)
#     labels = [vb_endplate_label[int(i)] for i in ids]
#     smoothed_vb_endplate_label.append(int(np.mean(labels)))

# vb_up = np.asarray([vb_endplate[i] for i, l in enumerate(smoothed_vb_endplate_label) if l > 0.5])
# vb_lp = np.asarray([vb_endplate[i] for i, l in enumerate(smoothed_vb_endplate_label) if l <= 0.5])
vb_up = np.asarray([(x, y, z) for x, y, z in vb_endplate if (z*m+c) > y])
vb_lp = np.asarray([(x, y, z) for x, y, z in vb_endplate if (z*m+c) < y])
# fit l1-regularized line
# vb_up = fit_l1(vb_up)
# vb_lp = fit_l1(vb_lp)

#separate anteroposterior boundaries
m, c = get_slope_intercept_from_two_points_y_z(*vb_axis2)
vb_ap = np.asarray([(x, y, z) for x, y, z in vb_anteroposterior if (y*m+c) > z])
vb_pp = np.asarray([(x, y, z) for x, y, z in vb_anteroposterior if (y*m+c) < z])
# fit L1-regularized line
# vb_ap = fit_l1(vb_ap)
# vb_pp = fit_l1(vb_pp)

a_bs = vedo.fit_line(np.asarray(vb_up))
a_bi = vedo.fit_line(np.asarray(vb_lp))
# update projectin lines to stay flush
a_bs = get_fitted_line_along_z(a_bs, vb_up)
a_bi = get_fitted_line_along_z(a_bi, vb_lp)

a_bs_0, a_bs_1 = a_bs.points()
a_bi_0, a_bi_1 = a_bi.points()
a_bm_0 = lerp(a_bs_0, a_bi_0, 0.5)
a_bm_1 = lerp(a_bs_1, a_bi_1, 0.5)
a_bm = vedo.fit_line(np.asarray([a_bm_0, a_bm_1]))


a_ba = vedo.fit_line(np.asarray(vb_ap))
a_bp = vedo.fit_line(np.asarray(vb_pp))


# update projection lines to stay flush (with no spikes https://github.com/naamiinepal/xrayto3D-morphometry/issues/18)
a_ba = get_fitted_line_along_y(a_ba, vb_ap)
a_bp = get_fitted_line_along_y(a_bp, vb_pp)

# spa is the angle between sp_axis1 and a_bs
a_bs_vec = get_vector_from_points(*a_bs.points())
spl_vec = get_vector_from_points(*sp_axis1)
spa = get_angle_between_vectors(spl_vec, a_bs_vec)

# # vertbral body measurements
anterior_vb_height = get_distance_between_points(*a_ba.points())
posterior_vb_height = get_distance_between_points(*a_bp.points())
superior_vb_length = get_distance_between_points(*a_bs.points())
inferior_vb_length = get_distance_between_points(*a_bi.points())


# coronal plane
# coronal_plane = vedo.Plane(pos=vbc, normal = (0, 1, 0), s = (60, 60), alpha = 0.5)
# PQ_m = lerp(v0, s0, 0.5)
# foramen = vedo.Cylinder(pos=PQ_m, r = vcl/2, height = 34, axis= (0,1,0))
# potential_cut_plane_origins = get_points_along_directions(PQ_m, direction = (0,0,1), num_points=20)
# vol_ratio = []
# for i in range(len(potential_cut_plane_origins)):
#     potential_coronal_cut = vert_mesh.clone(transformed=True).cut_with_plane(origin=potential_cut_plane_origins[i], normal=(0,0,1), invert=True).cap()
#     bounds = potential_coronal_cut.bounds()
#     length = bounds[1] - bounds[0]
#     width = bounds[3] - bounds[2]
#     height = bounds[5] - bounds[4]
#     vol = potential_coronal_cut.volume()
#     bbox_vol = length*width*height
#     vol_ratio.append(vol/bbox_vol)
#     print('cut volume', vol, ' bbox size',bbox_vol, 'vol/vol_bbox', vol/bbox_vol)
# cut_plane_origin = potential_cut_plane_origins[np.argmax(vol_ratio)]
# coronal_cut = vert_mesh.clone(transformed=True).cut_with_plane(origin=cut_plane_origin, normal = (0, 0, 1), invert=True)
# pedicle_boundaries = coronal_cut.boundaries()
# vedo.show(
#     # coronal_cut.c('white',0.75),
#     # vert_mesh.c('white', 0.75),
#     foramen,
#     pedicle_boundaries,
#     a_bm.lw(10),
#     axes=1
#     )

print(f'spl {spl:.2f}')
print(f'spa {180.0 - spa:.2f} degree')
print(f'avbh {anterior_vb_height:.2f} pvbh {posterior_vb_height:.2f} svbl {superior_vb_length:.2f} ivbl {inferior_vb_length:.2f}')
print(f'vcl {vcl:.2f}')
topview_cam = get_oriented_camera(vert_mesh, axis= 1, camera_dist = -200)
topview_cam['viewup'] = (-1, 0, 0)

sideview_cam = get_oriented_camera(vert_mesh, axis = 0, camera_dist = 200)
sideview_cam['viewup'] = (0,1,0)

vedo.show(
    # vert_mesh.c("white",1.0),
    vert_mesh.clone(transformed=True).cut_with_plane(normal=sym_plane.normal,invert=True).c('white',alpha=0.5),
    # vedo.Plane(normal=(1,0,0), s=(80, 80), alpha=0.5, c='green'),
    # sym_plane_boundaries,
    vedo.Points((vert_ctd, vbc, spc, v0, s0,),r=10),
    # vedo.Points(vb_endplate, r=8, c=(1.,0.,0.)),
    vedo.Points(vb_up,r=3,c=(1.,0.,0.)),
    vedo.Points(vb_lp,r=3,c=(1.,1.,0.)),
    vedo.Points(vb_anteroposterior, r=6, c='blue'),
    vedo.Points(spinous_process_points),
    # # vedo.Points(project_spp, r=3, c=(1.,0.,0.)),
    vedo.Point(posterior_most_spp, r=8, c='red'),
    vedo.Point(anterior_most_spp, r=8, c='red'),
    # vedo.Line(vb_axis1),
    vedo.Line(sp_axis1),
    vedo.Line(v0, s0),
    a_bs.lw(5), a_bi.lw(5),
    vedo.Points(a_bs.points(), r=8, c='white'),
    vedo.Points(a_bi.points(), r=8, c='white'),
    vedo.Points(a_ba.points(), r=8, c='white'), 
    vedo.Points(a_bp.points(), r=8, c='white'),
    # vedo.Point(a_ba.center, r=12, c='white'),
    # vedo.Point(a_bp.center, r=12, c='white'),
    a_ba.lw(5), a_bp.lw(5),
    axes=1,
    camera=sideview_cam,
    resetcam=False,
    offscreen=args.offscreen
)

outfile = Path('/mnt/driveD/repos/xrayto3D-morphometry/vertebra_debug_vis/sample.png').with_name(f'{Path(args.nifti_file).stem}.png')
print(outfile)
vedo.screenshot(str(outfile))
