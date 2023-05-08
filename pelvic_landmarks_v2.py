import argparse
from pathlib import Path
import shutil
import vedo

from xrayto3d_morphometry import (
    get_mesh_from_segmentation,
    move_to_origin,
    align_along_principal_axes,
    lerp,
    get_oriented_camera
)

from pelvic_landmark_utils import (
    get_asis_estimate,
    get_maximal_pelvic_points,
    get_quadrant_meshes,
    get_transverse_plane_height,
)


def get_landmarks(nifti_filename):
    """return landmarks as dict"""
    mesh = get_mesh_from_segmentation(nifti_filename)
    move_to_origin(mesh)
    aligned_mesh, T = align_along_principal_axes(mesh)

    mwp_p1_idx, mwp_p2_idx = get_maximal_pelvic_points(aligned_mesh)
    mwp_midpoint = lerp(
        aligned_mesh.points()[mwp_p1_idx], aligned_mesh.points()[mwp_p2_idx], 0.5
    )  # type: ignore
    tph_intercept, dlc, drc = get_transverse_plane_height(
        aligned_mesh, mwp_midpoint, alpha=0.6, additional_landmarks=True
    )  # type: ignore

    pt_p1_idx, pt_p2_idx, asis_p1_idx, asis_p2_idx, T = get_asis_estimate(
        aligned_mesh, (0, tph_intercept, 0), verbose=True
    )
    return {
        "ASIS_L": mesh.points()[asis_p1_idx],
        "ASIS_R": mesh.points()[asis_p2_idx],
        "PT_L": mesh.points()[pt_p1_idx],
        "PT_R": mesh.points()[pt_p2_idx],
    }


def process_predictions(run_id, debug=False):
    """compare gt and pred landmarks"""
    gt_path = sorted(
        list(Path(f"2d-3d-benchmark/{run_id}/evaluation").glob("*_gt.nii.gz"))
    )
    pred_path = sorted(
        list(Path(f"2d-3d-benchmark/{run_id}/evaluation").glob("*_pred.nii.gz"))
    )
    if not debug:
        # remove old visualizations
        if Path(f"2d-3d-benchmark/{run_id}/visualization_v2").exists():
            shutil.rmtree(Path(f"2d-3d-benchmark/{run_id}/visualization_v2"))
        Path(f"2d-3d-benchmark/{run_id}/visualization_v2").mkdir()

    for gt, pred in zip(gt_path, pred_path):
        pass


def process_predictions_entrypoint():
    """process a run directory"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", default=False, action="store_true")
    parser.add_argument("--run_id")

    args = parser.parse_args()

    if Path(f"2d-3d-benchmark/{args.run_id}/evaluation").exists():
        process_predictions(run_id=args.run_id, debug=args.debug)
    else:
        print(Path(f"2d-3d-benchmark/{args.run_id}/evaluation"), " does not exist")

def visualize_landmarks(nifti_filename, landmarks,screenshot=Falses):
    mesh = get_mesh_from_segmentation(nifti_filename)
    move_to_origin(mesh)
    landmark_points = vedo.Points([landmarks[k] for k in landmarks],c='white',r=15)
    cam = get_oriented_camera(mesh, axis=2, camera_dist=400)
    vedo.show(mesh, landmark_points, resetcam=False, camera = cam,axes=1,offscreen=screenshot)
    if screenshot:
        Path(screenshot_outdir).mkdir(exist_ok=True)
        out_filename = (
                Path(screenshot_outdir)
                / f'{i}_{Path(nifti_filename).with_suffix(".png").name}'
            )
        vedo.screenshot(str(out_filename))
        vedo.close()

def test_single_example():
    """test stub"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=False)
    parser.add_argument("--iterative", default=False, action="store_true")

    args = parser.parse_args()
    if args.file:
        sample_nifti = args.file
    else:
        sample_nifti = "test_data/s0014_hip_msk_pred.nii.gz"

    landmarks = get_landmarks(sample_nifti)
    print(landmarks)
    visualize_landmarks(sample_nifti,landmarks)

if __name__ == '__main__':
    test_single_example()
