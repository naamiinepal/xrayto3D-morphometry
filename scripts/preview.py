"""
Please install xvfb via sudo apt install xvfb
and xvfbwrapper via pip install xvfbwrapper 
to run this script.
"""
from fury import window
import numpy as np
import vtk
from vtk.util import numpy_support
from nibabel.processing import resample_to_output

# original code from https://github.com/wasserth/TotalSegmentator/blob/master/totalsegmentator/preview.py


def generate_preview(ct_in, file_out, label_id=1, **kwargs):
    from xvfbwrapper import Xvfb

    with Xvfb() as xvfb:
        plot_subject(ct_in, file_out, label_id=label_id, **kwargs)


def plot_subject(
    ct_img,
    output_path,
    label_id=1,
    window_size=[500, 500],
    background=[1.0, 1.0, 1.0],
    smoothing=10,
    color=[255.0, 193.0, 149.0],
    orientation="sagittal",
    projection="orthographic",
):
    scene = window.Scene()
    scene.background(background)
    window.ShowManager(scene, size=window_size, reset_camera=False).initialize()

    data = ct_img.get_fdata()
    data = reorient(data, orientation)

    roi_data = data == label_id
    affine = ct_img.affine
    affine[:3, 3] = 0
    roi_actor = marching_cubes(
        roi_data, affine, color=[c / 255.0 for c in color], smoothing=smoothing
    )
    roi_actor.SetPosition(0, 0, 0)
    scene.add(roi_actor)

    scene.projection(proj_type=projection)
    scene.reset_camera_tight(margin_factor=1.0)

    window.record(scene, size=window_size, out_path=output_path, reset_camera=False)
    scene.clear()


def reorient(data, orientation):
    """reorient numpy volume according to medical planes"""
    if orientation == "sagittal":
        data = data.transpose(1, 2, 0)
    elif orientation == "coronal":
        data = np.rot90(data)
        data = data[::-1, :, :]
        data = data[:, :, ::-1]
    return data


def marching_cubes(data, affine, color=(1, 0, 0), opacity=1, smoothing=0):
    vtk_major_version = vtk.vtkVersion.GetVTKMajorVersion()

    vol = np.interp(data, xp=[data.min(), data.max()], fp=[0, 255])

    im = vtk.vtkImageData()
    if vtk_major_version <= 5:
        im.SetScalarTypeToUnsignedChar()
    di, dj, dk = vol.shape[:3]
    im.SetDimensions(di, dj, dk)
    voxsz = (1.0, 1.0, 1.0)
    im.SetSpacing(voxsz[2], voxsz[0], voxsz[1])
    if vtk_major_version <= 5:
        im.AllocateScalars()
        im.SetNumberOfScalarComponents(1)
    else:
        im.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)
    # copy data
    vol = np.swapaxes(vol, 0, 2)
    vol = np.ascontiguousarray(vol)
    vol = vol.ravel()

    uchar_array = numpy_support.numpy_to_vtk(vol, deep=0)
    im.GetPointData().SetScalars(uchar_array)

    if affine is None:
        affine = np.eye(4)
    # Set the transform (identity if none given)
    transform = vtk.vtkTransform()
    transform_matrix = vtk.vtkMatrix4x4()
    transform_matrix.DeepCopy(
        (
            affine[0][0],
            affine[0][1],
            affine[0][2],
            affine[0][3],
            affine[1][0],
            affine[1][1],
            affine[1][2],
            affine[1][3],
            affine[2][0],
            affine[2][1],
            affine[2][2],
            affine[2][3],
            affine[3][0],
            affine[3][1],
            affine[3][2],
            affine[3][3],
        )
    )
    transform.SetMatrix(transform_matrix)
    transform.Inverse()

    # Set the reslicing
    image_resliced = vtk.vtkImageReslice()
    set_input(image_resliced, im)
    image_resliced.SetResliceTransform(transform)
    image_resliced.AutoCropOutputOn()

    # Adding this will allow to support anisotropic voxels
    # and also gives the opportunity to slice per voxel coordinates

    rzs = affine[:3, :3]
    zooms = np.sqrt(np.sum(rzs * rzs, axis=0))
    image_resliced.SetOutputSpacing(*zooms)

    image_resliced.SetInterpolationModeToLinear()
    image_resliced.Update()

    skin_extractor = vtk.vtkMarchingCubes()
    if vtk_major_version <= 5:
        skin_extractor.SetInput(image_resliced.GetOutput())
    else:
        skin_extractor.SetInputData(image_resliced.GetOutput())
    skin_extractor.SetValue(0, 100)

    if smoothing > 0:
        smoother = vtk.vtkSmoothPolyDataFilter()
        smoother.SetInputConnection(skin_extractor.GetOutputPort())
        smoother.SetNumberOfIterations(smoothing)
        smoother.SetRelaxationFactor(0.1)
        smoother.SetFeatureAngle(60)
        smoother.FeatureEdgeSmoothingOff()
        smoother.BoundarySmoothingOff()
        smoother.SetConvergence(0)
        smoother.Update()

    skin_normals = vtk.vtkPolyDataNormals()
    if smoothing > 0:
        skin_normals.SetInputConnection(smoother.GetOutputPort())
    else:
        skin_normals.SetInputConnection(skin_extractor.GetOutputPort())

    skin_normals.SetFeatureAngle(60.0)

    skin_mapper = vtk.vtkPolyDataMapper()
    skin_mapper.SetInputConnection(skin_normals.GetOutputPort())
    skin_mapper.ScalarVisibilityOff()

    skin_actor = vtk.vtkActor()
    skin_actor.SetMapper(skin_mapper)
    skin_actor.GetProperty().SetOpacity(opacity)
    skin_actor.GetProperty().SetColor(color)

    return skin_actor


def set_input(vtk_object, inp):
    """Set Generic input function which takes into account VTK 5 or 6.

    Parameters
    ----------
    vtk_object: vtk object
    inp: vtkPolyData or vtkImageData or vtkAlgorithmOutput

    Returns
    -------
    vtk_object

    Notes
    -------
    This can be used in the following way::
        from fury.utils import set_input
        poly_mapper = set_input(vtk.vtkPolyDataMapper(), poly_data)

    This function is copied from dipy.viz.utils
    """
    if isinstance(inp, (vtk.vtkPolyData, vtk.vtkImageData)):
        vtk_object.SetInputData(inp)
    elif isinstance(inp, vtk.vtkAlgorithmOutput):
        vtk_object.SetInputConnection(inp)
    vtk_object.Update()
    return vtk_object


if __name__ == "__main__":
    import nibabel as nib
    from pathlib import Path
    import argparse
    from xrayto3d_morphometry import multiply_tuple

    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="path to volume")
    parser.add_argument("--label-id", default=1, type=int)
    parser.add_argument("--size", nargs="+", type=int, default = [500, 500])
    parser.add_argument("--color", nargs="+", type=int, default=[255, 0, 0])
    parser.add_argument(
        "--orientation",
        type=str,
        choices=["sagittal", "coronal", "axial"],
        default="sagittal",
    )
    parser.add_argument("--smoothing", type=int, default=10)
    parser.add_argument(
        "--projection", choices=["parallel", "orthographic"], default="orthographic"
    )
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    ct_in_path = args.file
    ct_in = nib.load(ct_in_path)
    out_voxel_size = (max(ct_in.header.get_zooms()),)*len(ct_in.header.get_zooms())
    ct_in = resample_to_output(ct_in, voxel_sizes=out_voxel_size,mode='nearest')

    if args.out is None:
        parent = Path(ct_in_path).parent
        filestem = Path(ct_in_path).stem.split(".")[0]
        file_out = (
            parent / f"{filestem}_{args.orientation}.png"
        )  # COLONOG-0001.nii.gz -> COLONOG-0001_sagittal.png
    else:
        file_out = args.out

    generate_preview(
        ct_in,
        file_out,
        label_id=args.label_id,
        window_size=args.size,
        orientation=args.orientation,
        smoothing=args.smoothing,
        projection=args.projection,
        color=args.color,
    )
