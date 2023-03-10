import SimpleITK as sitk
from pathlib import Path
import os 
import vedo

def read_mesh(mesh_path:str)->vedo.Mesh:
    return vedo.load(mesh_path)

def read_volume(img_path)->sitk.Image:
    """returns the SimpleITK image read from given path
    Parameters:
    -----------
    pixeltype (ImagePixelType):
    """
    img_path = Path(img_path).resolve()
    img_path = str(img_path)


    return sitk.ReadImage(img_path)

def get_nifti_stem(path):
    """
    '/home/user/image.nii.gz' -> 'image'
    1.3.6.1.4.1.14519.5.2.1.6279.6001.905371958588660410240398317235.nii.gz ->1.3.6.1.4.1.14519.5.2.1.6279.6001.905371958588660410240398317235
    """
    def _get_stem(path_string) -> str:
        name_subparts = Path(path_string).name.split('.')
        return '.'.join(name_subparts[:-2]) # get rid of nii.gz
    if isinstance(path, (str, os.PathLike)):
        return _get_stem(path)