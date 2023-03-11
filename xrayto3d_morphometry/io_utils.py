import SimpleITK as sitk
from pathlib import Path
import os 
import vedo
import pandas as pd
import shutil
from typing import Optional,List,Dict

def write_csv(data: Dict[str,List], file_path, column_names: Optional[List[str]]=None):
    pd.DataFrame.from_dict(data,orient='index',columns=column_names).to_csv(file_path,header=True,float_format="%.2f")


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

def get_files_from_run_id(run_id:str,suffix_regex:str)-> List[Path]:
    files = sorted(list(Path(f'2d-3d-benchmark/{run_id}/evaluation').glob(suffix_regex)))


    return files