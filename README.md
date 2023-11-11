# xrayto3D-morphometry

This is a python package for extracting landmarks from pelvic segmentation volume and morphological parameters from Proximal femur. 

**For Pelvic Bone Segmentation**
This is a adaptation of 
[Fischer 2019] Fischer, M. C. M., Krooß, F., Habor, J. & Radermacher, K. - A robust method for automatic identification of landmarks on surface models of the pelvis. Scientific Reports, https://doi.org/10.1038/s41598-019-49573-4 (2019) from matlab to python.

Original MATLAB code: [https://github.com/RWTHmediTEC/PelvicLandmarkIdentification](https://github.com/RWTHmediTEC/PelvicLandmarkIdentification)

**For Proximal Femur**
This is a adapted implementation from [Cerveri, Pietro, et al. "Automated method for computing the morphological and clinical parameters of the proximal femur using heuristic modeling techniques." Annals of Biomedical Engineering 38 (2010): 1752-1766](https://pubmed.ncbi.nlm.nih.gov/20177779/)

**For Vertebra**
Adapted from Di Angelo, Luca, and Paolo Di Stefano. "A new method for the automatic identification of the dimensional features of vertebrae." Computer methods and programs in biomedicine 121.1 (2015): 36-48.
#### Usage
---
```shell
python pelvic_landmarks_v2.py --file test_data/s0014_hip_msk_pred.nii.gz
```
```json
{'ASIS_L': array([-79.967545, -21.783907, -50.073395], dtype=float32), 'ASIS_R': array([ 85.032455, -11.783906, -53.073395], dtype=float32), 'PT_L': array([-16.967546,  39.216095, -50.073395], dtype=float32), 'PT_R': array([ 13.032453,  44.216095, -50.073395], dtype=float32)}
```
![image](https://user-images.githubusercontent.com/10219364/236784319-8e6f7b76-cd3c-4f43-affa-983d6f800455.png)


---
```shell
python vertebra_landmarks.py test_data/sub-verse004_vert-23-seg-vert_msk.nii.gz
```

```json
{'spl': 36.453527302556076, 'spa': 138.60229619651807, 'avbh': 26.23448596400926, 'pvbh': 23.378738362530747, 'svbl': 29.54202176642218, 'ivbl': 30.34065592251007, 'vcl': 13.27892594709724}
```
![Alt text](test_data/sub-verse004_vert-23-seg-vert_msk_landmarks.png)