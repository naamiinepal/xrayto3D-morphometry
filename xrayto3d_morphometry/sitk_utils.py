import SimpleITK as sitk

def get_segmentation_labels(segmentation:sitk.Image):
    fltr = get_segmentation_stats(segmentation)
    return fltr.GetLabels()

def get_segmentation_stats(segmentation: sitk.Image)->sitk.LabelShapeStatisticsImageFilter:
    fltr:sitk.LabelShapeStatisticsImageFilter = sitk.LabelShapeStatisticsImageFilter()
    fltr.ComputeOrientedBoundingBoxOn()
    fltr.ComputeFeretDiameterOn()
    fltr.Execute(segmentation)
    return fltr
