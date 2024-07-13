import sys

sys.path.insert(0, '/build/SimpleITK-build/Wrapping/Python/Packaging/build/lib.linux-x86_64-3.6/SimpleITK')
import SimpleITK as sitk
import numpy as np
import os

def resample_to_fixed_size(image, new_size=[128, 128, 128]):
    original_size = image.GetSize()
    original_spacing = image.GetSpacing()
    new_spacing = [
        (original_size[0] - 1) * original_spacing[0] / (new_size[0] - 1),
        (original_size[1] - 1) * original_spacing[1] / (new_size[1] - 1),
        (original_size[2] - 1) * original_spacing[2] / (new_size[2] - 1)
    ]
    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(new_size)
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0)
    return resampler.Execute(image)

def perform_registration(fixed_image, moving_image):
    elastixImageFilter = sitk.ElastixImageFilter()
    elastixImageFilter.SetFixedImage(fixed_image)
    elastixImageFilter.SetMovingImage(moving_image)

    parameterMapVector = sitk.VectorOfParameterMap()

    rigid_params = sitk.GetDefaultParameterMap("rigid")
    rigid_params["MaximumNumberOfIterations"] = ["128"]
    parameterMapVector.append(rigid_params)

    affine_params = sitk.GetDefaultParameterMap("affine")
    affine_params["MaximumNumberOfIterations"] = ["512"]
    parameterMapVector.append(affine_params)

    bspline_params = sitk.GetDefaultParameterMap("bspline")
    bspline_params["MaximumNumberOfIterations"] = ["512"]
    parameterMapVector.append(bspline_params)

    elastixImageFilter.SetParameterMap(parameterMapVector)
    elastixImageFilter.Execute()

    result_image = elastixImageFilter.GetResultImage()
    transform_parameter_map = elastixImageFilter.GetTransformParameterMap()
    return result_image, transform_parameter_map

def apply_transform(transform_parameter_map, moving_image):
    transformixImageFilter = sitk.TransformixImageFilter()
    transformixImageFilter.SetTransformParameterMap(transform_parameter_map)
    transformixImageFilter.SetMovingImage(moving_image)
    transformixImageFilter.ComputeDeformationFieldOn()
    transformixImageFilter.Execute()
    deformation_field = transformixImageFilter.GetDeformationField()
    result_image = transformixImageFilter.GetResultImage()
    return result_image, deformation_field

base_path = '/nifti_data/train/'
mr_images_path = os.path.join(base_path, 'mr_images')
us_images_path = os.path.join(base_path, 'us_images')
mr_labels_path = os.path.join(base_path, 'mr_labels')
us_labels_path = os.path.join(base_path, 'us_labels')
results_path = os.path.join(base_path, 'results')
ddf_results_path = os.path.join(results_path, 'ddf')
label_results_path = os.path.join(results_path, 'label')

mr_image_files = sorted(os.listdir(mr_images_path))
us_image_files = sorted(os.listdir(us_images_path))

for mr_image_file, us_image_file in zip(mr_image_files, us_image_files):
    print(f"Processing {mr_image_file} and {us_image_file}")
    mr_image_full_path = os.path.join(mr_images_path, mr_image_file)
    us_image_full_path = os.path.join(us_images_path, us_image_file)
    mr_image = sitk.ReadImage(mr_image_full_path)
    us_image = sitk.ReadImage(us_image_full_path)
    mr_label_all = sitk.ReadImage(os.path.join(mr_labels_path, mr_image_file))
    us_label_all = sitk.ReadImage(os.path.join(us_labels_path, us_image_file))

    num_labels = 6
    warped_labels = []
    ddf_result = []

    for label_idx in range(num_labels):
        print(f"Processing label {label_idx} for {mr_image_file}")
        try:
            mr_label = mr_label_all[:, :, :, label_idx]
            us_label = us_label_all[:, :, :, label_idx]

            mr_prostate_roi = sitk.Mask(mr_image, mr_label)
            us_prostate_roi = sitk.Mask(us_image, us_label)
            mr_roi_resampled = resample_to_fixed_size(mr_prostate_roi)
            us_roi_resampled = resample_to_fixed_size(us_prostate_roi)

            mr_roi_resampled = sitk.Cast(mr_roi_resampled, sitk.sitkFloat32)
            us_roi_resampled = sitk.Cast(us_roi_resampled, sitk.sitkFloat32)

            registered_image, transform_parameter_map = perform_registration(mr_roi_resampled, us_roi_resampled)

            warped_moving_label, ddf = apply_transform(transform_parameter_map, us_label)

            ddf_array = sitk.GetArrayFromImage(ddf)
            if ddf_array.max() == 0 and label_idx != 5:
                print(f"\033[91mWarning: DDF is zero for {mr_image_file} at label {label_idx}\033[0m")

            warped_labels.append(sitk.GetArrayFromImage(warped_moving_label))
            ddf_result.append(ddf_array)
        except Exception as e:
            print(f"Error processing label {label_idx} for {mr_image_file}: {e}")
            continue

    while len(warped_labels) < 6:
        zero_label = np.zeros((128, 128, 128))
        warped_labels.append(zero_label)

    warped_labels_np = np.stack(warped_labels, axis=0)  # Shape: (6, 128, 128, 128)
    warped_labels_np = np.moveaxis(warped_labels_np, 0, -1)  # Shape: (128, 128, 128, 6)
    ddf_result_np = np.stack(ddf_result, axis=-1)  # Shape: (128, 128, 128, 3, 6)
    ddf_result_np = np.mean(ddf_result_np, axis=-1)  # Shape: (128, 128, 128, 3)

    assert warped_labels_np.shape == (128, 128, 128, 6), f"Expected (128, 128, 128, 6), but got {warped_labels_np.shape}"
    assert ddf_result_np.shape == (128, 128, 128, 3), f"Expected (128, 128, 128, 3), but got {ddf_result_np.shape}"

    joined_labels = sitk.GetImageFromArray(warped_labels_np)
    joined_ddf = sitk.GetImageFromArray(ddf_result_np)
    mr_resample = resample_to_fixed_size(mr_image)
    joined_labels.CopyInformation(mr_resample)

    label_save_path = os.path.join(label_results_path, "warped_labels_" + os.path.basename(mr_image_file))
    ddf_save_path = os.path.join(ddf_results_path, "ddf_" + os.path.basename(mr_image_file))
    sitk.WriteImage(joined_labels, label_save_path)
    sitk.WriteImage(joined_ddf, ddf_save_path)
