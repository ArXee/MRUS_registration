import sys
sys.path.insert(0, '/home/wyp/Downloads/medical/pytorch/build/SimpleITK-build/Wrapping/Python/Packaging/build/lib.linux-x86_64-3.6/SimpleITK')
import SimpleITK as sitk
import numpy as np
import os
from utils import registration_loss as rgl

def resample_to_fixed_size(image, new_size=[128, 128, 128]):
    if image.GetDimension() == 4:
        resampled_volumes = []
        for i in range(image.GetSize()[3]):
            volume = image[:, :, :, i]
            volume = sitk.Cast(volume, sitk.sitkFloat32)
            resampled_volume = resample_to_fixed_size(volume, new_size)
            resampled_volumes.append(resampled_volume)
        return sitk.JoinSeries(resampled_volumes)
    else:
        if image.GetNumberOfComponentsPerPixel() > 1:
            components = [sitk.VectorIndexSelectionCast(image, i, sitk.sitkFloat32) for i in range(image.GetNumberOfComponentsPerPixel())]
            image = sitk.Compose(components)
        else:
            image = sitk.Cast(image, sitk.sitkFloat32)

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

base_path = '/home/wyp/Downloads/medical/pytorch/nifti_data/train/'
mr_labels_path = os.path.join(base_path, 'mr_labels')
us_labels_path = os.path.join(base_path, 'us_labels')
results_path = os.path.join(base_path, 'results')
labels_results_path = os.path.join(results_path, 'label')
ddf_results_path = os.path.join(results_path, 'ddf')

mr_labels_files = sorted(os.listdir(mr_labels_path))
us_labels_files = sorted(os.listdir(us_labels_path))
eval_labels_files = sorted(os.listdir(labels_results_path))
eval_ddfs_files = sorted(os.listdir(ddf_results_path))

all_labels_fixed_labels = []
all_labels_moved_labels = []
all_labels_moving_labels = []
all_labels_ddfs = []

new_size = [128, 128, 128]

for mr_label_file, us_label_file, eval_label_file, eval_ddf_file in zip(mr_labels_files, us_labels_files, eval_labels_files, eval_ddfs_files):
    mr_label_full_path = os.path.join(mr_labels_path, mr_label_file)
    us_label_full_path = os.path.join(us_labels_path, us_label_file)
    eval_label_full_path = os.path.join(labels_results_path, eval_label_file)
    eval_ddf_full_path = os.path.join(ddf_results_path, eval_ddf_file)

    mr_label = sitk.ReadImage(mr_label_full_path)
    us_label = sitk.ReadImage(us_label_full_path)
    eval_label = sitk.ReadImage(eval_label_full_path)
    eval_ddf = sitk.ReadImage(eval_ddf_full_path)

    mr_label_resampled = resample_to_fixed_size(mr_label, new_size)
    us_label_resampled = resample_to_fixed_size(us_label, new_size)
    eval_label_resampled = resample_to_fixed_size(eval_label, new_size)

    mr_label_array = sitk.GetArrayFromImage(mr_label_resampled)
    us_label_array = sitk.GetArrayFromImage(us_label_resampled)

    eval_label_array = sitk.GetArrayFromImage(eval_label_resampled)
    eval_ddf_array = sitk.GetArrayFromImage(eval_ddf)

    all_labels_fixed_labels.append(mr_label_array)
    all_labels_moved_labels.append(us_label_array)
    all_labels_moving_labels.append(eval_label_array)
    all_labels_ddfs.append(eval_ddf_array)

all_dscs = []
all_rdscs = []
maes = []
all_hd95s = []
all_lim_hd95s = []
all_lim_maes = []

for sample_idx in range(len(all_labels_fixed_labels)):
    print(f"Processing sample {sample_idx + 1}/{len(all_labels_fixed_labels)}")
    ddf = all_labels_ddfs[sample_idx]
    moving_label = all_labels_moving_labels[sample_idx]
    print(f"DDF shape: {ddf.shape}")
    print(f"Eval label shape: {moving_label.shape}")
    fixed_label = all_labels_fixed_labels[sample_idx]
    for label_idx in range(6):
        eval_label = moving_label[:, :, :, label_idx]
        fixed_label_label = fixed_label[label_idx]
        moved_label = all_labels_moved_labels[sample_idx][label_idx]

        if label_idx == 0:
            dsc = rgl.compute_dice(fixed_label_label, eval_label)
            all_dscs.append(dsc)

        fixed_centroid = rgl.compute_centroid(fixed_label_label)
        moved_centroid = rgl.compute_centroid(moved_label)
        mae = rgl.compute_centroid_mae(fixed_centroid, moved_centroid)
        maes.append(mae)

        hd95 = rgl.compute_hausdorff95(fixed_label_label, moved_label)
        all_hd95s.append(hd95)

        lim_hd95 = rgl.compute_hausdorff95(fixed_label_label, eval_label)
        all_lim_hd95s.append(lim_hd95)

        lim_centroid = rgl.compute_centroid(eval_label)
        lim_mae = rgl.compute_centroid_mae(fixed_centroid, lim_centroid)
        all_lim_maes.append(lim_mae)

    rdsc = rgl.compute_robust_dice([x for x in all_dscs if not np.isnan(x)], 68)
    all_rdscs.append(rdsc)

fin_DSC = np.nanmean(all_dscs)
fin_RDSC = np.nanmean(all_rdscs)
fin_HD95 = np.nanmean([x for x in all_hd95s if not np.isnan(x)])
fin_lim_HD95 = np.nanmean([x for x in all_lim_hd95s if not np.isnan(x)])

# Convert MAEs to 2D array and filter out NaN values
maes = np.array([elem for elem in maes if not np.isnan(elem)]).reshape(-1, 1)
print("MAEs before reshaping:")
print(maes)
all_maes_array = maes if maes.ndim > 1 else maes.reshape(-1, 1)
print("All MAEs array shape:", all_maes_array.shape)

lim_maes = np.array([elem for elem in all_lim_maes if not np.isnan(elem)]).reshape(-1, 1)
print("Lim MAEs before reshaping:")
print(lim_maes)
all_lim_maes_array = lim_maes if lim_maes.ndim > 1 else lim_maes.reshape(-1, 1)
print("All lim MAEs array shape:", all_lim_maes_array.shape)

# Filter out zero values for RTRE and other calculations
all_maes_array = all_maes_array[all_maes_array != 0].reshape(-1, 1)
all_lim_maes_array = all_lim_maes_array[all_lim_maes_array != 0].reshape(-1, 1)

# Ensure arrays are not empty before computing metrics
if all_maes_array.size == 0:
    fin_TRE = np.nan
    fin_RTRE = np.nan
    fin_RTs = np.nan
else:
    fin_TRE = rgl.compute_tre(all_maes_array)
    fin_RTRE = rgl.compute_rtre(all_maes_array)
    fin_RTs = rgl.compute_rts(all_maes_array)

if all_lim_maes_array.size == 0:
    fin_lim_TRE = np.nan
    fin_lim_RTRE = np.nan
    fin_lim_RTs = np.nan
else:
    fin_lim_TRE = rgl.compute_tre(all_lim_maes_array)
    fin_lim_RTRE = rgl.compute_rtre(all_lim_maes_array)
    fin_lim_RTs = rgl.compute_rts(all_lim_maes_array)

print("Finished calculations!")
print("DSCs:", all_dscs)
print("Robust DSCs:", all_rdscs)
print("MAEs:", maes)
print("Hausdorff 95 Distances:", all_hd95s)
print("Lim Hausdorff 95 Distances:", all_lim_hd95s)
print("Lim MAEs:", all_lim_maes)


def score_calculator(fin_DSC, fin_RDSC, fin_HD95, fin_lim_HD95, fin_TRE, fin_lim_TRE, fin_RTRE, fin_lim_RTRE, fin_RTs, fin_lim_RTs):
    # Check if any input is NaN
    if any(np.isnan([fin_DSC, fin_RDSC, fin_HD95, fin_lim_HD95, fin_TRE, fin_lim_TRE, fin_RTRE, fin_lim_RTRE, fin_RTs, fin_lim_RTs])):
        return np.nan
    score = (
        0.2 * float(fin_DSC) +
        0.1 * float(fin_RDSC) +
        0.3 * (1 - np.clip(fin_TRE / fin_lim_TRE, 0, 1)) +
        0.1 * (1 - np.clip(fin_RTRE / fin_lim_RTRE, 0, 1)) +
        0.1 * (1 - np.clip(fin_RTs / fin_lim_RTs, 0, 1)) +
        0.2 * (1 - np.clip(fin_HD95 / fin_lim_HD95, 0, 1))
    )
    return score

score = score_calculator(fin_DSC, fin_RDSC, fin_HD95, fin_lim_HD95, fin_TRE, fin_lim_TRE, fin_RTRE, fin_lim_RTRE, fin_RTs, fin_lim_RTs)
print(f'\nFinal Score: {score}')
