import os
import numpy as np
from pydicom import dcmread
import matplotlib.pyplot as plt
from PIL import Image

def coordinates(points_file):
    line_count = 0
    x = []
    y = []
    with open(points_file) as fp:
        for line in fp:
            if line_count >= 1:
                end_point = 151  
            if 3 <= line_count < end_point:
                x_temp, y_temp = line.split() 
                y_temp = y_temp.strip()  
                x.append(float(x_temp))
                y.append(float(y_temp))
            line_count += 1
    return np.array(x), np.array(y)  

dicom_dir = r"C:\Users\ehret\OneDrive\Studium\Sem5 (Samk)\AI Theme 3 Medial Image Classification\2 ROI\data\dicoms"
landmarks_dir = r"C:\Users\ehret\OneDrive\Studium\Sem5 (Samk)\AI Theme 3 Medial Image Classification\2 ROI\data\landmarks"

for dcm in os.listdir(dicom_dir):
    dcm_name = os.path.splitext(dcm)[0]  
    dcm_path = os.path.join(dicom_dir, dcm)  
    dicom_data = dcmread(dcm_path)  

    landmark_path = os.path.join(landmarks_dir, f"{dcm_name}.pts")  
    Xs, Ys = coordinates(landmark_path)  
     ############## show original image ###############
    plt.imshow(dicom_data.pixel_array, cmap=plt.cm.gray)
    plt.title(f"Patient ID: {dicom_data.PatientID}")
    plt.axis('off')
    plt.show()

    # Landmarks 
    plt.imshow(dicom_data.pixel_array, cmap=plt.cm.gray)
    plt.scatter(Xs, Ys, c='red', marker='.', s=10)
    plt.title(f"Patient ID: {dicom_data.PatientID}")
    plt.show()

     ############# Pixel Spacing based cropping ############
    pixel_spacing = dicom_data.PixelSpacing  # pixel in mm
    pixel_spacing_mm = np.array(pixel_spacing, dtype=float)

    desired_crop_size_mm = 10  
    margin_pixels = desired_crop_size_mm / pixel_spacing_mm

    x_min = max(0, min(Xs) - margin_pixels[0])
    x_max = min(dicom_data.pixel_array.shape[1], max(Xs) + margin_pixels[0])
    y_min = max(0, min(Ys) - margin_pixels[1])
    y_max = min(dicom_data.pixel_array.shape[0], max(Ys) + margin_pixels[1])
    cropped_image = dicom_data.pixel_array[int(y_min):int(y_max), int(x_min):int(x_max)]
    plt.imshow(cropped_image, cmap=plt.cm.gray)
    adjusted_Xs = Xs - x_min
    adjusted_Ys = Ys - y_min
    mid_x = (max(Xs) + min(Xs)) / 2

    left_knee = cropped_image[:, :int(mid_x - x_min)]
    right_knee = cropped_image[:, int(mid_x - x_min):]

    left_indices = np.where(Xs <= mid_x)
    right_indices = np.where(Xs > mid_x)

    right_Xs = Xs[right_indices] - mid_x
    right_Ys = Ys[right_indices] - y_min
    left_Xs = Xs[left_indices] - x_min
    left_Ys = Ys[left_indices] - y_min

    left_output_filename = os.path.join(r"C:\Users\ehret\OneDrive\Studium\Sem5 (Samk)\AI Theme 3 Medial Image Classification\2 ROI\data", f"{dcm_name}_L.png")
    right_output_filename = os.path.join(r"C:\Users\ehret\OneDrive\Studium\Sem5 (Samk)\AI Theme 3 Medial Image Classification\2 ROI\data", f"{dcm_name}_R.png")

    plt.imsave(left_output_filename, left_knee, cmap='gray')
    plt.imsave(right_output_filename, right_knee, cmap='gray')

    plt.imshow(left_knee, cmap=plt.cm.gray)
    plt.scatter(left_Xs, left_Ys, c='green', marker='.', s=10)
    plt.title(f"Patient ID: {dicom_data.PatientID} - Left Knee")
    plt.show()

    plt.imshow(right_knee, cmap=plt.cm.gray)
    plt.scatter(right_Xs, right_Ys, c='green', marker='.', s=10)
    plt.title(f"Patient ID: {dicom_data.PatientID} - Right Knee")
    plt.show()

    print("finished")
