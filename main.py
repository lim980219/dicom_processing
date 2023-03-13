# import pydicom
# import numpy as np
# import matplotlib.pyplot as plt
# from pydicom.pixel_data_handlers.util import apply_modality_lut, apply_voi_lut
#
#
# window_center = -600
# window_width = 1600
#
# filename = "DATA/#11#21/CJS_CT/DCT0005.dcm"
# slice = pydicom.read_file(filename)
# s = int(slice.RescaleSlope)
# b = int(slice.RescaleIntercept)
# image = s * slice.pixel_array + b
#
# plt.subplot(1, 3, 1)
# plt.title('DICOM -> Array')
# plt.imshow(image, cmap='gray')
#
# # apply_modality_lut( ) & apply_voi_lut( )
# slice.WindowCenter = window_center
# slice.WindowWidth = window_width
# image = apply_modality_lut(image, slice)
# image2 = apply_voi_lut(image, slice)
# plt.subplot(1, 3, 2)
# plt.title('apply_voi_lut( )')
# plt.imshow(image2, cmap='gray')
#
# # normalization
# image3 = np.clip(image, window_center - (window_width / 2), window_center + (window_width / 2))
# plt.subplot(1, 3, 3)
# plt.title('normalize')
# plt.imshow(image3, cmap='gray')
#
# plt.show()




import SimpleITK as sitk
import numpy as np
import cv2
from matplotlib import pyplot as plt


def dicom_to_opencv(filename):


    image = sitk.ReadImage(filename)
    image_array = sitk.GetArrayFromImage(image).astype('float32')

    img = np.squeeze(image_array)
    copy_img = img.copy()
    min = np.min(copy_img)
    max = np.max(copy_img)

    copy_img1 = copy_img - np.min(copy_img)
    copy_img = copy_img1 / np.max(copy_img1)
    copy_img *= 2**8-1
    copy_img = copy_img.astype(np.uint8)

    copy_img = np.expand_dims(copy_img, axis=-1)
    copy_img = cv2.cvtColor(copy_img, cv2.COLOR_GRAY2RGB)

    return copy_img


filename = "DATA/#11#21/CJS_CT/DCT0066.dcm"
image = dicom_to_opencv(filename)

#노이즈 제거
denoised_img = cv2.fastNlMeansDenoisingColored(image, None, 15, 15, 5, 10)

cv2.imshow("before", image)
cv2.imshow("after", denoised_img)
cv2.waitKey(0)

plt.show()