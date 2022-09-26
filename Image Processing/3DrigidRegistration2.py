import SimpleITK as sitk
import numpy as np

from scipy.io import savemat
import scipy.io as sio

import matplotlib.pyplot as plt

# Always write output to a separate directory, we don't want to pollute the source directory. 
import os
OUTPUT_DIR = 'output\S14870_20_30'


def mask_image_multiply(mask, image):
    components_per_pixel = image.GetNumberOfComponentsPerPixel()
    if  components_per_pixel == 1:
        return mask*image
    else:
        return sitk.Compose([mask*sitk.VectorIndexSelectionCast(image,channel) for channel in range(components_per_pixel)])

def alpha_blend(image1, image2, alpha = 0.5, mask1=None,  mask2=None):
    '''
    Alaph blend two images, pixels can be scalars or vectors.
    The region that is alpha blended is controled by the given masks.
    '''
    
    if not mask1:
        mask1 = sitk.Image(image1.GetSize(), sitk.sitkFloat32) + 1.0
        mask1.CopyInformation(image1)
    else:
        mask1 = sitk.Cast(mask1, sitk.sitkFloat32)
    if not mask2:
        mask2 = sitk.Image(image2.GetSize(),sitk.sitkFloat32) + 1
        mask2.CopyInformation(image2)
    else:        
        mask2 = sitk.Cast(mask2, sitk.sitkFloat32)

    components_per_pixel = image1.GetNumberOfComponentsPerPixel()
    if components_per_pixel>1:
        img1 = sitk.Cast(image1, sitk.sitkVectorFloat32)
        img2 = sitk.Cast(image2, sitk.sitkVectorFloat32)
    else:
        img1 = sitk.Cast(image1, sitk.sitkFloat32)
        img2 = sitk.Cast(image2, sitk.sitkFloat32)
        
    intersection_mask = mask1*mask2
    
    intersection_image = mask_image_multiply(alpha*intersection_mask, img1) + \
                         mask_image_multiply((1-alpha)*intersection_mask, img2)
    return intersection_image + mask_image_multiply(mask2-intersection_mask, img2) + \
           mask_image_multiply(mask1-intersection_mask, img1)

def combining_two_images(fixed, moving, transform):
    
    moving = sitk.Resample(moving, fixed, transform, 
                           sitk.sitkLinear, 0.0, 
                           moving.GetPixelIDValue())
    
    # Obtain foreground masks for the two images using Otsu thresholding, we use these later on.
    msk_fixed = sitk.OtsuThreshold(fixed,0,1)
    msk_moving = sitk.OtsuThreshold(moving,0,1)


    # Having identified the desired intensity range for each of the 
    # images using the GUI above, we use these values to perform intensity windowing and map the intensity values
    # to [0,255] and cast to 8-bit unsigned int
    fixed_255 = sitk.Cast(sitk.IntensityWindowing(fixed, windowMinimum=2, windowMaximum=657, 
                                                outputMinimum=0.0, outputMaximum=255.0), sitk.sitkUInt8)
    moving_255 = sitk.Cast(sitk.IntensityWindowing(moving, windowMinimum=-1018, windowMaximum=1126, 
                                                outputMinimum=0.0, outputMaximum=255.0), sitk.sitkUInt8)

   # Combine the two volumes
    images_list = [(alpha_blend(fixed_255, moving_255), 'alpha_blend_standard'), 
                (alpha_blend(fixed_255, moving_255, mask1=msk_fixed), 'alpha_blend_mask_fixed'),
                (alpha_blend(fixed_255, moving_255, mask2=msk_moving),'alpha_blend_mask_moving'),
                (alpha_blend(fixed_255, moving_255, mask1=msk_fixed, mask2=msk_moving),'alpha_blend_maskf_maskm')]

    # Tile the volumes using the x-y plane (axial slices)
    all_montages = []
    
    for img,img_name in images_list:
        print("test")
        num_slices = img.GetDepth()
        tile_w = int(np.sqrt(num_slices))
        tile_h = int(np.ceil(num_slices/tile_w))
        tile_image = sitk.Tile([img[:,:,i] for i in range(num_slices)], (tile_w, tile_h))
        sitk.WriteImage(sitk.Cast(tile_image, sitk.sitkUInt8), os.path.join(OUTPUT_DIR, img_name +'.jpg'))
        all_montages.append(tile_image)


#----------------------------------------Registration--------------------------------------------

#read the images

fixed_image =  sitk.ReadImage("Test\S14870_S2020.mha", sitk.sitkFloat32)
moving_image = sitk.ReadImage("Test\S14890_S2020.mha", sitk.sitkFloat32)

#initial alignment of the two volumes
transform = sitk.CenteredTransformInitializer(fixed_image, 
                                              moving_image, 
                                              sitk.Euler3DTransform(), 
                                              sitk.CenteredTransformInitializerFilter.GEOMETRY)

#multi-resolution rigid registration using Mutual Information
registration_method = sitk.ImageRegistrationMethod()
registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
registration_method.SetMetricSamplingPercentage(0.01)
registration_method.SetInterpolator(sitk.sitkLinear)
registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, 
                                                  numberOfIterations=100, 
                                                  convergenceMinimumValue=1e-6, 
                                                  convergenceWindowSize=10)
registration_method.SetOptimizerScalesFromPhysicalShift()
registration_method.SetShrinkFactorsPerLevel(shrinkFactors = [4,2,1])
registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2,1,0])
registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
registration_method.SetInitialTransform(transform)

#combining two images and generating overlay result
combining_two_images(fixed_image,moving_image,transform)

registration_method.Execute(fixed_image, moving_image)

print((moving_image))
# save registered moving volumes as .mat
# scipy save to .mat 
result = sitk.GetArrayFromImage(moving_image)
savemat('result.mat', {'result': result})
imgplot = plt.imshow(result[:,:,0])
#print(np.shape(result))

#mdic = {"moving_image": result, "label": "test"}
#savemat("matlab_matrix.mat", mdic)



sitk.WriteTransform(transform, os.path.join(OUTPUT_DIR, '3drigidT.tfm'))
