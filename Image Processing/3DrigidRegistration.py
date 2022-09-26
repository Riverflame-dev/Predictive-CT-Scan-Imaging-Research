import SimpleITK as sitk
import numpy as np
import gui

def make_isotropic(image, interpolator = sitk.sitkLinear):
    '''
    Resample an image to isotropic pixels (using smallest spacing from original) and save to file. Many file formats 
    (jpg, png,...) expect the pixels to be isotropic. By default the function uses a linear interpolator. For
    label images one should use the sitkNearestNeighbor interpolator so as not to introduce non-existant labels.
    '''
    original_spacing = image.GetSpacing()
    # Image is already isotropic, just return a copy.
    if all(spc == original_spacing[0] for spc in original_spacing):
        return sitk.Image(image)
    # Make image isotropic via resampling.
    original_size = image.GetSize()
    min_spacing = min(original_spacing)
    new_spacing = [min_spacing]*image.GetDimension()
    new_size = [int(round(osz*(ospc/min_spacing))) for osz,ospc in zip(original_size, original_spacing)]
    resampled_img = sitk.Resample(image, new_size, sitk.Transform(), interpolator,
                         image.GetOrigin(), new_spacing, image.GetDirection(), 0.0,
                         image.GetPixelID())
    return sitk.Cast(sitk.RescaleIntensity(resampled_img), 
                                    sitk.sitkUInt8)

def save_combined_central_slice(fixed, moving, transform, file_name_prefix):
    global iteration_number 
    alpha = 0.7

    central_indexes = np.array([i/2 for i in fixed.GetSize()])
    central_indexes = np.around(central_indexes)
    central_indexes = central_indexes.astype(int)

    central_indexes_x = central_indexes[0].item()
    central_indexes_y = central_indexes[1].item()
    central_indexes_z = central_indexes[2].item()

    moving_transformed = sitk.Resample(moving, fixed, transform, 
                                       sitk.sitkLinear, 0.0, 
                                       moving_image.GetPixelIDValue())

    #extract the central slice in xy, xz, yz and alpha blend them                                   
    combined = [(1.0 - alpha)*fixed[:,:,central_indexes_z] + \
                   alpha*moving_transformed[:,:,central_indexes_z],
                  (1.0 - alpha)*fixed[:,central_indexes_y,:] + \
                  alpha*moving_transformed[:,central_indexes_y,:],
                  (1.0 - alpha)*fixed[central_indexes_x,:,:] + \
                  alpha*moving_transformed[central_indexes_x,:,:]]
    
    #resample the alpha blended images to be isotropic and rescale intensity
    #values so that they are in [0,255], this satisfies the requirements 
    #of the jpg format 
    combined_isotropic = []
    for i, img in enumerate(combined):
        combined[i] = make_isotropic(img)
    
    #tile the three images into one large image and save using the given file 
    #name prefix and the iteration number
    #print(file_name_prefix+ format(iteration_number, '03d') + '.jpg')
    gui.multi_image_display2D([sitk.Tile(combined,(1,3))],  
                          figure_size=(4,4),horizontal=False);
    
    saving_path = 'output/S14870_S3020toS2020.jpg'
    sitk.WriteImage(sitk.Tile(combined, (1,3)), 
                    file_name_prefix+ format(iteration_number, '03d') + '.jpg')
    iteration_number+=1
        



#read the images
fixed_image =  sitk.ReadImage("C:\LiverProject\S14870_S2020.mha", sitk.sitkFloat32)
moving_image = sitk.ReadImage("C:\LiverProject\S14870_S3020.mha", sitk.sitkFloat32) 

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

#add iteration callback, save central slice in xy, xz, yz planes
global iteration_number
iteration_number = 0
registration_method.AddCommand(sitk.sitkIterationEvent, 
                               lambda: save_combined_central_slice(fixed_image,
                                                                   moving_image,
                                                                   transform, 
                                                                   'output/iteration'))

registration_method.Execute(fixed_image, moving_image)

sitk.WriteTransform(transform, 'output/ct2mrT1.tfm')