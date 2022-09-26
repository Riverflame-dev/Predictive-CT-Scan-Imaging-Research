import SimpleITK as sitk
import glob
import os

#constants:
DATASET =r'dataset'
SAMPLES = ['S14870','S14890']
VOLUMES = ['S2020']
#,'S3020','S4020','S5020'
new_spacing = [0.697265625, 0.697265625, 1.5]

for sample in SAMPLES:
    for volume in VOLUMES:
        path = DATASET+'\\'+sample+'\\Masks\\'+volume + '\\*.jpg'
        sample_images = [sitk.ReadImage(file, imageIO="JPEGImageIO")  for file in sorted(glob.glob(path))]
        for image in sample_images:
            image.SetSpacing([0.697265625, 0.697265625])
        sample_vol = sitk.JoinSeries(sample_images) 
        print(sample_vol.GetSpacing())
        sample_vol.SetSpacing([0.697265625, 0.697265625, 1.5])
        print(sample_vol.GetSpacing())
        # fix spacing for sample_vol
        print(sample_vol)

        volume_name = 'Test\\'+ sample + '_' + volume + '.mha'
        sitk.WriteImage(sample_vol, volume_name)





