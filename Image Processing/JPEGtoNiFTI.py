import SimpleITK as sitk
import glob




file_names = glob.glob(r"C:\Users\17912\OneDrive\Desktop\ML Stuff\masks\S2020\*.jpg")
print(file_names)
reader = sitk.ImageSeriesReader()
reader.SetFileNames(file_names)
vol = reader.Execute()
sitk.WriteImage(vol, 'volume.nii.gz')

