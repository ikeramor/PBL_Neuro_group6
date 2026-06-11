import nibabel as nib
import sys
import os

def get_pixel_dimensions(filepath):
    if not os.path.exists(filepath):
        print(f"Error: File '{filepath}' not found.")
        return
    
    try:
        # Load the NIfTI file
        img = nib.load(filepath)
        
        # Get the header
        header = img.header
        
        # In the NIfTI header, 'pixdim' contains the voxel dimensions.
        # pixdim[0] is a special value (qfac), so the actual x, y, z dimensions 
        # are at indices 1, 2, and 3.
        pixdims = header['pixdim'][1:4]
        
        print(f"File: {filepath}")
        print(f"Pixel/Voxel Dimensions (x, y, z): {pixdims[0]:.4f} x {pixdims[1]:.4f} x {pixdims[2]:.4f} mm")
        
        # It's also often useful to know the total number of pixels/voxels in each dimension
        print(f"Image Shape (Number of voxels): {img.shape}")
        
    except Exception as e:
        print(f"Error reading file: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python get_nifti_dims.py <path_to_nifti_file.nii or .nii.gz>")
    else:
        filepath = sys.argv[1]
        get_pixel_dimensions(filepath)
