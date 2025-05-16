import os
import tarfile
import shutil

def move_files_to_target(source_dir, target_dir):
    """
    Move all files from subdirectories to target directory and remove empty directories.
    
    Args:
        source_dir (str): Source directory containing files and subdirectories
        target_dir (str): Target directory to move files to
    """
    # Create target directory if it doesn't exist
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    
    # Walk through all subdirectories
    for root, dirs, files in os.walk(source_dir, topdown=False):
        # Skip the target directory itself
        if root == target_dir:
            continue
            
        # Move all files to target directory
        for file in files:
            src_path = os.path.join(root, file)
            dst_path = os.path.join(target_dir, file)
            
            # If file already exists in target, add a number to the filename
            if os.path.exists(dst_path):
                base, ext = os.path.splitext(file)
                counter = 1
                while os.path.exists(dst_path):
                    new_name = f"{base}_{counter}{ext}"
                    dst_path = os.path.join(target_dir, new_name)
                    counter += 1
            
            try:
                shutil.move(src_path, dst_path)
                print(f"Moved {file} to {target_dir}")
            except Exception as e:
                print(f"Error moving {file}: {str(e)}")
        
        # Remove empty directory
        try:
            if root != source_dir:  # Don't remove the source directory itself
                os.rmdir(root)
                print(f"Removed empty directory: {root}")
        except Exception as e:
            print(f"Error removing directory {root}: {str(e)}")

def extract_tar_files(directory_path, target_dir=None, remove_tar=True):
    """
    Extract all tar files in the specified directory and optionally remove the tar files.
    
    Args:
        directory_path (str): Path to the directory containing tar files
        target_dir (str, optional): Path to extract files to. If None, extracts to the same directory.
        remove_tar (bool): Whether to remove tar files after extraction. Default is True.
    
    Returns:
        list: List of extracted tar file names
    """
    # Check if directory exists
    if not os.path.isdir(directory_path):
        print(f"Error: Directory '{directory_path}' does not exist.")
        return []
    
    # Create target directory if it doesn't exist
    if target_dir and not os.path.exists(target_dir):
        os.makedirs(target_dir)
        print(f"Created target directory: {target_dir}")
    
    extracted_files = []
    
    # Get all tar files in the directory
    tar_files = [f for f in os.listdir(directory_path) if f.endswith('.tar')]
    
    if not tar_files:
        print(f"No tar files found in '{directory_path}'.")
        return []
    
    # Extract each tar file
    for tar_file in tar_files:
        tar_path = os.path.join(directory_path, tar_file)
        print(f"Extracting {tar_file}...")
        
        try:
            with tarfile.open(tar_path, 'r') as tar:
                # Extract to target directory or same directory
                tar.extractall(path=target_dir if target_dir else directory_path)
            
            # Add to the list of successfully extracted files
            extracted_files.append(tar_file)
            
            # Remove the tar file if specified
            if remove_tar:
                os.remove(tar_path)
                print(f"Removed {tar_file} after extraction.")
            
        except Exception as e:
            print(f"Error extracting {tar_file}: {str(e)}")
    
    print(f"Extracted {len(extracted_files)} tar files.")
    
    # Move all files to target directory and remove empty directories
    if target_dir:
        print("\nMoving all files to target directory...")
        move_files_to_target(directory_path, target_dir)
    
    return extracted_files

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract tar files with options")
    parser.add_argument("--dir", type=str, default=".", help="Directory containing tar files")
    parser.add_argument("--target", type=str, default=None, help="Target directory to extract files to")
    parser.add_argument("--keep-tar", action="store_true", help="Keep tar files after extraction")
    args = parser.parse_args()
    
    if args.target is None:
        args.target = args.dir

    extract_tar_files(args.dir, args.target, not args.keep_tar)
