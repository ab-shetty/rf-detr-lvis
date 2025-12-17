import os
import zipfile
import tarfile
import argparse
import json
from pathlib import Path
from rfdetr import RFDETRSmall


def extract_dataset(archive_path, extract_to="/tmp/dataset"):
    """Extract the LVIS dataset from tar.gz or zip file."""
    
    # Check if already extracted
    dataset_dir = os.path.join(extract_to, 'lvis_rfdetr_dataset')
    if os.path.exists(dataset_dir) and os.path.exists(os.path.join(dataset_dir, 'train')):
        print(f"✅ Dataset already extracted at {dataset_dir}")
        return dataset_dir
    
    print(f"Extracting dataset from {archive_path}...")
    
    # Check if file exists and get size
    if not os.path.exists(archive_path):
        raise FileNotFoundError(f"Dataset archive file not found: {archive_path}")
    
    file_size_gb = os.path.getsize(archive_path) / (1024**3)
    print(f"  Archive file size: {file_size_gb:.2f} GB")
    
    os.makedirs(extract_to, exist_ok=True)
    
    # Check file header to determine actual type
    with open(archive_path, 'rb') as f:
        header = f.read(10)
        print(f"  File header (hex): {header[:10].hex()}")
    
    # Determine file type and extract accordingly
    if archive_path.endswith('.tar.gz') or archive_path.endswith('.tgz') or archive_path.endswith('.tar'):
        print(f"  Detected tar archive")
        
        # Strategy 1: Try system tar command (most reliable)
        import subprocess
        try:
            print(f"  Attempting extraction with system tar command...")
            result = subprocess.run(
                ['tar', '-xzf', archive_path, '-C', extract_to],
                capture_output=True,
                text=True,
                timeout=7200  # 2 hour timeout for large file
            )
            if result.returncode == 0:
                print(f"✅ Dataset extracted to {extract_to}")
                return verify_and_return_dataset(dataset_dir, extract_to)
            else:
                print(f"  System tar failed: {result.stderr}")
                print(f"  Trying alternative methods...")
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
            print(f"  System tar not available or failed: {e}")
        
        # Strategy 2: Try Python tarfile with auto-detection
        try:
            print(f"  Trying Python tarfile with auto-detection...")
            with tarfile.open(archive_path, 'r:*') as tar_ref:
                members = tar_ref.getmembers()
                print(f"  Found {len(members)} files in archive")
                tar_ref.extractall(path=extract_to)
                print(f"✅ Dataset extracted to {extract_to}")
                return verify_and_return_dataset(dataset_dir, extract_to)
        except tarfile.ReadError as e:
            print(f"  Auto-detection failed: {e}")
        
        # Strategy 3: Try as uncompressed tar
        try:
            print(f"  Trying as uncompressed tar...")
            with tarfile.open(archive_path, 'r:') as tar_ref:
                members = tar_ref.getmembers()
                print(f"  Found {len(members)} files in archive")
                tar_ref.extractall(path=extract_to)
                print(f"✅ Dataset extracted to {extract_to}")
                return verify_and_return_dataset(dataset_dir, extract_to)
        except Exception as e:
            print(f"  Uncompressed tar failed: {e}")
        
        # Strategy 4: Try manually with gzip and tar
        try:
            import gzip
            import shutil
            print(f"  Trying manual gzip decompression...")
            temp_tar = archive_path.replace('.gz', '.tmp')
            with gzip.open(archive_path, 'rb') as f_in:
                with open(temp_tar, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            
            with tarfile.open(temp_tar, 'r:') as tar_ref:
                members = tar_ref.getmembers()
                print(f"  Found {len(members)} files in archive")
                tar_ref.extractall(path=extract_to)
            
            os.remove(temp_tar)
            print(f"✅ Dataset extracted to {extract_to}")
            return verify_and_return_dataset(dataset_dir, extract_to)
        except Exception as e:
            print(f"  Manual gzip decompression failed: {e}")
        
        raise Exception(f"All extraction methods failed for tar archive")
    
    elif archive_path.endswith('.zip'):
        print(f"  Detected zip archive")
        # Try using system unzip command first (more reliable for large files)
        import subprocess
        try:
            print(f"  Attempting extraction with system unzip command...")
            result = subprocess.run(
                ['unzip', '-q', '-o', archive_path, '-d', extract_to],
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )
            if result.returncode == 0:
                print(f"✅ Dataset extracted to {extract_to}")
                return verify_and_return_dataset(dataset_dir, extract_to)
            else:
                print(f"  System unzip failed: {result.stderr}")
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
            print(f"  System unzip not available or failed: {e}")
        
        # Fallback to Python zipfile
        try:
            print(f"  Falling back to Python zipfile...")
            with zipfile.ZipFile(archive_path, 'r', allowZip64=True) as zip_ref:
                file_list = zip_ref.namelist()
                print(f"  Found {len(file_list)} files in archive")
                zip_ref.extractall(extract_to)
                print(f"✅ Dataset extracted to {extract_to}")
                return verify_and_return_dataset(dataset_dir, extract_to)
        except Exception as e:
            raise Exception(f"Error extracting zip file: {e}")
    else:
        raise Exception(f"Unsupported archive format: {archive_path}")


def verify_and_return_dataset(dataset_dir, extract_to):
    """Verify dataset structure and return path."""
    if os.path.exists(dataset_dir):
        print(f"✅ Found dataset directory: {dataset_dir}")
        for split in ['train', 'valid']:
            split_dir = os.path.join(dataset_dir, split)
            anno_file = os.path.join(split_dir, '_annotations.coco.json')
            if os.path.exists(split_dir):
                num_images = len([f for f in os.listdir(split_dir) if f.endswith('.jpg')])
                print(f"  - {split}: {num_images} images, annotations: {os.path.exists(anno_file)}")
            else:
                print(f"  ⚠️  {split} directory not found!")
    else:
        print(f"⚠️  Expected directory not found: {dataset_dir}")
        print(f"Available directories in {extract_to}:")
        for item in os.listdir(extract_to):
            print(f"  - {item}")
    
    return dataset_dir


def fix_lvis_annotations(dataset_dir):
    """
    Fix LVIS annotations for RF-DETR compatibility:
    1. Add 'supercategory' field
    2. Remap category IDs to be contiguous (0 to N-1)
    """
    print("\n🔧 Checking and fixing LVIS annotations for RF-DETR compatibility...")
    
    # First pass: collect all unique category IDs across splits
    all_categories = {}
    
    for split in ['train', 'valid']:
        anno_path = os.path.join(dataset_dir, split, '_annotations.coco.json')
        
        if not os.path.exists(anno_path):
            continue
        
        with open(anno_path, 'r') as f:
            data = json.load(f)
        
        for cat in data['categories']:
            if cat['id'] not in all_categories:
                all_categories[cat['id']] = cat
    
    # Create mapping from old IDs to new contiguous IDs (0 to N-1)
    old_to_new_id = {}
    sorted_cat_ids = sorted(all_categories.keys())
    
    for new_id, old_id in enumerate(sorted_cat_ids):
        old_to_new_id[old_id] = new_id
    
    print(f"  📊 Found {len(all_categories)} categories")
    print(f"  🔄 Remapping category IDs: {min(sorted_cat_ids)}-{max(sorted_cat_ids)} → 0-{len(all_categories)-1}")
    
    # Second pass: fix each split
    for split in ['train', 'valid']:
        anno_path = os.path.join(dataset_dir, split, '_annotations.coco.json')
        
        if not os.path.exists(anno_path):
            print(f"  ⚠️  Annotation file not found: {anno_path}")
            continue
        
        # Load annotations
        with open(anno_path, 'r') as f:
            data = json.load(f)
        
        # Check if already fixed
        if (data['categories'] and 
            'supercategory' in data['categories'][0] and
            data['categories'][0]['id'] == 0):
            print(f"  ✅ {split} annotations already fixed")
            continue
        
        print(f"  📝 Fixing {split} annotations...")
        
        # Fix categories: add supercategory and remap IDs
        for category in data['categories']:
            old_id = category['id']
            category['id'] = old_to_new_id[old_id]
            category['supercategory'] = 'object'
        
        # Fix annotations: remap category IDs
        for annotation in data['annotations']:
            old_cat_id = annotation['category_id']
            annotation['category_id'] = old_to_new_id[old_cat_id]
        
        # Save fixed annotations
        with open(anno_path, 'w') as f:
            json.dump(data, f)
        
        print(f"  ✅ Fixed {len(data['categories'])} categories and {len(data['annotations'])} annotations in {split}")
    
    print(f"  ✅ Category ID remapping complete")


def train_rfdetr(args):
    """Train RF-DETR model on LVIS dataset."""
    print("="*60)
    print("RF-DETR Training on LVIS Dataset")
    print("="*60)
    
    # Initialize model
    print("\n📦 Initializing RF-DETR Small model...")
    model = RFDETRSmall()
    
    # Check if dataset is already extracted or needs extraction
    dataset_dir = None
    
    # First, check if dataset is already in the expected structure (from HuggingFace)
    possible_dataset_dir = os.path.join(args.dataset_dir, 'lvis_rfdetr_dataset')
    if os.path.exists(possible_dataset_dir) and os.path.exists(os.path.join(possible_dataset_dir, 'train')):
        print(f"\n📂 Found pre-extracted dataset at: {possible_dataset_dir}")
        dataset_dir = possible_dataset_dir
    
    # Or check if the dataset_dir itself is the dataset root
    elif os.path.exists(os.path.join(args.dataset_dir, 'train')) and os.path.exists(os.path.join(args.dataset_dir, 'valid')):
        print(f"\n📂 Found pre-extracted dataset at: {args.dataset_dir}")
        dataset_dir = args.dataset_dir
    
    # Otherwise, look for archive files to extract
    else:
        archive_files = [f for f in os.listdir(args.dataset_dir) 
                         if f.endswith(('.zip', '.tar.gz', '.tgz', '.tar'))]
        if not archive_files:
            raise FileNotFoundError(
                f"No dataset found in {args.dataset_dir}. "
                f"Expected either:\n"
                f"  - Pre-extracted dataset with train/ and valid/ directories\n"
                f"  - Archive file (.zip, .tar.gz) to extract"
            )
        
        dataset_archive = os.path.join(args.dataset_dir, archive_files[0])
        print(f"\n📂 Found dataset archive: {archive_files[0]}")
        
        # Extract dataset (already in correct RF-DETR format)
        dataset_dir = extract_dataset(dataset_archive)
    
    # Verify dataset structure
    if not os.path.exists(os.path.join(dataset_dir, 'train')):
        raise FileNotFoundError(f"Train directory not found in {dataset_dir}")
    if not os.path.exists(os.path.join(dataset_dir, 'valid')):
        raise FileNotFoundError(f"Valid directory not found in {dataset_dir}")
    
    # Show dataset info
    for split in ['train', 'valid']:
        split_dir = os.path.join(dataset_dir, split)
        anno_file = os.path.join(split_dir, '_annotations.coco.json')
        if os.path.exists(split_dir):
            num_images = len([f for f in os.listdir(split_dir) if f.endswith('.jpg')])
            print(f"  - {split}: {num_images} images, annotations: {os.path.exists(anno_file)}")
    
    # Fix LVIS annotations for RF-DETR compatibility
    fix_lvis_annotations(dataset_dir)
    
    # Create test directory as symlink to valid (RF-DETR expects test split)
    test_dir = os.path.join(dataset_dir, 'test')
    valid_dir = os.path.join(dataset_dir, 'valid')
    if not os.path.exists(test_dir) and os.path.exists(valid_dir):
        print(f"\n🔗 Creating test directory as symlink to valid...")
        os.symlink(valid_dir, test_dir, target_is_directory=True)
        print(f"  ✅ Test directory created (points to valid)")
    
    print(f"\n🎯 Starting training...")
    print(f"  - Dataset: {dataset_dir}")
    print(f"  - Epochs: {args.epochs}")
    print(f"  - Batch size: {args.batch}")
    print(f"  - Gradient accumulation steps: {args.grad_accum_steps}")
    print(f"  - Learning rate: {args.lr}")
    print(f"  - Output directory: {args.out_dir}")
    print()
    
    # Train model
    model.train(
        dataset_dir=dataset_dir,
        epochs=args.epochs,
        batch_size=args.batch,
        grad_accum_steps=args.grad_accum_steps,
        lr=args.lr,
        output_dir=args.out_dir
    )
    
    print("\n✅ Training completed!")
    print(f"📁 Checkpoints saved to: {args.out_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Train RF-DETR on LVIS Dataset (Flex AI)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Dataset and output paths
    parser.add_argument('--dataset_dir', type=str, default='/input',
                        help='Directory containing the dataset zip file')
    parser.add_argument('--out_dir', type=str, 
                        default=os.environ.get('FLEXAI_OUTPUT_CHECKPOINT_DIR', '/output-checkpoints'),
                        help='Output directory for checkpoints')
    
    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--batch', type=int, default=16,
                        help='Batch size for training')
    parser.add_argument('--grad_accum_steps', type=int, default=4,
                        help='Gradient accumulation steps')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    
    args = parser.parse_args()
    
    # Print configuration
    print("\n" + "="*60)
    print("Configuration:")
    print("="*60)
    for arg, value in vars(args).items():
        print(f"{arg:20s}: {value}")
    print("="*60 + "\n")
    
    train_rfdetr(args)


if __name__ == '__main__':
    main()
