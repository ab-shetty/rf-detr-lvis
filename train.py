import os
import zipfile
import tarfile
import argparse
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
    
    # Determine file type and extract accordingly
    if archive_path.endswith('.tar.gz') or archive_path.endswith('.tgz'):
        print(f"  Detected tar.gz archive")
        try:
            print(f"  Extracting tar.gz to {extract_to}...")
            with tarfile.open(archive_path, 'r:gz') as tar_ref:
                # Get list of members
                members = tar_ref.getmembers()
                print(f"  Found {len(members)} files in archive")
                
                # Extract all
                tar_ref.extractall(path=extract_to)
                print(f"✅ Dataset extracted to {extract_to}")
        except Exception as e:
            raise Exception(f"Error extracting tar.gz file: {e}")
    
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
            else:
                print(f"  System unzip failed: {result.stderr}")
                raise Exception("System unzip failed")
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
            print(f"  System unzip not available or failed: {e}")
            print(f"  Falling back to Python zipfile...")
            
            try:
                # Use allowZip64=True for large files
                with zipfile.ZipFile(archive_path, 'r', allowZip64=True) as zip_ref:
                    # Get list of files
                    file_list = zip_ref.namelist()
                    print(f"  Found {len(file_list)} files in archive")
                    
                    # Extract with progress indication
                    print(f"  Extracting to {extract_to}...")
                    for i, file in enumerate(file_list):
                        zip_ref.extract(file, extract_to)
                        if (i + 1) % 1000 == 0:
                            print(f"    Extracted {i + 1}/{len(file_list)} files...")
                    
                    print(f"✅ Dataset extracted to {extract_to}")
            
            except zipfile.BadZipFile:
                raise Exception(f"Error: {archive_path} is not a valid zip file or is corrupted")
            except OSError as e:
                raise Exception(f"Error extracting zip file: {e}. The file may be corrupted.")
    else:
        raise Exception(f"Unsupported archive format: {archive_path}. Expected .tar.gz or .zip")
    
    # Verify the structure
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


def train_rfdetr(args):
    """Train RF-DETR model on LVIS dataset."""
    print("="*60)
    print("RF-DETR Training on LVIS Dataset")
    print("="*60)
    
    # Initialize model
    print("\n📦 Initializing RF-DETR Small model...")
    model = RFDETRSmall()
    
    # Find and extract dataset
    archive_files = [f for f in os.listdir(args.dataset_dir) 
                     if f.endswith(('.zip', '.tar.gz', '.tgz'))]
    if not archive_files:
        raise FileNotFoundError(f"No archive file (.zip, .tar.gz) found in {args.dataset_dir}")
    
    dataset_archive = os.path.join(args.dataset_dir, archive_files[0])
    print(f"\n📂 Found dataset: {archive_files[0]}")
    
    # Extract dataset (already in correct RF-DETR format)
    dataset_dir = extract_dataset(dataset_archive)
    
    # Verify dataset structure
    if not os.path.exists(os.path.join(dataset_dir, 'train')):
        raise FileNotFoundError(f"Train directory not found in {dataset_dir}")
    if not os.path.exists(os.path.join(dataset_dir, 'valid')):
        raise FileNotFoundError(f"Valid directory not found in {dataset_dir}")
    
    print(f"\n🎯 Starting training...")
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
