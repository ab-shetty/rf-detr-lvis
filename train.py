import os
import zipfile
import argparse
from pathlib import Path
from rfdetr import RFDETRSmall


def extract_dataset(zip_path, extract_to="/tmp/dataset"):
    """Extract the LVIS dataset zip file."""
    print(f"Extracting dataset from {zip_path}...")
    os.makedirs(extract_to, exist_ok=True)
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    
    print(f"✅ Dataset extracted to {extract_to}")
    
    # The dataset should be in lvis_rfdetr_dataset/ folder after extraction
    dataset_dir = os.path.join(extract_to, 'lvis_rfdetr_dataset')
    
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
    zip_files = [f for f in os.listdir(args.dataset_dir) if f.endswith('.zip')]
    if not zip_files:
        raise FileNotFoundError(f"No zip file found in {args.dataset_dir}")
    
    dataset_zip = os.path.join(args.dataset_dir, zip_files[0])
    print(f"\n📂 Found dataset: {zip_files[0]}")
    
    # Extract dataset (already in correct RF-DETR format)
    dataset_dir = extract_dataset(dataset_zip)
    
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
