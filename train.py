import os
import zipfile
import tarfile
import argparse
import json
from pathlib import Path
from rfdetr import RFDETRSmall
import wandb


def extract_dataset(archive_path, extract_to="/tmp/dataset"):
    """Extract the LVIS dataset from tar.gz or zip file."""
    
    # Check if already extracted
    dataset_dir = os.path.join(extract_to, 'lvis_rfdetr_dataset')
    if os.path.exists(dataset_dir) and os.path.exists(os.path.join(dataset_dir, 'train')):
        print(f"Dataset already extracted at {dataset_dir}")
        return dataset_dir
    
    print(f"Extracting dataset from {archive_path}...")
    os.makedirs(extract_to, exist_ok=True)
    
    # Extract based on file type
    if archive_path.endswith(('.tar.gz', '.tgz')):
        with tarfile.open(archive_path, 'r:gz') as tar_ref:
            tar_ref.extractall(path=extract_to)
    elif archive_path.endswith('.tar'):
        with tarfile.open(archive_path, 'r:') as tar_ref:
            tar_ref.extractall(path=extract_to)
    elif archive_path.endswith('.zip'):
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
    else:
        raise ValueError(f"Unsupported archive format: {archive_path}")
    
    print(f"Dataset extracted to {extract_to}")
    return dataset_dir


def fix_lvis_annotations(dataset_dir):
    """
    Fix LVIS annotations for RF-DETR compatibility:
    1. Add 'supercategory' field
    2. Remap category IDs to be contiguous (0 to N-1)
    3. Add 'iscrowd' field to annotations
    """
    print("Fixing LVIS annotations for RF-DETR compatibility...")
    
    # Collect all unique category IDs
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
    
    # Create ID mapping (1-N → 0 to N-1)
    old_to_new_id = {old_id: new_id for new_id, old_id in enumerate(sorted(all_categories.keys()))}
    
    # Fix each split
    for split in ['train', 'valid']:
        anno_path = os.path.join(dataset_dir, split, '_annotations.coco.json')
        if not os.path.exists(anno_path):
            continue
        
        with open(anno_path, 'r') as f:
            data = json.load(f)
        
        # Check if already fixed
        if (data['categories'] and data['categories'][0]['id'] == 0 and 
            'supercategory' in data['categories'][0] and
            data['annotations'] and 'iscrowd' in data['annotations'][0]):
            continue
        
        # Fix categories
        for category in data['categories']:
            category['id'] = old_to_new_id[category['id']]
            category['supercategory'] = 'object'
        
        # Fix annotations
        for annotation in data['annotations']:
            annotation['category_id'] = old_to_new_id[annotation['category_id']]
            # Add iscrowd field if missing (LVIS doesn't have crowd annotations)
            if 'iscrowd' not in annotation:
                annotation['iscrowd'] = 0
        
        with open(anno_path, 'w') as f:
            json.dump(data, f)
    
    print(f"  ✅ Fixed {len(all_categories)} categories")

def train_rfdetr(args):
    """Train RF-DETR model on LVIS dataset."""

    # Set W&B entity before RF-DETR initializes
    os.environ['WANDB_ENTITY'] = 'ashetty21-university-of-california-berkeley'
    wandb.login(key=os.environ.get('WANDB_API_KEY'))
    
    # Initialize model
    print("Initializing RF-DETR Small model...")
    model = RFDETRSmall()
    
    # Locate dataset
    dataset_dir = None
    possible_dataset_dir = os.path.join(args.dataset_dir, 'lvis_rfdetr_dataset')
    
    if os.path.exists(possible_dataset_dir) and os.path.exists(os.path.join(possible_dataset_dir, 'train')):
        dataset_dir = possible_dataset_dir
    elif os.path.exists(os.path.join(args.dataset_dir, 'train')) and os.path.exists(os.path.join(args.dataset_dir, 'valid')):
        dataset_dir = args.dataset_dir
    else:
        # Look for archive to extract
        archive_files = [f for f in os.listdir(args.dataset_dir) 
                         if f.endswith(('.zip', '.tar.gz', '.tgz', '.tar'))]
        if not archive_files:
            raise FileNotFoundError(f"No dataset found in {args.dataset_dir}")
        
        dataset_archive = os.path.join(args.dataset_dir, archive_files[0])
        print(f"Extracting: {archive_files[0]}")
        dataset_dir = extract_dataset(dataset_archive)
    
    print(f"Dataset: {dataset_dir}")
    
    # Fix annotations and create test symlink
    fix_lvis_annotations(dataset_dir)
    
    test_dir = os.path.join(dataset_dir, 'test')
    valid_dir = os.path.join(dataset_dir, 'valid')
    if not os.path.exists(test_dir):
        os.symlink(valid_dir, test_dir, target_is_directory=True)
    
    print(f"\n Starting training...")
    print(f"  - Epochs: {args.epochs}")
    print(f"  - Batch size: {args.batch}")
    print(f"  - Grad accum: {args.grad_accum_steps}")
    print(f"  - Learning rate: {args.lr}")
    print()
    
    # Train
    model.train(
        dataset_dir=dataset_dir,
        epochs=args.epochs,
        batch_size=args.batch,
        grad_accum_steps=args.grad_accum_steps,
        lr=args.lr,
        output_dir=args.out_dir,
        wandb=True,
        project=args.wandb_project,
    )

    # Get the run ID that RF-DETR just used
    print("Saving model artifact to W&B...")
    api = wandb.Api()
    runs = api.runs(f"ashetty21-university-of-california-berkeley/{args.wandb_project}", order="-created_at")
    latest_run_id = runs[0].id
    
    # Resume that run to add artifact
    run = wandb.init(
        project=args.wandb_project,
        entity="ashetty21-university-of-california-berkeley",
        id=latest_run_id,
        resume="must"
    )
    
    artifact = wandb.Artifact(
        name=f"rfdetr-small-model",
        type="model",
        description=f"RF-DETR Small trained for {args.epochs} epochs"
    )
    
    # Add checkpoint files to artifact
    for file in os.listdir(args.out_dir):
        if file.endswith('_best_total.pth'):
            artifact.add_file(os.path.join(args.out_dir, file))
    
    # Log the artifact
    wandb.log_artifact(artifact)
    
    # Finish the run
    wandb.finish()
    print("Model artifact saved to W&B")
    print("Training completed!")
    print(f"Checkpoints: {args.out_dir}")


def main():
    parser = argparse.ArgumentParser(description='Train RF-DETR on LVIS Dataset')
    
    parser.add_argument('--dataset_dir', type=str, default='/input',
                        help='Directory containing the dataset')
    parser.add_argument('--out_dir', type=str, 
                        default=os.environ.get('FLEXAI_OUTPUT_CHECKPOINT_DIR', '/output-checkpoints'),
                        help='Output directory for checkpoints')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--batch', type=int, default=16,
                        help='Batch size for training')
    parser.add_argument('--grad_accum_steps', type=int, default=4,
                        help='Gradient accumulation steps')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--wandb_project', type=str, default='rfdetr-training',
                        help='W&B project name')
    
    args = parser.parse_args()
    train_rfdetr(args)


if __name__ == '__main__':
    main()
