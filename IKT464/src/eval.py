"""
EVALUATION SCRIPT


Usage example:
    python eval.py \
        --ckpt_path "./checkpoints_v74_hier_gating_W222_D24/best_epoch=29_val_wer_clean=0.246.ckpt" \
        --config_path "./checkpoints_v74_hier_gating_W222_D24/config.json" \
        --data_path "../../hub_data/librispeech"
"""

import argparse
import lightning as L
from torch.utils.data import DataLoader
from datasets import load_from_disk
import torch
from IKT464_AST_SSSSM import LitASR, LibriSpeechDataset, collate_fn, Config



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', type=str, required=True, help="Path to best_model.ckpt")
    parser.add_argument('--config_path', type=str, default=None, help="Path to config.json (optional if params match)")
    parser.add_argument('--data_path', type=str, default="../hub_data/librispeech", help="Path to dataset")
    args = parser.parse_args()

    # 1. Load the model from the checkpoint
    print(f"Loading model from: {args.ckpt_path}")
    model = LitASR.load_from_checkpoint(args.ckpt_path)
    model.eval()
    
    # 2. Load the Test Data
    print(f"Loading data from: {args.data_path}")
    ds_dict_clean = load_from_disk(args.data_path)
    
    # Use the 'test' split from the saved dictionary
    test_dataset = ds_dict_clean["test"]

    # 3. Initialize the Dataset Wrapper
    test_ds = LibriSpeechDataset(
        test_dataset, 
        split_name="TEST-CLEAN",
        use_specaugment=False # No augmentation during inference
    )

    # 4. Create DataLoader
    test_loader = DataLoader(
        test_ds, 
        batch_size=32, 
        collate_fn=collate_fn, 
        num_workers=4, 
        pin_memory=True
    )

    # 5. Run Inference
    trainer = L.Trainer(accelerator='gpu', devices=1, logger=False)
    print("\nRunning evaluation on TEST-CLEAN...")
    trainer.test(model, dataloaders=test_loader)

if __name__ == "__main__":
    main()
