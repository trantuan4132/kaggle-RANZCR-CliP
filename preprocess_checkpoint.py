import torch
import timm
import argparse
import importlib

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', type=str, default='checkpoint/convnext_tiny_1k_224_ema.pth')
    parser.add_argument('--model', type=str, default='convnext')
    parser.add_argument('--variant', type=str, default='convnext_tiny')
    return parser.parse_args()

def main():
    args = parse_args()
    mod = importlib.import_module(f'timm.models.{args.model}')
    if hasattr(mod, 'checkpoint_filter_fn'):
        model = timm.create_model(args.variant, pretrained=False)
        checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
        checkpoint = mod.checkpoint_filter_fn(checkpoint, model)
        new_path = args.checkpoint_path.replace('.pth', '_altered.pth')
        torch.save(checkpoint, new_path)
        print("Preprocessed checkpoint saved to", new_path)
    else:
        print("No preprocessing function found or no need to preprocess")

if __name__ == "__main__":
    main()