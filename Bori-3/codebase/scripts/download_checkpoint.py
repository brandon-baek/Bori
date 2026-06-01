import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Download SFT/CPT model checkpoints from W&B Artifacts")
    parser.add_argument(
        "--artifact_path", 
        type=str, 
        required=True, 
        help="W&B artifact path in the format: 'entity/project/artifact_name'"
    )
    parser.add_argument(
        "--alias", 
        type=str, 
        default="latest", 
        help="Artifact version alias (e.g. 'latest', 'final', or 'step-500')"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./downloaded_sft_checkpoint", 
        help="Local directory path to save the model weights"
    )
    args = parser.parse_args()

    print(f"🚀 Initializing W&B API...")
    try:
        import wandb
    except ImportError:
        print("❌ Error: 'wandb' package is not installed. Please run: pip install wandb")
        return

    # Check W&B authentication
    if wandb.run is None and not os.environ.get("WANDB_API_KEY"):
        print("💡 Notice: If you are not authenticated, please run 'wandb login' in your terminal first.")

    api = wandb.Api()
    full_path = f"{args.artifact_path}:{args.alias}"
    print(f"📦 Fetching W&B Artifact: {full_path} ...")
    
    try:
        artifact = api.artifact(full_path)
        print(f"📥 Downloading weights to: {args.output_dir} ...")
        artifact.download(root=args.output_dir)
        print(f"✅ Success! Checkpoint successfully downloaded to {args.output_dir}!")
        
        # Output guidance on running inference
        script_dir = os.path.dirname(os.path.abspath(__file__))
        test_script = os.path.join(script_dir, "test_inference.py")
        
        print("\n📝 To run a quick local chat test using this checkpoint, execute:")
        print(f"python3 {test_script} --model_path {args.output_dir} --tokenizer_path /path/to/merged_tokenizer")
        
    except Exception as e:
        print(f"❌ Failed to download artifact: {e}")
        print("Double-check the artifact path format, permissions, and W&B login status.")

if __name__ == "__main__":
    main()
