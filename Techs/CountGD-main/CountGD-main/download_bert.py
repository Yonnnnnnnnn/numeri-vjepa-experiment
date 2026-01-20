import argparse
import os
from transformers import BertConfig, BertModel, AutoTokenizer


def main():
    parser = argparse.ArgumentParser(
        description="Download BERT weights to specific directory"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./checkpoints",
        help="Directory to save model",
    )
    args = parser.parse_args()

    # Ensure target directory is specifically for bert-base-uncased
    # If the user passes ".../checkpoints", we append "bert-base-uncased"
    # If the user passes ".../checkpoints/bert-base-uncased", we use it as is.

    target_dir = args.output_dir
    # Simple check: if path doesn't end with model name, append it
    if not target_dir.endswith("bert-base-uncased"):
        target_dir = os.path.join(target_dir, "bert-base-uncased")

    os.makedirs(target_dir, exist_ok=True)
    print(f"Downloading BERT to: {target_dir}")

    config = BertConfig.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained(
        "bert-base-uncased", add_pooling_layer=False, config=config
    )
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    config.save_pretrained(target_dir)
    model.save_pretrained(target_dir)
    tokenizer.save_pretrained(target_dir)
    print("Download complete.")


if __name__ == "__main__":
    main()
