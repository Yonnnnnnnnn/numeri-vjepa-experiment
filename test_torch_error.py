import torch

try:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Try forcing error
    t = torch.tensor([1])
    try:
        t.to(dtype=device)
    except TypeError as e:
        print(f"Caught expected error: {e}")
    except Exception as e:
        print(f"Caught unexpected error: {e}")

except Exception as e:
    print(f"Failed to run test: {e}")
