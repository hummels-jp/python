import torch

def test_pytorch_environment():
    print(f"PyTorch version: {torch.__version__}")
    
    # 检查CUDA是否可用
    if torch.cuda.is_available():
        print("CUDA is available!")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        print(f"Current GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA is not available. Running on CPU.")

if __name__ == "__main__":
    test_pytorch_environment()