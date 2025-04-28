import argparse
import torch

def to_deployment(path: str, output: str):
    state = torch.load(path)
    torch.save({
        "model_state": state["model_state"]
    }, output)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="result/ginka.pth")
    parser.add_argument("--output", type=str, default="result/ginka_deploy.pth")
    args = parser.parse_args()
    to_deployment(args.input, args.output)
    