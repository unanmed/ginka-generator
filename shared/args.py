import argparse    
    
def parse_arguments(from_default: str, train_default: str, val_default: str):
    parser = argparse.ArgumentParser(description="training codes")
    parser.add_argument("--resume", type=bool, default=False)
    parser.add_argument("--from_state", type=str, default=from_default)
    parser.add_argument("--load_optim", type=bool, default=False)
    parser.add_argument("--train", type=str, default=train_default)
    parser.add_argument("--validate", type=str, default=val_default)
    parser.add_argument("--epochs", type=int, default=150)
    args = parser.parse_args()
    return args
