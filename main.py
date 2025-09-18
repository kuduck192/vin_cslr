import argparse
from cslr_demo import CSLRDemo

def parse_args():
    parser = argparse.ArgumentParser(description="Run CSLR Demo")
    parser.add_argument('--model_name', type=str, default='corrnet', help='SLR model name')
    parser.add_argument('--device', type=str, default='cpu', help='Device for model: cpu or cuda')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    demo = CSLRDemo(model_name=args.model_name, device=args.device)
    demo.run()
