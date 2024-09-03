import logging
import platform

from invoke import Context

NOT_WINDOWS = platform.system() != "Windows"


def on_ovartaci() -> bool:
    import platform

    if platform.node() == "RMAPPS1279":
        logging.info("On Ovartaci")
        return True

    logging.info("Not on Ovartaci")
    return False


def test_pytorch_cuda(c: Context):
    python_command = "import torch; t=torch.tensor(1); t.to(torch.device('cuda'))"
    c.run(f'python -c "{python_command}"', pty=NOT_WINDOWS)
    print("Pytorch CUDA works!")
