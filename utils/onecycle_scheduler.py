from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, OneCycleLR

def get_onecycle_schedule(optimizer: Optimizer, learning_rate: float, total_steps: int):
    """
    Create a schedule with a OneCycleLR learning rate from PyTorch, using the learning rate set in optimizer.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """
    return OneCycleLR(optimizer, max_lr=learning_rate, total_steps=total_steps)