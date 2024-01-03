from torch.optim import lr_scheduler


def get_scheduler(scheduler_str) -> object:
    scheduler = None

    if scheduler_str == "CosineAnnealingLR":
        scheduler = lr_scheduler.CosineAnnealingLR

    elif scheduler_str == "CosineAnnealingWarmRestarts":
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts
    elif scheduler_str == "StepLR":
        scheduler = lr_scheduler.StepLR
    elif scheduler_str == "MultiStepLR":
        scheduler = lr_scheduler.MultiStepLR

    return scheduler
