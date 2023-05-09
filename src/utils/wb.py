try:
    import wandb as wb
    assert hasattr(wb, '__version__')
except (ImportError, AssertionError):
    wb = None

def on_train_start(args):
    """Initiate and start project if module is present"""
    wb.init(project=args.project or 'DCGAN',
            name=args.experiment_name,
            config=args) if not wb.run else wb.run

def in_train_epoch(args = dict, step = int):
    """Log metrics each training epoch."""
    wb.run.log(args, step)

def end_train_epoch(args = dict, step = int):
    """Log metrics and save images at the end of each training epoch."""
    wb.run.log(args, step)

def log_image(image = dict):
    wb.run.log({key : wb.Image(value) for key, value in image.items()})
