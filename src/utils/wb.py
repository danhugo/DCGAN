try:
    import wandb as wb
    assert hasattr(wb, '__version__')
except (ImportError, AssertionError):
    wb = None

def on_pretrain_start(args):
    """Initiate and start project if module is present"""
    wb.init(project=args.project or 'DCGAN',
            name=args.experiment_name,
            config=args) if not wb.run else wb.run

def on_training_epoch(args):
    """Log metrics each training epoch."""
    wb.run.log({f'train/{key}' : value for key, value in vars(args).items()}, step=args.step + 1)

def on_train_epoch_end(args):
    """Log metrics and save images at the end of each training epoch."""
    wb.run.log({'train/mean_Gloss' : args.Gloss}, step=args.step + 1)
    wb.run.log({'train/mean_Dloss' : args.Dloss}, step=args.step + 1)
    wb.run.log({'train/lr' : args.lr}, step=args.step + 1)
    wb.run.log({args.gen_image.name: wb.Image(args.gen_image.data)})

