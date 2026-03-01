# wandb_writer.py
import os
from gxl import ExperimentConfig


class WandBWriter:
    def __init__(self, cfg: ExperimentConfig):
        import wandb
        self.wandb = wandb
        run_dir = cfg.log_dir

        # Check if wandb is already initialized (e.g., from a sweep agent)
        if wandb.run is not None:
            # Already initialized by sweep agent, just update config
            self.run = wandb.run
            wandb.config.update(cfg.parameter_dict(), allow_val_change=True)
        else:
            # Normal initialization — name matches the directory name
            self.run = wandb.init(
                project='SS-GNN',
                name=cfg.name,
                config=cfg.parameter_dict(),
                dir=run_dir,
                tags=[cfg.dataset_name, cfg.model_name, cfg.model_config.mpnn_type]
            )

    def add_scalar(self, tag, value, step=None):
        self.wandb.log({tag: value}, step=step)

    def add_text(self, tag, text, step=None):
        self.wandb.log({tag: self.wandb.Html(f"<pre>{text}</pre>")}, step=step)

    def add_histogram(self, tag, values, step=None):
        self.wandb.log({tag: self.wandb.Histogram(values)}, step=step)

    def add_image(self, tag, img, step=None):
        self.wandb.log({tag: self.wandb.Image(img)}, step=step)

    def watch(self, model, log="all", log_freq=200):
        self.wandb.watch(model, log=log, log_freq=log_freq)

    def close(self):
        self.wandb.finish()

class DummyWriter:
    def __init__(self):
        pass

    def add_scalar(self, tag, value, step=None):
        pass

    def add_text(self, tag, text, step=None):
        pass

    def add_histogram(self, tag, values, step=None):
        pass

    def add_image(self, tag, img, step=None):
        pass

    def watch(self, model, log="all", log_freq=200):
        pass

    def close(self):
        pass
