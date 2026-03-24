"""world_model_lens CLI — entry point registered as `wml`."""

from __future__ import annotations

import typer
from rich import print as rprint

from world_model_lens import __version__

app = typer.Typer(
    name="wml",
    help="world_model_lens: interpretability & analysis toolkit for world models.",
    add_completion=True,
    rich_markup_mode="rich",
)


# ---------------------------------------------------------------------------
# Top-level commands
# ---------------------------------------------------------------------------

@app.command()
def version() -> None:
    """Print the installed version of world_model_lens."""
    rprint(f"[bold cyan]world_model_lens[/] [green]{__version__}[/]")


@app.command()
def info() -> None:
    """Show runtime information (device, versions)."""
    import torch

    from world_model_lens.utils.device import get_device

    device = get_device()
    rprint(f"[bold]world_model_lens[/] v{__version__}")
    rprint(f"  torch      : {torch.__version__}")
    rprint(f"  device     : {device}")
    rprint(f"  cuda avail : {torch.cuda.is_available()}")


# ---------------------------------------------------------------------------
# Sub-command groups (stubs — expand as submodules are implemented)
# ---------------------------------------------------------------------------

probe_app = typer.Typer(help="Train and evaluate linear probes.")
app.add_typer(probe_app, name="probe")

patch_app = typer.Typer(help="Run activation patching experiments.")
app.add_typer(patch_app, name="patch")

sae_app = typer.Typer(help="Sparse autoencoder training and analysis.")
app.add_typer(sae_app, name="sae")

bench_app = typer.Typer(help="Run world-model benchmarks.")
app.add_typer(bench_app, name="bench")


@probe_app.command("train")
def probe_train(
    model: str = typer.Argument(..., help="Model identifier or path."),
    env: str = typer.Option("CartPole-v1", help="Gymnasium environment id."),
) -> None:
    """[stub] Train a linear probe on a world model's latent space."""
    rprint(f"[yellow]probe train[/] — stub (model={model!r}, env={env!r})")


@sae_app.command("train")
def sae_train(
    model: str = typer.Argument(..., help="Model identifier or path."),
    n_features: int = typer.Option(4096, help="Number of SAE features."),
) -> None:
    """[stub] Train a sparse autoencoder on model activations."""
    rprint(f"[yellow]sae train[/] — stub (model={model!r}, n_features={n_features})")


@bench_app.command("run")
def bench_run(
    suite: str = typer.Argument("all", help="Benchmark suite name."),
) -> None:
    """[stub] Run a benchmark suite."""
    rprint(f"[yellow]bench run[/] — stub (suite={suite!r})")


if __name__ == "__main__":
    app()
