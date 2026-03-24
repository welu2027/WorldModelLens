"""CLI commands for World Model Lens using Typer.

This CLI supports both RL and non-RL world models:
- RL models: Full analysis including rewards, values, imagination faithfulness
- Non-RL models: Latent geometry, surprise, disentanglement, SAE analysis
"""

from typing import Optional, List, Literal
import typer
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

from world_model_lens import __version__

app = typer.Typer(
    name="wml",
    help="World Model Lens: Interpretability toolkit for world models.",
    add_completion=False,
)
console = Console()


@app.command()
def version():
    """Print version information."""
    console.print(f"[bold blue]World Model Lens[/bold blue] v{__version__}")


@app.command()
def info():
    """Show device and version information."""
    import torch
    from world_model_lens.utils.device import get_device

    device = get_device()
    console.print(f"[bold]World Model Lens[/bold] v{__version__}")
    console.print(f"PyTorch: {torch.__version__}")
    console.print(f"Device: [green]{device}[/green]")
    console.print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        console.print(f"GPU: {torch.cuda.get_device_name(0)}")


@app.command()
def analyze(
    checkpoint_path: str = typer.Argument(..., help="Path to model checkpoint"),
    env: str = typer.Option("Pendulum-v1", help="Gymnasium environment ID"),
    backend: str = typer.Option("dreamerv3", help="Backend architecture"),
    n_episodes: int = typer.Option(10, help="Number of episodes to collect"),
    output_dir: str = typer.Option("./results", help="Output directory"),
    mode: str = typer.Option("auto", help="Analysis mode: auto, rl, non-rl"),
    probes: bool = typer.Option(False, help="Run probe sweep"),
    patching_sweep: bool = typer.Option(False, help="Run patching sweep"),
    report: str = typer.Option("html", help="Report format (html/text)"),
):
    """Analyze a world model checkpoint.

    Supports both RL and non-RL world models. Use --mode auto (default) to
    automatically detect the model type based on its capabilities, or specify
    --mode rl or --mode non-rl to override.

    RL mode runs:
    - Imagination faithfulness
    - Reward attribution
    - Value analysis
    - Surprise calibration

    Non-RL mode runs:
    - Latent geometry analysis
    - Temporal memory analysis
    - Surprise timeline
    - Disentanglement metrics
    """
    from world_model_lens import HookedWorldModel, WorldModelConfig
    from world_model_lens.hub import ModelHub
    from world_model_lens.envs import EpisodeCollector

    console.print(f"[bold]Analyzing:[/bold] {checkpoint_path}")
    console.print(f"[bold]Backend:[/bold] {backend}")
    console.print(f"[bold]Episodes:[/bold] {n_episodes}")
    console.print(f"[bold]Mode:[/bold] {mode}")

    try:
        cfg = WorldModelConfig(d_action=4, d_obs=12288, backend=backend)
        wm = HookedWorldModel.from_checkpoint(checkpoint_path, backend=backend, cfg=cfg)

        caps = wm.capabilities if hasattr(wm, "capabilities") else None

        if mode == "auto" and caps is not None:
            if caps.is_rl_model():
                mode = "rl"
            else:
                mode = "non-rl"

        _print_analysis_profile(console, mode, caps)

        if report == "text":
            console.print("[yellow]Analysis report generation coming soon...[/yellow]")
        else:
            console.print("[yellow]HTML report generation coming soon...[/yellow]")

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


def _print_analysis_profile(console: Console, mode: str, caps) -> None:
    """Print a table showing which analyses will run based on mode and capabilities."""
    table = Table(title=f"Analysis Profile: {mode.upper()}")
    table.add_column("Analysis", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Requires", style="yellow")

    if mode == "rl" or (caps and caps.is_rl_model()):
        analyses = [
            ("Surprise Timeline", "Yes", "latents"),
            ("Latent Geometry", "Yes", "latents"),
            ("Temporal Memory", "Yes", "latents"),
            ("Disentanglement", "Yes", "latents + factors"),
            (
                "Reward Attribution",
                "Yes" if (caps and caps.has_reward_head) else "No",
                "reward head",
            ),
            ("Value Analysis", "Yes" if (caps and caps.has_critic) else "No", "critic head"),
            (
                "Imagination Faithfulness",
                "Yes" if (caps and caps.uses_actions) else "No",
                "actions",
            ),
            ("Safety Audit", "Yes" if (caps and caps.has_reward_head) else "No", "reward head"),
        ]
    else:
        analyses = [
            ("Surprise Timeline", "Yes", "latents"),
            ("Latent Geometry", "Yes", "latents"),
            ("Temporal Memory", "Yes", "latents"),
            ("Disentanglement", "Yes", "latents + factors"),
            ("SAE Analysis", "Yes", "latents"),
            ("Reward Attribution", "Skipped", "no reward head"),
            ("Value Analysis", "Skipped", "no critic head"),
            ("Imagination Faithfulness", "Skipped", "no actions"),
        ]

    for name, status, req in analyses:
        if status == "Yes":
            style = "green"
        elif status == "Skipped":
            style = "dim"
        else:
            style = "yellow"
        table.add_row(name, f"[{style}]{status}[/{style}]", req)

    console.print(table)


@app.command()
def compare(
    checkpoint_a: str,
    checkpoint_b: str,
    env: str = typer.Option("Pendulum-v1", help="Environment ID"),
    n_episodes: int = typer.Option(30, help="Number of episodes per model"),
    mode: str = typer.Option("auto", help="Comparison mode: auto, rl, non-rl"),
):
    """Compare two model checkpoints.

    Compares latent geometry, surprise, and (for RL models) reward prediction.
    """
    from world_model_lens import HookedWorldModel, WorldModelConfig

    console.print("[bold]Model Comparison[/bold]")
    console.print(f"Model A: {checkpoint_a}")
    console.print(f"Model B: {checkpoint_b}")
    console.print(f"Episodes: {n_episodes}")
    console.print(f"Mode: {mode}")
    console.print("[yellow]Comparison coming soon...[/yellow]")


@app.command()
def imagine(
    checkpoint_path: str,
    env: str = typer.Option("Pendulum-v1", help="Environment ID"),
    fork_at: int = typer.Option(10, help="Fork point"),
    horizon: int = typer.Option(30, help="Imagination horizon"),
    n_branches: int = typer.Option(5, help="Number of branches"),
    output_dir: str = typer.Option("./imagination", help="Output directory"),
):
    """Run imagination branching experiments."""
    console.print("[bold]Imagination Branching[/bold]")
    console.print(f"Fork at: t={fork_at}")
    console.print(f"Horizon: {horizon}")
    console.print(f"Branches: {n_branches}")
    console.print("[yellow]Imagination coming soon...[/yellow]")


@app.command()
def probe(
    checkpoint_path: str,
    env: str = typer.Option("Pendulum-v1", help="Environment ID"),
    concepts: str = typer.Option("all", help="Concepts to probe"),
    activations: str = typer.Option("z_posterior,h", help="Activations to probe"),
):
    """Train linear probes on activations."""
    console.print("[bold]Probe Training[/bold]")
    console.print(f"Checkpoint: {checkpoint_path}")
    console.print(f"Activations: {activations}")
    console.print("[yellow]Probe training coming soon...[/yellow]")


@app.command()
def benchmark(
    checkpoint_path: str = typer.Argument(..., help="Path to model checkpoint"),
    backend: str = typer.Option("dreamerv3", help="Backend architecture"),
    tasks: str = typer.Option("all", help="Benchmark tasks: all, imagination, latent, geometry"),
    output_dir: str = typer.Option("./benchmark_results", help="Output directory"),
    mode: str = typer.Option("auto", help="Benchmark mode: auto, rl, non-rl"),
):
    """Run benchmark suite on a world model.

    RL mode benchmarks:
    - Imagination reward prediction
    - Value estimation accuracy
    - Planning horizon analysis

    Non-RL mode benchmarks:
    - Latent trajectory coherence
    - Dynamics prediction accuracy
    - Manifold structure preservation
    """
    console.print(f"[bold]Benchmarking:[/bold] {checkpoint_path}")
    console.print(f"[bold]Backend:[/bold] {backend}")
    console.print(f"[bold]Mode:[/bold] {mode}")
    console.print("[yellow]Benchmarking coming soon...[/yellow]")


@app.command()
def capabilities(
    checkpoint_path: str = typer.Argument(..., help="Path to model checkpoint"),
    backend: str = typer.Option("dreamerv3", help="Backend architecture"),
):
    """Show the capabilities of a world model.

    Displays which optional features (decoder, reward head, critic, etc.)
    are available in the model.
    """
    from world_model_lens import HookedWorldModel, WorldModelConfig

    console.print(f"[bold]Inspecting:[/bold] {checkpoint_path}")
    console.print(f"[bold]Backend:[/bold] {backend}")

    try:
        cfg = WorldModelConfig(d_action=4, d_obs=12288, backend=backend)
        wm = HookedWorldModel.from_checkpoint(checkpoint_path, backend=backend, cfg=cfg)

        caps = wm.capabilities if hasattr(wm, "capabilities") else None

        if caps is None:
            console.print("[yellow]Capabilities not available[/yellow]")
            return

        table = Table(title="World Model Capabilities")
        table.add_column("Capability", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Has Decoder", str(caps.has_decoder))
        table.add_row("Has Reward Head", str(caps.has_reward_head))
        table.add_row("Has Continue Head", str(caps.has_continue_head))
        table.add_row("Has Actor", str(caps.has_actor))
        table.add_row("Has Critic", str(caps.has_critic))
        table.add_row("Uses Actions", str(caps.uses_actions))
        table.add_row("Is RL Trained", str(caps.is_rl_trained))

        console.print(table)

        model_type = "RL" if caps.is_rl_model() else "Non-RL"
        console.print(f"\n[bold]Model Type:[/bold] {model_type}")

        if caps.is_rl_model():
            console.print(
                "[green]This is an RL-trained world model with full analysis support.[/green]"
            )
        else:
            console.print(
                "[yellow]This is a non-RL world model. RL-specific analyses will be skipped.[/yellow]"
            )

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@app.command()
def ui(
    port: int = typer.Option(8080, help="Port for web UI"),
    debug: bool = typer.Option(False, help="Enable debug mode"),
):
    """Launch interactive web UI."""
    console.print(f"[bold]Starting World Model Lens UI on port {port}...[/bold]")
    console.print("[yellow]Web UI coming soon...[/yellow]")


@app.command()
def run(
    experiment_yaml: str,
):
    """Run experiment from YAML file."""
    console.print(f"[bold]Running experiment:[/bold] {experiment_yaml}")
    console.print("[yellow]YAML runner coming soon...[/yellow]")


if __name__ == "__main__":
    app()
