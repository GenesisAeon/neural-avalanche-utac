"""CLI entry point for neural-avalanche-utac.

Commands:
  neural-utac run               — run a UTAC simulation cycle
  neural-utac criticality-check — check branching ratio from spike data
  neural-utac gamma-universality — compute and compare Γ_brain vs Γ_AMOC
  neural-utac benchmark         — run full validation suite
  neural-utac version           — show version
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    name="neural-utac",
    help="Brain Criticality & Neuronal Avalanche Threshold — GenesisAeon Package 20",
    add_completion=False,
)
console = Console()


@app.command()
def run(
    duration: float = typer.Option(3600.0, help="Simulation duration in seconds"),
    neurons: int = typer.Option(500, help="Number of neurons"),
    seed: int = typer.Option(42, help="Random seed"),
    segment: float = typer.Option(60.0, help="Segment length in seconds"),
) -> None:
    """Run a NeuralAvalancheUTAC simulation cycle."""
    from neural_avalanche_utac.system import NeuralAvalancheUTAC

    console.print(f"[bold cyan]NeuralAvalancheUTAC[/bold cyan] — running {duration}s cycle "
                  f"({neurons} neurons, seed={seed})")

    model = NeuralAvalancheUTAC(n_neurons=neurons, seed=seed, segment_s=segment)
    result = model.run_cycle(duration_seconds=duration)

    table = Table(title="UTAC Cycle Result", show_header=True)
    table.add_column("Metric", style="bold")
    table.add_column("Value")

    table.add_row("H_final (σ_b)", f"{result['H_final']:.4f}")
    table.add_row("σ_b mean", f"{result['sigma_b_mean']:.4f}")
    table.add_row("Γ mean", f"{result['gamma_mean']:.4f}")
    table.add_row("Γ theoretical (critical)", "0.251")
    table.add_row("Avalanches detected", str(result["n_avalanches_total"]))
    table.add_row("Phase events", str(result["n_phase_events"]))
    table.add_row("Is critical?", "✓" if result["is_critical"] else "✗")

    console.print(table)

    univ = model.gamma_universality_check()
    if univ["universality_confirmed"]:
        console.print("[green]✓ Γ_brain ≈ Γ_AMOC ≈ 0.251 — cross-domain universality confirmed[/green]")
    else:
        console.print(f"[yellow]Γ_brain = {univ['gamma_brain_measured']:.3f}  "
                      f"Γ_AMOC = {univ['gamma_amoc']:.3f}  (more data needed)[/yellow]")


@app.command(name="criticality-check")
def criticality_check(
    data: Path = typer.Argument(..., help="Path to .npz spike train file"),
    key: str = typer.Option("spikes_critical", help="Array key inside the .npz"),
) -> None:
    """Check branching ratio and criticality from a spike train .npz file."""
    import numpy as np

    from neural_avalanche_utac.branching import BranchingRatioEstimator
    from neural_avalanche_utac.crep_neural import NeuralCREPTensor

    if not data.exists():
        console.print(f"[red]File not found: {data}[/red]")
        raise typer.Exit(1)

    npz = np.load(str(data))
    if key not in npz:
        console.print(f"[red]Key '{key}' not found. Available: {list(npz.keys())}[/red]")
        raise typer.Exit(1)

    spikes = npz[key]
    est = BranchingRatioEstimator()
    sigma_b = est.estimate(spikes)
    crep = NeuralCREPTensor()
    crep_out = crep.compute(spikes)

    console.print(f"[bold]σ_b (branching ratio):[/bold] {sigma_b:.4f}  "
                  f"{'✓ CRITICAL' if abs(sigma_b - 1.0) < 0.05 else '✗ not critical'}")
    console.print(f"[bold]Γ_brain:[/bold] {crep_out['Gamma']:.4f}  "
                  f"(target: 0.251, Δ = {abs(crep_out['Gamma'] - 0.251):.4f})")
    console.print(f"  C={crep_out['C']:.3f}  R={crep_out['R']:.3f}  "
                  f"E={crep_out['E']:.3f}  P={crep_out['P']:.3f}")


@app.command(name="gamma-universality")
def gamma_universality(
    neurons: int = typer.Option(300, help="Neurons for simulation"),
    duration: float = typer.Option(300.0, help="Duration in seconds"),
    seed: int = typer.Option(42, help="Random seed"),
) -> None:
    """Compute Γ_brain and compare with Γ_AMOC (cross-domain universality check)."""
    from neural_avalanche_utac.constants import CREP_SPECTRUM
    from neural_avalanche_utac.system import NeuralAvalancheUTAC

    model = NeuralAvalancheUTAC(n_neurons=neurons, seed=seed)
    model.run_cycle(duration_seconds=duration)
    univ = model.gamma_universality_check()

    console.print(f"\n[bold]Γ_brain (measured):[/bold]     {univ['gamma_brain_measured']:.4f}")
    console.print(f"[bold]Γ_brain (theoretical):[/bold]  {univ['gamma_brain_theoretical']:.4f}")
    console.print(f"[bold]Γ_AMOC (Package 18):[/bold]    {univ['gamma_amoc']:.4f}")
    console.print(f"[bold]Universality confirmed:[/bold]  {univ['universality_confirmed']}")
    console.print(f"\n[italic]{univ['interpretation']}[/italic]\n")

    table = Table(title="CREP Criticality Spectrum (GenesisAeon Atlas)", show_header=True)
    table.add_column("Domain")
    table.add_column("Package")
    table.add_column("Γ", justify="right")
    rows = [
        ("Solar Flare", "P21", 0.014),
        ("Cygnus X-1 Jet", "P17", 0.046),
        ("Amazon Rainforest", "P19", 0.116),
        ("AMOC Ocean", "P18", 0.251),
        ("Neural Criticality", "P20 ← THIS", univ["gamma_brain_measured"]),
        ("BTW Sandpile", "P22", 0.296),
        ("Manna Sandpile", "P22", 0.376),
        ("ERA5 Arctic", "P01", 0.920),
    ]
    for domain, pkg, gamma in rows:
        highlight = "bold cyan" if "THIS" in pkg else ""
        table.add_row(f"[{highlight}]{domain}[/{highlight}]" if highlight else domain,
                      pkg, f"{gamma:.3f}")
    console.print(table)


@app.command()
def benchmark(
    seed: int = typer.Option(42, help="Random seed"),
    neurons: int = typer.Option(300, help="Neurons for benchmark"),
    duration: float = typer.Option(300.0, help="Duration in seconds"),
) -> None:
    """Run the full validation benchmark suite."""
    from neural_avalanche_utac.benchmark import run_benchmarks

    results = run_benchmarks(seed=seed, verbose=True, n_neurons=neurons, duration_s=duration)
    summary = results["_summary"]
    if summary["passed"] == summary["total"]:
        console.print("[green]All benchmarks passed.[/green]")
    else:
        console.print(f"[red]{summary['total'] - summary['passed']} benchmark(s) failed.[/red]")
        raise typer.Exit(1)


@app.command()
def version() -> None:
    """Show version information."""
    from neural_avalanche_utac import __version__

    console.print(f"neural-avalanche-utac {__version__} — GenesisAeon Package 20")
