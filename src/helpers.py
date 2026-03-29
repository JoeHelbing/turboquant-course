"""Shared utilities and exercise feedback functions for the TurboQuant course.

These helpers provide programmatic feedback on exercises -- not just pass/fail,
but specific guidance on what the result means and where to look next.
"""

import math

import numpy as np
import torch
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

console = Console()


# ---------------------------------------------------------------------------
# Exercise checking
# ---------------------------------------------------------------------------

def check_intuition(
    result,
    expected,
    your_input=None,
    hint: str = "",
    tolerance: float = 0.1,
    label: str = "Result",
):
    """Check a numeric result against an expected value with rich feedback.

    Not just pass/fail -- explains what the result means.

    Args:
        result: The value the learner's code produced.
        expected: The target value (or callable predicate).
        your_input: What the learner provided (for feedback context).
        hint: Hint text shown on failure.
        tolerance: Relative tolerance for numeric comparison.
        label: Label for the result in output.
    """
    if callable(expected):
        passed = expected(result)
    elif isinstance(expected, (int, float)):
        if expected == 0:
            passed = abs(result) < tolerance
        else:
            passed = abs(result - expected) / abs(expected) < tolerance
    elif isinstance(expected, torch.Tensor):
        passed = torch.allclose(
            torch.as_tensor(result, dtype=torch.float32),
            expected.float(),
            rtol=tolerance,
            atol=tolerance * 0.1,
        )
    else:
        passed = result == expected

    if passed:
        console.print(Panel(
            f"[bold green]Correct![/bold green]\n{label}: {_fmt(result)}",
            title="[green]Exercise Check[/green]",
            border_style="green",
        ))
    else:
        msg = Text()
        msg.append("Not quite.\n", style="bold red")
        msg.append(f"Your {label}: {_fmt(result)}\n")
        msg.append(f"Expected: {_fmt(expected)}\n")
        if your_input is not None:
            msg.append(f"Your input: {_fmt(your_input)}\n")
        if hint:
            msg.append(f"\nHint: {hint}", style="italic yellow")
        console.print(Panel(msg, title="[red]Exercise Check[/red]", border_style="red"))

    return passed


def check_shape(tensor, expected_shape, name: str = "tensor"):
    """Check that a tensor has the expected shape."""
    actual = tuple(tensor.shape)
    if actual == tuple(expected_shape):
        console.print(f"[green]Shape of {name}: {actual} -- correct[/green]")
        return True
    else:
        console.print(
            f"[red]Shape of {name}: {actual}, expected {tuple(expected_shape)}[/red]"
        )
        return False


def check_close(a, b, name: str = "values", rtol: float = 1e-3, atol: float = 1e-5):
    """Check two tensors are close, with informative error."""
    a_t = torch.as_tensor(a, dtype=torch.float32)
    b_t = torch.as_tensor(b, dtype=torch.float32)
    if torch.allclose(a_t, b_t, rtol=rtol, atol=atol):
        max_diff = (a_t - b_t).abs().max().item()
        console.print(f"[green]{name}: match (max diff: {max_diff:.2e})[/green]")
        return True
    else:
        max_diff = (a_t - b_t).abs().max().item()
        mean_diff = (a_t - b_t).abs().mean().item()
        console.print(
            f"[red]{name}: mismatch (max diff: {max_diff:.2e}, "
            f"mean diff: {mean_diff:.2e})[/red]"
        )
        return False


# ---------------------------------------------------------------------------
# Experiment runners (used in exercises)
# ---------------------------------------------------------------------------

def run_quantize_experiment(quantize_fn, dequantize_fn, vectors, bits_label=""):
    """Run a quantization experiment and report distortion stats.

    Args:
        quantize_fn: Function that takes vectors -> codes (or any intermediate).
        dequantize_fn: Function that takes codes -> reconstructed vectors.
        vectors: Input tensor of shape (n, d).
        bits_label: Label for the bit-width being tested.

    Returns:
        Dict with mse, max_error, relative_error stats.
    """
    codes = quantize_fn(vectors)
    reconstructed = dequantize_fn(codes)
    residuals = vectors - reconstructed
    mse = (residuals ** 2).sum(dim=-1).mean().item()
    max_err = residuals.abs().max().item()
    rel_err = (residuals.norm(dim=-1) / vectors.norm(dim=-1).clamp(min=1e-10)).mean().item()

    console.print(Panel(
        f"MSE distortion: [bold]{mse:.6f}[/bold]\n"
        f"Max abs error:  {max_err:.6f}\n"
        f"Mean relative error: {rel_err:.4f} ({rel_err*100:.1f}%)",
        title=f"Quantization Results {bits_label}",
        border_style="cyan",
    ))

    return {"mse": mse, "max_error": max_err, "relative_error": rel_err}


def compare_distortions(results: dict[str, float], paper_values: dict[str, float]):
    """Compare empirical distortions against paper's Table 1.

    Args:
        results: Dict mapping bit-width labels to empirical MSE values.
        paper_values: Dict mapping same labels to paper's reported values.
    """
    console.print("\n[bold]Comparison with Paper (Table 1)[/bold]")
    console.print(f"{'Bits':<8} {'Yours':<12} {'Paper':<12} {'Ratio':<8}")
    console.print("-" * 40)
    for label in results:
        yours = results[label]
        paper = paper_values.get(label, float("nan"))
        ratio = yours / paper if paper > 0 else float("nan")
        color = "green" if ratio < 2.0 else "yellow" if ratio < 5.0 else "red"
        console.print(f"{label:<8} {yours:<12.6f} {paper:<12.6f} [{color}]{ratio:<8.2f}[/{color}]")


def visualize_1d_quantizer(boundaries, centroids, pdf_fn=None, title="Scalar Quantizer"):
    """Create a plotly figure showing a 1D quantizer: boundaries, centroids, and PDF.

    Args:
        boundaries: Array of decision boundaries.
        centroids: Array of reconstruction centroids.
        pdf_fn: Optional callable pdf(x) -> density. If None, uses standard normal.
        title: Plot title.

    Returns:
        plotly Figure object.
    """
    import plotly.graph_objects as go

    if pdf_fn is None:
        from scipy.stats import norm
        pdf_fn = norm.pdf

    x = np.linspace(-4, 4, 500)
    y = np.array([pdf_fn(xi) for xi in x])

    fig = go.Figure()

    # PDF curve
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name="PDF",
                             line=dict(color="steelblue", width=2)))

    # Decision boundaries
    for i, b in enumerate(boundaries):
        fig.add_vline(x=b, line_dash="dash", line_color="gray",
                      annotation_text=f"b{i}" if len(boundaries) <= 8 else None)

    # Centroids
    centroid_y = [pdf_fn(c) for c in centroids]
    fig.add_trace(go.Scatter(
        x=list(centroids), y=centroid_y, mode="markers+text",
        name="Centroids",
        marker=dict(color="red", size=10, symbol="diamond"),
        text=[f"c{i}" for i in range(len(centroids))],
        textposition="top center",
    ))

    fig.update_layout(title=title, xaxis_title="Value", yaxis_title="Density",
                      template="plotly_white", height=400)
    return fig


def norm_preservation_test(transform_fn, inverse_fn, vectors, name="Transform"):
    """Test that a transform preserves L2 norms.

    Args:
        transform_fn: Forward transform.
        inverse_fn: Inverse transform.
        vectors: Input tensor.
        name: Name for reporting.

    Returns:
        True if norms are preserved and roundtrip works.
    """
    original_norms = vectors.norm(dim=-1)
    transformed = transform_fn(vectors)
    transformed_norms = transformed.norm(dim=-1)
    reconstructed = inverse_fn(transformed)
    recon_norms = reconstructed.norm(dim=-1)

    norm_diff = (original_norms - transformed_norms).abs().max().item()
    roundtrip_err = (vectors - reconstructed).abs().max().item()

    ok = norm_diff < 1e-4 and roundtrip_err < 1e-4
    color = "green" if ok else "red"
    console.print(Panel(
        f"Max norm difference (forward): {norm_diff:.2e}\n"
        f"Max roundtrip error: {roundtrip_err:.2e}",
        title=f"[{color}]{name} Norm Preservation[/{color}]",
        border_style=color,
    ))
    return ok


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _fmt(val):
    """Format a value for display."""
    if isinstance(val, float):
        return f"{val:.6f}"
    if isinstance(val, torch.Tensor):
        if val.numel() <= 10:
            return str(val)
        return f"Tensor(shape={tuple(val.shape)}, mean={val.float().mean():.4f}, std={val.float().std():.4f})"
    if callable(val):
        return "<predicate>"
    return str(val)


def section_header(title: str, subtitle: str = ""):
    """Print a styled section header."""
    console.print(f"\n[bold blue]{'='*60}[/bold blue]")
    console.print(f"[bold blue]{title}[/bold blue]")
    if subtitle:
        console.print(f"[dim]{subtitle}[/dim]")
    console.print(f"[bold blue]{'='*60}[/bold blue]\n")
