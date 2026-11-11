"""Action-space checks; these are essential in embodied applications."""
import torch


def rotation6d_to_matrix_svd(rotation6d: torch.Tensor) -> torch.Tensor:
    """Project an arbitrary 6D rotation parameterization to SO(3) via SVD.

    First interpret the 6 entries as the first two rotation columns, create
    the third using cross product, then project onto the nearest proper
    orthogonal matrix. SVD's determinant correction avoids reflections.
    """
    if rotation6d.shape[-1] != 6:
        raise ValueError("Last dimension must be 6")
    col1, col2 = rotation6d[..., :3], rotation6d[..., 3:]
    candidates = torch.stack((col1, col2, torch.linalg.cross(col1, col2)), dim=-1)
    U, _, Vh = torch.linalg.svd(candidates)
    fix = torch.ones_like(U[..., 0, 0])
    fix = torch.where(torch.linalg.det(U @ Vh) < 0, -fix, fix)
    diagonal = torch.ones(*fix.shape, 3, device=rotation6d.device, dtype=rotation6d.dtype)
    diagonal[..., -1] = fix
    return U @ torch.diag_embed(diagonal) @ Vh


def validate_actions(actions: torch.Tensor, dimension: int) -> None:
    if actions.ndim != 3 or actions.shape[-1] != dimension:
        raise ValueError(f"Expected action tensor [batch, time, {dimension}]")
    if not torch.isfinite(actions).all():
        raise ValueError("Actions contain NaN or infinity")
