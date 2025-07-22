"""
Noise quilting algorithm, for stitching together the tiles of noise that we generate.

Based on "Image Quilting for Texture Synthesis and Transfer" (Efros and Freeman 2001)

"""

import sys
import heapq
import warnings
from itertools import count
from typing import Iterable

import numpy as np
from skimage.segmentation import flood

__all__ = [
    "optimally_choose_patches",
    "randomly_choose_patches",
    "naive_quilt",
    "quilt",
]

AdjacencyList = dict[tuple[int, int] | str, set[tuple[tuple[int, int] | str, float]]]
""" An adjacency list for a weighted graph; {node: (neighbour, weight)} """


class PatchError(Exception):
    """
    General exception for an issue with the patches

    """


class PatchSizeMismatchError(PatchError):
    """
    Patches are not all the same size, somehow
    """


class GraphConstructionError(PatchError):
    """
    General exception for an issue with constructing the graph
    """


class GraphTraversalError(PatchError):
    """
    General exception for an issue with traversing the graph
    """


class StitchingError(PatchError):
    """
    Error stitching the patches together
    """


def _ceildiv(a: int, b: int) -> int:
    """
    Return the ceiling of the division of a by b.

    :param a: the numerator
    :param b: the denominator
    :return: the ceiling of a / b
    """
    return -(a // -b)


def _patch_layout(
    target_size: tuple[int, ...], patch_size: tuple[int, ...], patch_overlap: int
) -> tuple[int, ...]:
    """
    Calculate how many patches are needed to fill the target size, given the patch size and overlap.

    If the patch size does not fit exactly into the target size, this will give the number of patches
    needed to fill the target size, with some extra which should be dealt with by the caller.

    :param target_size: the size of the desired quilt of patches
    :param patch_size: the size of each patch
    :param patch_overlap: how much patches should overlap when building up the quilt (in pixels)

    :raises ValueError: if the patch size is larger than the target size, if the overlap is negative
                        or if the target and patch have different dimensions
    """
    if patch_overlap < 0:
        raise ValueError(f"Patch overlap must be non-negative, got {patch_overlap}")
    if patch_overlap >= min(patch_size):
        raise ValueError(
            f"Patch overlap must be smaller than the patch size, got {patch_overlap} and {patch_size}"
        )
    if len(target_size) != len(patch_size):
        raise ValueError(
            f"Target size {target_size} and patch size {patch_size} must have the same number of dimensions"
        )

    return tuple(
        _ceildiv(t - patch_overlap, p - patch_overlap)
        for t, p in zip(target_size, patch_size)
    )


def _verify_patches(patches: Iterable[np.ndarray]) -> None:
    """
    Perform some basic checks on the patches to ensure they are suitable for quilting.

    Checks that they are all the same size and 2d.

    """
    if not all((patch.shape == patches[0].shape for patch in patches)):
        raise PatchSizeMismatchError("All patches must be the same size")
    if not all((len(patch.shape) == 2 for patch in patches)):
        raise PatchSizeMismatchError("All patches must be 2d")


def randomly_choose_patches(
    patches: Iterable[np.ndarray],
    target_size: tuple[int, int],
    patch_overlap: int,
    allow_rotation: bool = False,
    *,
    rng: np.random.Generator,
) -> list[list[np.ndarray]]:
    """
    Randomly choose patches that will at least fill the target size when stitched together.

    Chooses patches with replacement.

    :param patches: the patches to choose from
    :param target_size: the size of the desired quilt of patches
    :param patch_overlap: how much patches should overlap when building up the quilt (in pixels)
    :param allow_rotation: whether to allow patches to be rotated when matching them
    :param rng: a random number generator

    :raises PatchSizeMismatchError: if the patches are not all the same size or they are not all 2d
    :return: patches that will at least fill the target size when stitched together.
             Note that after stitching, the resulting array may be larger than the target size.

    """
    if allow_rotation:
        raise NotImplementedError(
            "Randomly choosing patches with rotation is not yet implemented"
        )

    _verify_patches(patches)
    n_col, n_row = _patch_layout(target_size, patches[0].shape, patch_overlap)

    out_list = [[None for _ in range(n_col)] for _ in range(n_row)]

    for i in range(n_row):
        for j in range(n_col):
            out_list[i][j] = rng.choice(patches)

    return out_list


def _get_best_patch(
    patches: Iterable[np.ndarray],
    comparison_patch: np.ndarray,
    patch_slice: tuple[slice, slice],
    n_uses: list[int],
    repeat_penalty: float,
) -> np.ndarray:
    """
    Iterate over patches, and find the one that best matches the comparison patch
    in the specified slice.

    Comparison patch should be already sliced
    """
    score = float("inf")
    best_patch = None
    best_idx = None
    for i, patch in enumerate(patches):
        overlap_region = patch[patch_slice]
        mse = np.sum((overlap_region - comparison_patch) ** 2) * (
            1.0 + repeat_penalty * n_uses[i]
        )
        if mse < score:
            score = mse
            best_patch = patch
            best_idx = i

    n_uses[best_idx] += 1

    return best_patch


def _best_patch_compare_left(
    patches: Iterable[np.ndarray],
    comparison_patch: np.ndarray,
    patch_overlap: int,
    n_uses: list,
    repeat_penalty: float,
) -> np.ndarray:
    """
    Iterate over patches, and find the one that best matches the comparison patch
    along its right edge
    """
    # We just want to take the right bit of the comparison patch
    comparison_patch = comparison_patch[:, -patch_overlap:]

    return _get_best_patch(
        patches,
        comparison_patch,
        (slice(None), slice(None, patch_overlap)),
        n_uses,
        repeat_penalty,
    )


def _best_patch_compare_top(
    patches: Iterable[np.ndarray],
    comparison_patch: np.ndarray,
    patch_overlap: int,
    n_uses: list[int],
    repeat_penalty: float,
) -> np.ndarray:
    """ """
    # We just want to take the bottom bit of the comparison patch
    comparison_patch = comparison_patch[-patch_overlap:, :]

    return _get_best_patch(
        patches,
        comparison_patch,
        (slice(None, patch_overlap), slice(None)),
        n_uses,
        repeat_penalty,
    )


def _best_patch_compare_top_left(
    patches: Iterable[np.ndarray],
    left_comparison_patch: np.ndarray,
    top_comparison_patch: np.ndarray,
    patch_overlap: int,
    n_uses: list[int],
    repeat_penalty: float,
) -> np.ndarray:
    """
    Find the best patch that matches both the left and top edges of the comparison patches.

    This double-counts the overlap region, since it compares it to both the left and top edges,
    but I think this is fine
    """
    # We want the right of the patch on the left, and the bottom of the patch on top
    left_comparison_patch = left_comparison_patch[:, -patch_overlap:]
    top_comparison_patch = top_comparison_patch[-patch_overlap:, :]

    score = float("inf")
    best_patch = None
    best_idx = None
    for i, patch in enumerate(patches):
        left_overlap_region = patch[:, :patch_overlap]
        top_overlap_region = patch[:patch_overlap, :]
        mse = np.sum((left_overlap_region - left_comparison_patch) ** 2) + np.sum(
            (top_overlap_region - top_comparison_patch) ** 2
        )
        mse *= 1.0 + repeat_penalty * n_uses[i]
        if mse < score:
            score = mse
            best_patch = patch
            best_idx = i

    n_uses[best_idx] += 1

    return best_patch


def optimally_choose_patches(
    patches: Iterable[np.ndarray],
    target_size: tuple[int, int],
    patch_overlap: int,
    allow_rotation: bool = False,
    *,
    rng: np.random.Generator,
    repeat_penalty: float = 0.0,
) -> list[list[np.ndarray]]:
    """
    Choose patches that will at least fill the target size when stitched together, such that the overlap
    between patches is optimal.

    Chooses the first patch randomly, then builds up the first row by matching the left edge of each patch
    to the right edge of the previous patch.
    The first patch in subsequent rows are chosen according to its match with the bottom edge of the first patch,
    then the rest of the patches in the row are chosen by matching the top and left edges of the patches.

    Patches are chosen with replacement, but repeatedly using the same patch can be penalised by setting
    `repeat_penalty` to a positive value.

    :param patches: the patches to choose from
    :param target_size: the size of the desired quilt of patches
    :param patch_overlap: how much patches should overlap when building up the quilt (in pixels)
    :param allow_rotation: whether to allow patches to be rotated when matching them
    :param rng: a random number generator, used to choose the first patch
    :param repeat_penalty: a penalty for using the same patch multiple times, to encourage diversity in the patches used

    :return: patches that will at least fill the target size when stitched together.
             Note that after stitching, the resulting array may be larger than the target size.

    """
    if allow_rotation:
        raise NotImplementedError(
            "Randomly choosing patches with rotation is not yet implemented"
        )

    if patch_overlap == 0:
        raise PatchError(
            "Patch overlap must be non-zero; if you want no overlap, use `randomly_choose_patches`"
        )

    if repeat_penalty < -1:
        warnings.warn(
            f"Repeat penalty {repeat_penalty} <-1 will force patch reuse",
            RuntimeWarning,
        )
    elif repeat_penalty == -1:
        raise ValueError(
            "Repeat penalty should be positive; if you want no penalty, use 0"
        )
    elif repeat_penalty < 0:
        warnings.warn(
            f"Repeat penalty {repeat_penalty} <0; this will encourage patch reuse",
            RuntimeWarning,
        )

    _verify_patches(patches)
    n_col, n_row = _patch_layout(target_size, patches[0].shape, patch_overlap)

    # We want to track how many times each patch has been used, to apply the repeat penalty
    n_uses = [0 for _ in patches]

    out_list = [[None for _ in range(n_col)] for _ in range(n_row)]

    # Choose the first patch randomly
    first_patch_index = rng.choice(len(patches))
    n_uses[first_patch_index] += 1
    out_list[0][0] = patches[first_patch_index]

    # Choose the first row by matching the left edge of each patch to the right edge of the previous patch
    for i in range(1, n_col):
        # We want to compare the overlap region on the right of the previous patch
        # with the overlap region on the left of the current patch
        comparison_patch = out_list[0][i - 1]

        # Iterate over all the patches and find the best one
        out_list[0][i] = _best_patch_compare_left(
            patches, comparison_patch, patch_overlap, n_uses, repeat_penalty
        )

    # For the next rows, choose the first patch according to its match with the bottom edge of the first patch
    for i in range(1, n_row):
        # Compare the first one to the bottom edge of the first patch
        out_list[i][0] = _best_patch_compare_top(
            patches, out_list[i - 1][0], patch_overlap, n_uses, repeat_penalty
        )

        # Compare the rest of them to the top and left edges of the previous patches
        for j in range(1, n_col):
            out_list[i][j] = _best_patch_compare_top_left(
                patches,
                out_list[i][j - 1],
                out_list[i - 1][j],
                patch_overlap,
                n_uses,
                repeat_penalty,
            )

    return out_list


def _unfilled_image(shape: tuple[int, int]) -> np.ndarray:
    """
    Create an image of the given shape filled with `np.inf`, to indicate that it is unfilled.

    :param shape: the shape of the image to create
    :return: an image of the given shape filled with `np.inf`
    """
    return np.full(shape, np.inf, dtype=np.float32)


def _image_region(
    existing_image: np.ndarray, patch_shape: tuple[int, int], position: tuple[int, int]
) -> np.ndarray:
    """
    Get the relevant region of the existing image if we're overlapping a patch at the given position.
    """
    return existing_image[
        position[0] : position[0] + patch_shape[0],
        position[1] : position[1] + patch_shape[1],
    ]


def overlap_cost(
    existing_image: np.ndarray,
    candidate_patch: np.ndarray,
    position: tuple[int, int],
) -> np.ndarray:
    """
    Find the cost matrix of overlapping a candidate patch onto an existing image at a given position.

    Returns an array the same shape as the candidate patch, where each pixel is the cost of overlapping
    that pixel onto the existing image at the given position.

    Unfilled pixels in the existing image should be labelled with `np.inf`.

    The values are located from the image's perspective; for example, adding a patch
    on to the right of the existing image will give non-inf values on the right of the cost matrix.

    :param existing_image: the existing image to overlap onto
    :param candidate_patch: the patch to overlap
    :param position: the position (y, x) of the top-left corner of the candidate patch in the existing image

    :return: a cost matrix of the same shape as the candidate patch, where each pixel is the cost of overlapping
             that pixel onto the existing image at the given position.
             The cost is defined as the squared difference between the existing image and the candidate patch.
    """
    costs = (
        _image_region(existing_image, candidate_patch.shape, position) - candidate_patch
    ) ** 2

    return costs


def _terminal_nodes(edge: str, height: int, width: int) -> set[tuple[int, int]]:
    if edge == "left":
        return {(y, 0) for y in range(height)}
    if edge == "bottom":
        return {(height - 1, x) for x in range(width)}
    if edge == "right":
        return {(y, width - 1) for y in range(height)}
    if edge == "top":
        return {(0, x) for x in range(width)}
    raise GraphConstructionError(f"Invalid edge: {edge}")


def cost_to_graph(cost_matrix: np.ndarray, start: str, end: str) -> AdjacencyList:
    """
    Convert a cost matrix to a graph (represented as an adjacency matrix).

    :param cost_matrix: a cost matrix, where each pixel is the cost of overlapping that pixel onto the existing image
                      at the given position.
    :param start: which edge to start traversal from
    :param end: which edge to end traversal at

    :return: a graph represented as an adjacency list, where each node is a pixel in the cost matrix
    """
    assert start in {"left", "right", "top", "bottom"}, f"Invalid edge: {start=}"
    assert end in {"left", "right", "top", "bottom"}, f"Invalid edge: {end=}"

    height, width = cost_matrix.shape

    graph: AdjacencyList = {}

    for y in range(height):
        for x in range(width):
            if cost_matrix[y, x] == np.inf:
                continue

            neighbours = set()
            if x + 1 < width and cost_matrix[y, x + 1] != np.inf:
                neighbours.add(((y, x + 1), cost_matrix[y, x + 1]))
            if y + 1 < height and cost_matrix[y + 1, x] != np.inf:
                neighbours.add(((y + 1, x), cost_matrix[y + 1, x]))
            if x - 1 >= 0 and cost_matrix[y, x - 1] != np.inf:
                neighbours.add(((y, x - 1), cost_matrix[y, x - 1]))
            if y - 1 >= 0 and cost_matrix[y - 1, x] != np.inf:
                neighbours.add(((y - 1, x), cost_matrix[y - 1, x]))

            if neighbours:
                graph[(y, x)] = neighbours

    # Add the start and end nodes
    start_nodes = _terminal_nodes(start, height, width)
    end_nodes = _terminal_nodes(end, height, width)

    graph["START"] = set()
    for node in start_nodes:
        # We only want to add reachable (i.e. non-inf) start nodes
        if cost_matrix[node] != np.inf:
            graph["START"].add((node, cost_matrix[node]))
    if not graph["START"]:
        for start_node in start_nodes:
            print(
                f"Node {start_node} not in graph: {cost_matrix[start_node]}",
                file=sys.stderr,
            )
        raise GraphConstructionError(f"No valid start nodes found for edge {start}")

    for node in end_nodes:
        # We only want to add the end node if it is reachable
        if cost_matrix[node] != np.inf and node in graph:
            graph[node].add(("END", 0))
    if not any(
        ("END", 0) in graph.get(node, set()) for node in end_nodes if node in graph
    ):
        for end_node in end_nodes:
            print(
                f"Node {end_node} not in graph: {cost_matrix[end_node]}",
                file=sys.stderr,
            )
        raise GraphConstructionError(f"No valid end nodes found for edge {end}")

    return graph


def shortest_path(graph: AdjacencyList) -> list[tuple[int, int]]:
    """
    Find the shortest path through the graph from the start node to the end node.

    :param graph: the graph to traverse, represented as an adjacency list
    :returns: a list of nodes in the shortest path, starting from "START" and ending at "END"
    """
    dist = {}
    prev = {}
    visited = set()
    queue = []
    counter = count()  # Unique sequence count

    heapq.heappush(queue, (0, next(counter), "START"))
    dist["START"] = 0

    while queue:
        current_cost, _, current_node = heapq.heappop(queue)
        if current_node in visited:
            continue
        visited.add(current_node)

        if current_node == "END":
            break

        for neighbor, cost in graph.get(current_node, []):
            if neighbor in visited:
                continue
            new_cost = current_cost + cost
            if neighbor not in dist or new_cost < dist[neighbor]:
                dist[neighbor] = new_cost
                prev[neighbor] = current_node
                heapq.heappush(queue, (new_cost, next(counter), neighbor))

    # Reconstruct path from END to START
    path = []
    node = prev.get("END")
    while node and node != "START":
        path.append(node)
        node = prev.get(node)
    path.reverse()
    return path


def seam_nodes(cost_matrix: np.ndarray, start: str, end: str) -> list[tuple[int, int]]:
    """
    Find the nodes that form the optimal seam through the cost matrix.
    """
    graph = cost_to_graph(cost_matrix, start, end)
    return shortest_path(graph)


def _seam_edge(costs: np.ndarray) -> bool:
    """
    Identify whether this edge (1d array) is a seam edge.
    """
    if np.all(costs == np.inf):
        return False
    if np.all(np.isfinite(costs)):
        return False
    return True


def _identify_seam_edges(cost_matrix: np.ndarray) -> set[str, str]:
    """
    Find the edges of the cost matrix that are suitable for finding a seam.

    These should be the edges that are not fully filled with `np.inf` or numerical values.
    """
    edges = set()
    if _seam_edge(cost_matrix[0]):
        edges.add("top")
    if _seam_edge(cost_matrix[-1]):
        edges.add("bottom")
    if _seam_edge(cost_matrix[:, 0]):
        edges.add("left")
    if _seam_edge(cost_matrix[:, -1]):
        edges.add("right")

    return edges


def _reseed(
    seam_edges: set[str], mask: np.ndarray
) -> tuple[tuple[int, int] | None, tuple[int, int] | None]:
    """
    Find a new seed to start from, based on the current mask and seam edges.

    Returns None if no new seed is appropriate or could be found

    Returns (existing_seed, candidate_seed)
    """
    unfilled = np.argwhere(mask == -1)

    if seam_edges == {"bottom", "right"}:
        # We shouldn't have to re-seed the candidate patch since
        # all pixels should be fillable from the bottom right corner.
        # So we'll just look for a new seed for the existing patch
        # on the left or top edges
        for y, x in unfilled:
            if y == 0 or x == 0:
                return (y, x), None

    if seam_edges == {"top", "bottom"}:
        # Look for new seeds for the existing patch on the left
        # and for the candidate patch on the right
        for y, x in unfilled:
            if x == 0:
                existing_seed = (y, x)
                break
        else:
            existing_seed = None
        for y, x in unfilled:
            if x == mask.shape[1] - 1:
                candidate_seed = (y, x)
                break
        else:
            candidate_seed = None
        return existing_seed, candidate_seed

    if seam_edges == {"left", "right"}:
        # Look for new seeds for the existing patch on the top
        # and for the candidate patch on the bottom
        for y, x in unfilled:
            if y == 0:
                existing_seed = (y, x)
                break
        else:
            existing_seed = None
        for y, x in unfilled:
            if y == mask.shape[0] - 1:
                candidate_seed = (y, x)
                break
        else:
            candidate_seed = None
        return existing_seed, candidate_seed

    raise NotImplementedError(
        "Reseeding for this seam edge not implemented - we shouldn't be able to get here"
    )


def _merge_mask(
    patch_shape: tuple[int, int], seam: list[tuple[int, int]], seam_edges: set[str]
) -> np.ndarray:
    """
    A mask indicating which pixel to fill with values from which array.

    0 - the seam
    1 - the existing patch
    2 - the candidate patch
    """
    if len(seam_edges) != 2:
        raise ValueError(f"Expected exactly two seam edges, got {seam_edges}")

    if seam_edges not in ({"top", "bottom"}, {"left", "right"}, {"bottom", "right"}):
        warnings.warn(
            f"Did not expect to find seam from {' to '.join(seam_edges)}",
            RuntimeWarning,
        )

    mask = np.full(patch_shape, -1, dtype=int)

    # Fill the seam with 0s
    seam = np.array(seam)
    y_seam, x_seam = seam[:, 0], seam[:, 1]
    mask[y_seam, x_seam] = 0

    # Start by trying to seed the mask from the top-left and bottom-right
    h, w = patch_shape
    existing_seed = (0, 0)
    candidate_seed = (h - 1, w - 1)

    footprint = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)

    while np.any(mask == -1):
        n_to_fill = np.sum(mask == -1)

        # Propagate out from our seeds and fill in the mask
        if existing_seed is not None:
            reach = flood(mask != 0, existing_seed, footprint=footprint)
            mask[reach & (mask == -1)] = 1

        if candidate_seed is not None:
            reach = flood(mask != 0, candidate_seed, footprint=footprint)
            mask[reach & (mask == -1)] = 2

        # If we didn't fill any, something has gone wrong
        if n_to_fill == np.sum(mask == -1):
            raise StitchingError("Failed to fill mask")

        # Otherwise, try to find a new seed
        if np.any(mask == -1):
            existing_seed, candidate_seed = _reseed(seam_edges, mask)

    return mask


def _merge_patches(
    existing_patch: np.ndarray,
    candidate_patch: np.ndarray,
    seam: list[tuple[int, int]],
    seam_edges: set[str],
) -> np.ndarray:
    """
    Merge the existing patch and candidate patch along a seam.

    :param existing_patch: the existing patch to merge
    :param candidate_patch: the candidate patch to merge
    :param seam: the seam, as a list of (y, x) co-ords, to use for merging
    :param seam_edges: the edges of the cost matrix that are suitable for finding a seam

    :return: the merged patch
    """
    if existing_patch.shape != candidate_patch.shape:
        raise PatchError(
            f"Cannot merge patches of different shapes: {existing_patch.shape} and {candidate_patch.shape}"
        )

    # Find where we should take pixels from each patch (and where we should average)
    mask = _merge_mask(existing_patch.shape, seam, seam_edges)

    # Fill in the pixels of the merged patch
    merged_patch = np.full_like(existing_patch, np.nan, dtype=np.float32)
    merged_patch[mask == 1] = existing_patch[mask == 1]
    merged_patch[mask == 2] = candidate_patch[mask == 2]

    # The seam takes the average of the two patches
    merged_patch[mask == 0] = (
        existing_patch[mask == 0] + candidate_patch[mask == 0]
    ) / 2.0

    return merged_patch


def add_patch(
    existing_image: np.ndarray, candidate_patch: np.ndarray, position: tuple[int, int]
) -> np.ndarray:
    """
    Add a candidate patch to an existing image at a given position, stitching them using the optimal seam.

    Unfilled pixels in the existing image should be labelled with `np.inf`.
    An average of the pixel values in the two images along the seam will be used.

    :param existing_image: the existing image to add the patch to
    :param candidate_patch: the patch to add
    :param position: the position (y, x) of the top-left corner of the candidate patch in the existing image

    :return: the existing image with the candidate patch added, with the overlap stitched
    """
    # Pick out the relevant region of the existing image
    existing_patch = _image_region(existing_image, candidate_patch.shape, position)

    # Find the overlap of the patch and image
    cost = overlap_cost(existing_image, candidate_patch, position)
    if np.all(cost == np.inf):
        raise PatchError(
            "No overlap between existing image and candidate patch; cannot stitch"
        )

    # Find the type of overlap - should either be left to right, top to bottom or bottom to right
    seam_edges: tuple[str, str] = _identify_seam_edges(cost)
    if seam_edges not in ({"top", "bottom"}, {"left", "right"}, {"bottom", "right"}):
        warnings.warn(
            f"Did not expect to find seam from {' to '.join(seam_edges)}",
            RuntimeWarning,
        )

    # Find the minimal cost seam through the cost matrix
    seam = seam_nodes(cost, *seam_edges)

    # Use this seam to stitch the patch onto the existing image
    merged_patch = _merge_patches(existing_patch, candidate_patch, seam, seam_edges)

    # Place the merged patch back into the existing image
    result = existing_image.copy()
    result[
        position[0] : position[0] + merged_patch.shape[0],
        position[1] : position[1] + merged_patch.shape[1],
    ] = merged_patch

    return result


def naive_quilt(
    patches: list[list[np.ndarray]], patch_overlap: int, target_size: tuple[int, int]
) -> np.ndarray:
    """
    Join the patches together using a naive averaging in the overlap.

    :param patches: the patches to join together
    :param patch_overlap: how much patches should overlap when building up the quilt (in pixels)

    :raises PatchError: if patches is a ragged array or if the patches are not all the same size
    :raises PatchError: if the number of patches passed in does not work with the given target size and patch overlap
    :return: the joined patches, with the overlap averaged
    """
    if any(len(row) != len(patches[0]) for row in patches):
        raise PatchError("Patches must be a rectangular array")
    if any(any(patch.shape != patches[0][0].shape for patch in row) for row in patches):
        raise PatchError("Patches must all be the same size")
    if (len(patches[0]), len(patches)) != _patch_layout(
        target_size, patches[0][0].shape, patch_overlap
    ):
        raise PatchError(
            f"Number of patches {len(patches), len(patches[0])} incompatible with {target_size=} and {patch_overlap=}"
        )

    n_rows, n_cols = len(patches), len(patches[0])

    result = np.zeros(target_size)
    # Track how many times each pixel has been used, so we can average the overlaps
    counts = np.zeros(target_size)

    patch_h, patch_w = patches[0][0].shape
    for i in range(n_rows):
        for j in range(n_cols):
            start_y = i * (patch_h - patch_overlap)
            end_y = min(start_y + patch_h, target_size[0])

            start_x = j * (patch_w - patch_overlap)
            end_x = min(start_x + patch_w, target_size[1])

            # We might have to crop the patch if it doesn't fit exactly
            y_slice = (
                slice(0, end_y - start_y) if end_y - start_y < patch_h else slice(None)
            )
            x_slice = (
                slice(0, end_x - start_x) if end_x - start_x < patch_w else slice(None)
            )

            result[start_y:end_y, start_x:end_x] += patches[i][j][y_slice, x_slice]
            counts[start_y:end_y, start_x:end_x] += np.ones_like(
                patches[i][j][y_slice, x_slice]
            )

    return result / counts


def quilt(
    patches: Iterable[np.ndarray],
    *,
    target_size: tuple[int, int],
    patch_overlap: int,
    rng: np.random.Generator,
    repeat_penalty: float = 0.0,
) -> np.ndarray:
    """
    Quilt together a collection of patches to give an array of the provided size.

    Chooses patches from `patches` randomly (with replacement), stitching them together until
    the desired `target_size` is reached.
    Uses the quilting algorithm in [1] to choose which patches should be aligned next to each other
    and the optimal cut through each patch.

    :param patches: the patches to stitch together
    :param target_size: the size of the desired quilt of patches
    :param patch_overlap: how much patches should overlap when building up the quilt (in pixels)

    :raises UserWarning: if the provided target_size, patch size and overlap
    :raises ValueError: if the patches are not all the same size or if they are not 2d
    :return: the optimal quilt of patches

    """
    # Check that the patches are all the same size
    patch_sizes = {patch.shape for patch in patches}
    try:
        (patch_size,) = patch_sizes
    except ValueError as e:
        raise PatchError(f"All patches must be the same size; got {patch_sizes}") from e

    # Check the patches are all 2d
    if len(patch_size) != 2:
        raise PatchError(f"Patches must be 2d, not {patch_size}")

    # Find how many patches we need to build up to the target size
    n_col, n_row = _patch_layout(target_size, patch_size, patch_overlap)
    patch_grid = optimally_choose_patches(
        patches,
        target_size,
        patch_overlap,
        rng=rng,
        repeat_penalty=repeat_penalty,
    )

    # TODO warn if the patches are larger than the target - we'll need to crop

    # Init an array of the right size ()
    array_size = (
        n_row * (patch_size[0] - patch_overlap) + patch_overlap,
        n_col * (patch_size[1] - patch_overlap) + patch_overlap,
    )
    result = _unfilled_image(array_size)

    result[0 : patch_size[0], 0 : patch_size[1]] = patch_grid[0][0]

    # Now stitch the patches together
    for i in range(n_row):
        for j in range(n_col):
            if i == 0 and j == 0:
                continue
            # Get the position of the patch in the result array
            pos_y = i * (patch_size[0] - patch_overlap)
            pos_x = j * (patch_size[1] - patch_overlap)

            result = add_patch(result, patch_grid[i][j], (pos_y, pos_x))

    # Check that the quilt is the right size

    return result
