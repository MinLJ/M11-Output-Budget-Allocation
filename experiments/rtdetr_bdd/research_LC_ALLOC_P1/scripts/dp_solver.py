"""Exact multiple-choice allocation for LC-ALLOC-P1.

The production problem is one 40-image group.  Every image receives exactly
one prefix length ``K`` in ``5..50`` and, for every requested mean budget, the
chosen lengths must sum to ``40 * mean_budget``.  Marginal values may be
arbitrary finite float64 values: in particular, they need not be decreasing,
non-negative, or concave.

Tie rule (declared here rather than inferred from any outcome data): among
allocations with exactly equal float64 DP objectives, choose the
lexicographically smallest complete K vector in the caller's fixed row order.
Thus lower row indices have tie priority and prefer lower K.  Objective
comparisons use strict float64 ``>`` and exact float64 ``==``; there is no
data-dependent tolerance and the solver has no GT input.

All requested capacities share one forward DP through the maximum requested
capacity.  Parent choices for every layer are retained, then each requested
capacity is backtracked from that single table construction.
"""
from __future__ import annotations

from itertools import product
import json
from typing import Iterable, Sequence

import numpy as np


GROUP_SIZE = 40
K_MIN = 5
K_MAX = 50
BUDGET_MEANS = (10, 15, 20, 30, 40)
TIE_BREAK = "lexicographically smallest K vector in fixed input-row order"


def _integer_vector(values: Iterable[int], name: str) -> np.ndarray:
    """Return a non-empty one-dimensional int64 vector without truncation."""

    items = list(values)
    if not items:
        raise ValueError(f"{name} must be non-empty")
    converted: list[int] = []
    for value in items:
        if isinstance(value, (bool, np.bool_)):
            raise ValueError(f"{name} must contain integers, not booleans")
        try:
            numeric = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{name} must contain finite integers") from exc
        if not np.isfinite(numeric) or not numeric.is_integer():
            raise ValueError(f"{name} must contain finite integers")
        integer = int(value)
        if float(integer) != numeric:
            raise ValueError(f"{name} contains an integer outside exact float64 range")
        converted.append(integer)
    return np.asarray(converted, dtype=np.int64)


def choice_values_from_marginals(
    marginal_values: np.ndarray,
    *,
    k_min: int = K_MIN,
    k_max: int = K_MAX,
) -> np.ndarray:
    """Convert per-slot marginals to cumulative utilities for every allowed K.

    ``marginal_values[i, j]`` is the value of slot ``j + 1`` for image ``i``.
    The result column ``k - k_min`` is the float64 sum of slots ``1..k``.
    No monotonicity or sign restriction is imposed on the marginals.
    """

    if isinstance(k_min, (bool, np.bool_)) or isinstance(k_max, (bool, np.bool_)):
        raise ValueError("k_min and k_max must be integers")
    if int(k_min) != k_min or int(k_max) != k_max:
        raise ValueError("k_min and k_max must be integers")
    k_min = int(k_min)
    k_max = int(k_max)
    if k_min < 0 or k_max < k_min:
        raise ValueError("require 0 <= k_min <= k_max")

    values = np.asarray(marginal_values, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] < k_max:
        raise ValueError(f"marginal_values must have shape [N, at least {k_max}]")
    if not np.isfinite(values).all():
        raise ValueError("marginal_values contains NaN or infinity")

    prefix = np.empty((values.shape[0], k_max + 1), dtype=np.float64)
    prefix[:, 0] = np.float64(0.0)
    with np.errstate(over="ignore", invalid="ignore"):
        np.cumsum(
            values[:, :k_max],
            axis=1,
            dtype=np.float64,
            out=prefix[:, 1:],
        )
    if not np.isfinite(prefix).all():
        raise FloatingPointError("float64 overflow while accumulating marginals")
    return prefix[:, k_min : k_max + 1].copy()


def choice_values_from_optional_marginals(
    marginal_values: np.ndarray,
) -> np.ndarray:
    """Convert optional next-slot marginals to consecutive K-choice values.

    This is the native LC-ALLOC-P1 representation: column zero is the marginal
    for moving from K=5 to K=6, and the final column moves from K=49 to K=50.
    The returned first choice, K=5, has utility zero; later choices are prefix
    sums of the supplied optional marginals.  A per-image K=5 base utility is
    constant across every feasible allocation and is therefore intentionally
    omitted from both optimization and the reported predicted objective.
    """

    values = np.asarray(marginal_values, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] < 1:
        raise ValueError("marginal_values must have non-empty shape [N, optional_slots]")
    if not np.isfinite(values).all():
        raise ValueError("marginal_values contains NaN or infinity")
    choices = np.empty((values.shape[0], values.shape[1] + 1), dtype=np.float64)
    choices[:, 0] = np.float64(0.0)
    with np.errstate(over="ignore", invalid="ignore"):
        np.cumsum(values, axis=1, dtype=np.float64, out=choices[:, 1:])
    if not np.isfinite(choices).all():
        raise FloatingPointError("float64 overflow while accumulating marginals")
    return choices


def _validated_choice_problem(
    choice_values: np.ndarray,
    capacities: Sequence[int],
    k_min: int,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    if isinstance(k_min, (bool, np.bool_)) or int(k_min) != k_min or int(k_min) < 0:
        raise ValueError("k_min must be a non-negative integer")
    k_min = int(k_min)
    values = np.asarray(choice_values, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] < 1:
        raise ValueError("choice_values must have non-empty shape [N, number_of_K_choices]")
    if not np.isfinite(values).all():
        raise ValueError("choice_values contains NaN or infinity")
    target_capacities = _integer_vector(capacities, "capacities")
    image_count, choice_count = values.shape
    k_max = k_min + choice_count - 1
    minimum = image_count * k_min
    maximum = image_count * k_max
    if np.any(target_capacities < minimum) or np.any(target_capacities > maximum):
        raise ValueError(
            f"every exact capacity must be in [{minimum}, {maximum}] for this problem"
        )
    return values, target_capacities, k_min, k_max


def solve_exact_multiple_choice(
    choice_values: np.ndarray,
    capacities: Sequence[int],
    *,
    k_min: int = K_MIN,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve all exact capacities with one float64 forward DP.

    ``choice_values[i, k-k_min]`` is the complete utility obtained by choosing
    K=``k`` for image ``i``.  Columns therefore represent mutually exclusive
    choices, not marginal slots.  The values may have any finite shape across
    K; the algorithm makes no concavity or monotonicity assumption.

    Returns ``(K, objectives)``.  Row ``r`` of K and element ``r`` of
    objectives correspond to ``capacities[r]``.  Duplicate capacities and an
    unsorted request order are preserved.
    """

    values, target_capacities, k_min, k_max = _validated_choice_problem(
        choice_values, capacities, k_min
    )
    image_count = int(values.shape[0])
    max_capacity = int(np.max(target_capacities))
    unreachable_rank = np.iinfo(np.int64).max

    # A state rank orders canonical prefixes lexicographically.  On an exact
    # objective tie, candidate keys (previous_prefix_rank, current_K) therefore
    # select the lexicographically smallest complete prefix without consulting
    # utilities beyond the equality comparison.
    previous_objective = np.full(max_capacity + 1, -np.inf, dtype=np.float64)
    previous_rank = np.full(max_capacity + 1, unreachable_rank, dtype=np.int64)
    previous_objective[0] = np.float64(0.0)
    previous_rank[0] = 0
    parent_k = np.full((image_count, max_capacity + 1), -1, dtype=np.int64)

    for image_index in range(image_count):
        current_objective = np.full(max_capacity + 1, -np.inf, dtype=np.float64)
        current_parent_rank = np.full(
            max_capacity + 1, unreachable_rank, dtype=np.int64
        )
        current_k = np.full(max_capacity + 1, -1, dtype=np.int64)

        for offset, k in enumerate(range(k_min, k_max + 1)):
            if k > max_capacity:
                break
            source_objective = previous_objective[: max_capacity + 1 - k]
            source_rank = previous_rank[: max_capacity + 1 - k]
            reachable = source_rank != unreachable_rank
            if not np.any(reachable):
                continue
            with np.errstate(over="ignore", invalid="ignore"):
                candidate = source_objective + values[image_index, offset]
            if not np.isfinite(candidate[reachable]).all():
                raise FloatingPointError("float64 overflow during DP accumulation")

            target_objective = current_objective[k:]
            target_parent_rank = current_parent_rank[k:]
            target_k = current_k[k:]
            better_objective = candidate > target_objective
            equal_objective = candidate == target_objective
            better_tie_key = (source_rank < target_parent_rank) | (
                (source_rank == target_parent_rank) & (k < target_k)
            )
            take = reachable & (better_objective | (equal_objective & better_tie_key))
            target_objective[take] = candidate[take]
            target_parent_rank[take] = source_rank[take]
            target_k[take] = k

        reachable_destinations = np.flatnonzero(current_k >= 0)
        if reachable_destinations.size == 0:
            raise AssertionError(f"DP has no reachable state after image {image_index}")

        # Rank selected canonical prefixes for use by the next layer.  The
        # primary key is the previous prefix's lexicographic rank; current K is
        # the secondary key because it is the last coordinate of this prefix.
        order = np.lexsort(
            (
                current_k[reachable_destinations],
                current_parent_rank[reachable_destinations],
            )
        )
        current_rank = np.full(max_capacity + 1, unreachable_rank, dtype=np.int64)
        current_rank[reachable_destinations[order]] = np.arange(
            reachable_destinations.size, dtype=np.int64
        )
        parent_k[image_index] = current_k
        previous_objective = current_objective
        previous_rank = current_rank

    if np.any(previous_rank[target_capacities] == unreachable_rank):
        raise AssertionError("DP did not reach a requested feasible exact capacity")

    allocations = np.empty(
        (target_capacities.size, image_count), dtype=np.int64
    )
    for target_index, capacity_value in enumerate(target_capacities):
        remaining = int(capacity_value)
        for image_index in range(image_count - 1, -1, -1):
            chosen = int(parent_k[image_index, remaining])
            if chosen < k_min or chosen > k_max:
                raise AssertionError(
                    f"invalid parent while backtracking capacity {capacity_value}"
                )
            allocations[target_index, image_index] = chosen
            remaining -= chosen
        if remaining != 0:
            raise AssertionError(
                f"backtracking failed exact capacity {capacity_value}: remainder {remaining}"
            )

    objectives = previous_objective[target_capacities].astype(np.float64, copy=True)

    # Recompute in the same fixed image order as the DP.  This guards parent
    # corruption while retaining exactly the float64 arithmetic being reported.
    for target_index in range(target_capacities.size):
        reconstructed = np.float64(0.0)
        for image_index, chosen in enumerate(allocations[target_index]):
            reconstructed = np.float64(
                reconstructed + values[image_index, int(chosen) - k_min]
            )
        if reconstructed != objectives[target_index]:
            raise AssertionError("DP objective and backtracked allocation disagree")
        if int(np.sum(allocations[target_index], dtype=np.int64)) != int(
            target_capacities[target_index]
        ):
            raise AssertionError("backtracked allocation does not use the exact budget")

    return allocations, objectives


def solve_group_choice_values(
    choice_values: np.ndarray,
    budget_means: Sequence[int] = BUDGET_MEANS,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve a production 40-image group from cumulative K=5..50 utilities."""

    values = np.asarray(choice_values, dtype=np.float64)
    expected_shape = (GROUP_SIZE, K_MAX - K_MIN + 1)
    if values.shape != expected_shape:
        raise ValueError(f"choice_values must have production shape {expected_shape}")
    means = _integer_vector(budget_means, "budget_means")
    if np.any(means < K_MIN) or np.any(means > K_MAX):
        raise ValueError(f"budget_means must lie in [{K_MIN}, {K_MAX}]")
    capacities = means * np.int64(GROUP_SIZE)
    return solve_exact_multiple_choice(values, capacities, k_min=K_MIN)


def group_choice_values_from_marginals(marginal_values: np.ndarray) -> np.ndarray:
    """Build the production [40,46] choice matrix from [40,45] ranks 6..50."""

    marginals = np.asarray(marginal_values, dtype=np.float64)
    expected_shape = (GROUP_SIZE, K_MAX - K_MIN)
    if marginals.shape != expected_shape:
        raise ValueError(
            "marginal_values must have production shape "
            f"{expected_shape} for additions K=6..50"
        )
    choices = choice_values_from_optional_marginals(marginals)
    if choices.shape != (GROUP_SIZE, K_MAX - K_MIN + 1):
        raise AssertionError("production choice-utility construction has the wrong shape")
    if not np.array_equal(choices[:, 0], np.zeros(GROUP_SIZE, dtype=np.float64)):
        raise AssertionError("production K=5 utility must be exactly zero")
    return choices


def solve_group_allocations(
    marginal_values: np.ndarray,
    budget_means: Sequence[int] = BUDGET_MEANS,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve a production group from its 45 optional next-slot marginals.

    With the default anchors the returned K rows correspond, in order, to mean
    budgets 10, 15, 20, 30, and 40, and use exact total capacities 400, 600,
    800, 1200, and 1600.  Input columns correspond to additions K=6 through
    K=50; K=5 has predicted utility zero.
    """

    choices = group_choice_values_from_marginals(marginal_values)
    return solve_group_choice_values(choices, budget_means)


def solve_group_full_slot_marginals(
    marginal_values: np.ndarray,
    budget_means: Sequence[int] = BUDGET_MEANS,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve from full slot-1..50 marginals when that representation is needed."""

    marginals = np.asarray(marginal_values, dtype=np.float64)
    if marginals.shape != (GROUP_SIZE, K_MAX):
        raise ValueError(
            f"marginal_values must have full-slot shape {(GROUP_SIZE, K_MAX)}"
        )
    choices = choice_values_from_marginals(
        marginals, k_min=K_MIN, k_max=K_MAX
    )
    return solve_group_choice_values(choices, budget_means)


def brute_force_exact_multiple_choice(
    choice_values: np.ndarray,
    capacities: Sequence[int],
    *,
    k_min: int = K_MIN,
    maximum_vectors: int = 2_000_000,
) -> tuple[np.ndarray, np.ndarray]:
    """Independent exhaustive oracle for tiny validation problems.

    This intentionally enumerates complete K vectors and does not reuse DP
    states, parents, transition ordering, or rank logic.  It is not intended
    for a 40-image production group.
    """

    values, target_capacities, k_min, k_max = _validated_choice_problem(
        choice_values, capacities, k_min
    )
    image_count = int(values.shape[0])
    vector_count = (k_max - k_min + 1) ** image_count
    if vector_count > maximum_vectors:
        raise ValueError(
            f"brute force would enumerate {vector_count} vectors; limit is {maximum_vectors}"
        )

    requested = {int(capacity) for capacity in target_capacities}
    best_vector: dict[int, tuple[int, ...]] = {}
    best_objective: dict[int, np.float64] = {}
    for vector in product(range(k_min, k_max + 1), repeat=image_count):
        capacity = sum(vector)
        if capacity not in requested:
            continue
        objective = np.float64(0.0)
        for image_index, chosen in enumerate(vector):
            objective = np.float64(
                objective + values[image_index, chosen - k_min]
            )
        incumbent = best_objective.get(capacity)
        if (
            incumbent is None
            or objective > incumbent
            or (
                objective == incumbent
                and vector < best_vector[capacity]
            )
        ):
            best_objective[capacity] = objective
            best_vector[capacity] = vector

    if requested != set(best_vector):
        raise AssertionError("brute force did not find every feasible exact capacity")
    allocations = np.asarray(
        [best_vector[int(capacity)] for capacity in target_capacities], dtype=np.int64
    )
    objectives = np.asarray(
        [best_objective[int(capacity)] for capacity in target_capacities],
        dtype=np.float64,
    )
    return allocations, objectives


def _brute_force_at_most(
    choice_values: np.ndarray,
    capacity: int,
    *,
    k_min: int,
) -> tuple[tuple[int, ...], np.float64]:
    """Tiny independent oracle used only to prove exact-vs-at-most behavior."""

    values = np.asarray(choice_values, dtype=np.float64)
    k_max = k_min + values.shape[1] - 1
    best_vector: tuple[int, ...] | None = None
    best_objective: np.float64 | None = None
    for vector in product(range(k_min, k_max + 1), repeat=values.shape[0]):
        if sum(vector) > capacity:
            continue
        objective = np.float64(0.0)
        for image_index, chosen in enumerate(vector):
            objective = np.float64(
                objective + values[image_index, chosen - k_min]
            )
        if (
            best_objective is None
            or objective > best_objective
            or (objective == best_objective and vector < best_vector)
        ):
            best_vector = vector
            best_objective = objective
    if best_vector is None or best_objective is None:
        raise AssertionError("at-most brute force found no feasible allocation")
    return best_vector, best_objective


def self_test() -> dict[str, object]:
    """Run deterministic synthetic DP parity checks against brute force.

    There are nine hand-authored edge instances and twenty seeded instances.
    Every tiny instance is checked for objective, complete K-vector tie result,
    and exact capacity.  A separate 40-image smoke check covers all five
    production mean-budget anchors in one solver call.
    """

    def case_from_marginals(
        name: str,
        marginals: Sequence[Sequence[float]],
        k_min: int,
        capacities: Sequence[int],
        *,
        expected_k: Sequence[Sequence[int]] | None = None,
        leaving_unused: bool = False,
    ) -> dict[str, object]:
        raw = np.asarray(marginals, dtype=np.float64)
        choices = choice_values_from_marginals(
            raw, k_min=k_min, k_max=raw.shape[1]
        )
        return {
            "name": name,
            "choices": choices,
            "k_min": k_min,
            "capacities": tuple(capacities),
            "expected_k": expected_k,
            "leaving_unused": leaving_unused,
        }

    cases: list[dict[str, object]] = [
        case_from_marginals(
            "lower_saturation",
            [[3, -2, 7], [1, 4, -5], [2, 0, 9]],
            1,
            [3],
            expected_k=[(1, 1, 1)],
        ),
        case_from_marginals(
            "upper_saturation",
            [[3, -2, 7], [1, 4, -5], [2, 0, 9]],
            1,
            [9],
            expected_k=[(3, 3, 3)],
        ),
        case_from_marginals(
            "all_zero_ties",
            np.zeros((3, 4), dtype=np.float64),
            1,
            [3, 6, 12],
            expected_k=[(1, 1, 1), (1, 1, 4), (4, 4, 4)],
        ),
        case_from_marginals(
            "identical_linear_ties",
            [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]],
            1,
            [5, 8],
            expected_k=[(1, 1, 3), (1, 3, 4)],
        ),
        case_from_marginals(
            "nonconcave_zero_to_one",
            [[3, 0, 1, -4], [2, 0, 1, -4], [1, 0, 1, -4]],
            1,
            [5, 7],
        ),
        case_from_marginals(
            "forced_exact_negative_slots",
            [[0, -4, -4], [0, -3, -3]],
            1,
            [5],
            leaving_unused=True,
        ),
        case_from_marginals(
            "forced_exact_negative_tail",
            [[5, -8, -9, -10], [4, -7, -8, -9], [3, -6, -7, -8]],
            1,
            [8],
            leaving_unused=True,
        ),
        case_from_marginals(
            "mixed_sign_nonmonotone",
            [[2, -5, 8, -1], [-1, 7, -3, 6], [0, 0, 4, -9]],
            0,
            [2, 5, 9],
        ),
        case_from_marginals(
            "production_K_labels_tiny_N",
            [[1, 2, 3, 4, 5, 0, 1], [7, 6, 5, 4, 3, 1, 0]],
            5,
            [10, 12, 14],
        ),
    ]

    rng = np.random.default_rng(20260914)
    for case_index in range(20):
        image_count = 2 + case_index % 3
        k_min = case_index % 2
        k_max = k_min + 2 + case_index % 3
        marginals = rng.integers(
            -4, 6, size=(image_count, k_max), dtype=np.int64
        ).astype(np.float64)
        minimum = image_count * k_min
        maximum = image_count * k_max
        capacities = sorted(
            {
                minimum,
                maximum,
                (minimum + maximum) // 2,
                int(rng.integers(minimum, maximum + 1)),
            }
        )
        cases.append(
            case_from_marginals(
                f"seeded_{case_index:02d}", marginals, k_min, capacities
            )
        )

    checked_capacities = 0
    exact_vs_at_most_checks = 0
    for case in cases:
        choices = np.asarray(case["choices"], dtype=np.float64)
        k_min = int(case["k_min"])
        capacities = tuple(int(value) for value in case["capacities"])
        actual_k, actual_objectives = solve_exact_multiple_choice(
            choices, capacities, k_min=k_min
        )
        expected_k, expected_objectives = brute_force_exact_multiple_choice(
            choices, capacities, k_min=k_min
        )
        if not np.array_equal(actual_k, expected_k):
            raise AssertionError(
                f"{case['name']}: K mismatch: {actual_k.tolist()} != {expected_k.tolist()}"
            )
        if not np.array_equal(actual_objectives, expected_objectives):
            raise AssertionError(
                f"{case['name']}: objective mismatch: "
                f"{actual_objectives.tolist()} != {expected_objectives.tolist()}"
            )
        requested = np.asarray(capacities, dtype=np.int64)
        if not np.array_equal(
            np.sum(actual_k, axis=1, dtype=np.int64), requested
        ):
            raise AssertionError(f"{case['name']}: exact capacity was not consumed")
        if actual_objectives.dtype != np.dtype(np.float64):
            raise AssertionError(f"{case['name']}: objectives are not float64")
        expected_declared_k = case["expected_k"]
        if expected_declared_k is not None and not np.array_equal(
            actual_k, np.asarray(expected_declared_k, dtype=np.int64)
        ):
            raise AssertionError(f"{case['name']}: declared edge-case K mismatch")

        if bool(case["leaving_unused"]):
            exact_capacity = capacities[0]
            at_most_k, at_most_objective = _brute_force_at_most(
                choices, exact_capacity, k_min=k_min
            )
            if not (
                sum(at_most_k) < exact_capacity
                and at_most_objective > actual_objectives[0]
            ):
                raise AssertionError(
                    f"{case['name']}: instance does not distinguish exact from at-most"
                )
            exact_vs_at_most_checks += 1
        checked_capacities += len(capacities)

    if len(cases) < 20:
        raise AssertionError("self-test must retain at least twenty tiny instances")

    # Production-shape smoke check: all-zero values make the declared
    # lexicographic tie result analytically transparent at every anchor.
    group_marginals = np.zeros((GROUP_SIZE, K_MAX - K_MIN), dtype=np.float64)
    group_k, group_objectives = solve_group_allocations(group_marginals)
    production_capacities = np.asarray(BUDGET_MEANS, dtype=np.int64) * GROUP_SIZE
    if group_k.shape != (len(BUDGET_MEANS), GROUP_SIZE):
        raise AssertionError("production K result has the wrong shape")
    if not np.array_equal(
        np.sum(group_k, axis=1, dtype=np.int64), production_capacities
    ):
        raise AssertionError("production smoke check missed an exact capacity")
    if not np.array_equal(group_objectives, np.zeros(len(BUDGET_MEANS))):
        raise AssertionError("all-zero production objectives must be zero")
    for row_index, capacity in enumerate(production_capacities):
        extra = int(capacity - GROUP_SIZE * K_MIN)
        analytic: list[int] = []
        for image_index in range(GROUP_SIZE):
            later_room = (GROUP_SIZE - image_index - 1) * (K_MAX - K_MIN)
            added_here = max(0, extra - later_room)
            analytic.append(K_MIN + added_here)
            extra -= added_here
        if not np.array_equal(group_k[row_index], np.asarray(analytic)):
            raise AssertionError("production all-zero lexicographic tie mismatch")

    return {
        "status": "PASS",
        "tiny_instances": len(cases),
        "tiny_capacity_checks": checked_capacities,
        "exact_vs_at_most_checks": exact_vs_at_most_checks,
        "production_group_size": GROUP_SIZE,
        "production_budget_means": list(BUDGET_MEANS),
        "production_capacity_checks": len(BUDGET_MEANS),
        "tie_break": TIE_BREAK,
        "objective_dtype": "float64",
    }


if __name__ == "__main__":
    print(json.dumps(self_test(), indent=2, sort_keys=True))
