"""Validation pipeline for adsorption configurations.

9 rules: 3 hard reject, 6 scored. Configurations must pass all hard rules
and achieve min_score to be written.
"""

import numpy as np
from dataclasses import dataclass, field
from ase.geometry import get_distances
from ase.neighborlist import natural_cutoffs, NeighborList

from .chemistry import (
    get_covalent_radius,
    get_vdw_radius,
    get_bond_range,
    get_max_coordination,
)


@dataclass
class RuleResult:
    rule_id: str
    rule_name: str
    passed: bool
    hard_reject: bool
    score: float  # 0-max_points for scored rules, 0 or 1 for hard rules
    max_points: float
    message: str


@dataclass
class ValidationResult:
    passed: bool
    score: int
    details: list = field(default_factory=list)

    @property
    def rejection_reasons(self):
        return [d.message for d in self.details if not d.passed]


def validate_config(combined, reference, adsorbate_atoms, tm_index,
                    binding_indices, molecule_def, system_type, config):
    """Run full validation pipeline on a combined structure.

    Args:
        combined: ASE Atoms with reference + adsorbate
        reference: original ASE Atoms (without adsorbate)
        adsorbate_atoms: list of (element, [x,y,z]) original adsorbate geometry
        tm_index: index of TM in combined
        binding_indices: list of adsorbate atom indices in combined
        molecule_def: molecule definition dict from molecules.py
        system_type: "zeolite", "pom", or "slab"
        config: parsed YAML config dict

    Returns:
        ValidationResult
    """
    n_ref = len(reference)
    ads_indices = list(range(n_ref, len(combined)))
    min_dist = config.get("validation", {}).get("min_distance", 1.0)
    min_score = config.get("validation", {}).get("min_score", 70)

    results = []

    # Hard reject rules
    results.append(_check_overlap(combined, n_ref, ads_indices, tm_index,
                                  binding_indices, min_dist))
    results.append(_check_internal_geometry(combined, ads_indices,
                                            adsorbate_atoms, molecule_def))
    results.append(_check_not_in_wall(combined, n_ref, ads_indices))

    # Scored rules
    tm_symbol = combined.get_chemical_symbols()[tm_index]
    results.append(_check_steric(combined, n_ref, ads_indices))
    results.append(_check_bond_distance(combined, tm_index, binding_indices, tm_symbol))
    results.append(_check_coordination(combined, reference, tm_index, n_ref, tm_symbol))
    results.append(_check_bond_angle(combined, tm_index, binding_indices,
                                     ads_indices, molecule_def))

    if system_type == "zeolite":
        results.append(_check_channel_blockage(combined, n_ref, ads_indices))
    else:
        # Give full points if not zeolite
        results.append(RuleResult("R8", "Channel blockage", True, False, 10, 10,
                                  "N/A (not zeolite)"))

    results.append(_check_framework_competition(combined, reference, tm_index,
                                                binding_indices, n_ref))

    # Aggregate
    hard_fail = any(r.hard_reject and not r.passed for r in results)
    total_score = sum(r.score for r in results if not r.hard_reject)
    max_score = sum(r.max_points for r in results if not r.hard_reject)

    if max_score > 0:
        normalized_score = int(round(100 * total_score / max_score))
    else:
        normalized_score = 100

    passed = not hard_fail and normalized_score >= min_score

    return ValidationResult(passed=passed, score=normalized_score, details=results)


def _check_overlap(combined, n_ref, ads_indices, tm_index, binding_indices, min_dist):
    """R1: No adsorbate atom within 0.75 × (r_cov_i + r_cov_j) of any reference atom.
    Exclude intended TM-binding pair."""
    symbols = combined.get_chemical_symbols()
    positions = combined.get_positions()
    cell = combined.get_cell()
    pbc = combined.get_pbc()

    # Pairs to exclude: TM with each binding atom
    exclude_pairs = set()
    for bi in binding_indices:
        exclude_pairs.add((tm_index, bi))
        exclude_pairs.add((bi, tm_index))

    for ads_i in ads_indices:
        for ref_j in range(n_ref):
            if (ref_j, ads_i) in exclude_pairs or (ads_i, ref_j) in exclude_pairs:
                continue

            diff = positions[ads_i] - positions[ref_j]
            if any(pbc):
                scaled = np.linalg.solve(cell.T, diff)
                scaled -= np.round(scaled)
                diff = cell.T @ scaled
            dist = np.linalg.norm(diff)

            cov_sum = (get_covalent_radius(symbols[ads_i]) +
                       get_covalent_radius(symbols[ref_j]))
            threshold = 0.75 * cov_sum

            if dist < max(threshold, min_dist):
                return RuleResult(
                    "R1", "Atom overlap", False, True, 0, 0,
                    f"Overlap: {symbols[ads_i]}(ads {ads_i}) - "
                    f"{symbols[ref_j]}(ref {ref_j}) = {dist:.3f} Å "
                    f"< {threshold:.3f} Å"
                )

    return RuleResult("R1", "Atom overlap", True, True, 0, 0, "No overlaps")


def _check_internal_geometry(combined, ads_indices, adsorbate_atoms, molecule_def):
    """R2: Internal bond lengths/angles preserved after placement (±0.02 Å, ±2°)."""
    positions = combined.get_positions()

    # Check bonds
    for i, j, expected_len in molecule_def.get("internal_bonds", []):
        actual_i = ads_indices[i]
        actual_j = ads_indices[j]
        actual_len = np.linalg.norm(positions[actual_i] - positions[actual_j])
        if abs(actual_len - expected_len) > 0.02:
            return RuleResult(
                "R2", "Adsorbate integrity", False, True, 0, 0,
                f"Bond {i}-{j} distorted: {actual_len:.4f} vs {expected_len:.4f} Å"
            )

    # Check angles
    for i, j, k, expected_angle in molecule_def.get("internal_angles", []):
        actual_i = ads_indices[i]
        actual_j = ads_indices[j]
        actual_k = ads_indices[k]
        v1 = positions[actual_i] - positions[actual_j]
        v2 = positions[actual_k] - positions[actual_j]
        cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cos_a = np.clip(cos_a, -1, 1)
        actual_angle = np.degrees(np.arccos(cos_a))
        if abs(actual_angle - expected_angle) > 2.0:
            return RuleResult(
                "R2", "Adsorbate integrity", False, True, 0, 0,
                f"Angle {i}-{j}-{k} distorted: {actual_angle:.1f}° vs {expected_angle:.1f}°"
            )

    return RuleResult("R2", "Adsorbate integrity", True, True, 0, 0,
                      "Internal geometry preserved")


def _check_not_in_wall(combined, n_ref, ads_indices):
    """R7: Adsorbate COM not embedded in framework wall."""
    positions = combined.get_positions()
    symbols = combined.get_chemical_symbols()
    cell = combined.get_cell()
    pbc = combined.get_pbc()

    # COM of adsorbate
    ads_positions = positions[ads_indices]
    com = np.mean(ads_positions, axis=0)

    # Find min distance from COM to any framework atom
    min_dist = float("inf")
    for j in range(n_ref):
        diff = com - positions[j]
        if any(pbc):
            scaled = np.linalg.solve(cell.T, diff)
            scaled -= np.round(scaled)
            diff = cell.T @ scaled
        dist = np.linalg.norm(diff)
        if dist < min_dist:
            min_dist = dist

    # Check against VdW radius (use generic value since COM isn't a real atom)
    vdw_threshold = 1.5  # Å — conservative threshold for COM being inside wall
    if min_dist < vdw_threshold:
        return RuleResult(
            "R7", "Not inside framework", False, True, 0, 0,
            f"Adsorbate COM too close to framework: {min_dist:.3f} Å < {vdw_threshold} Å"
        )

    return RuleResult("R7", "Not inside framework", True, True, 0, 0,
                      f"COM-framework distance: {min_dist:.2f} Å")


def _check_steric(combined, n_ref, ads_indices):
    """R3: Steric feasibility — adsorbate atoms vs framework VdW sum. Max 20 pts."""
    positions = combined.get_positions()
    symbols = combined.get_chemical_symbols()
    cell = combined.get_cell()
    pbc = combined.get_pbc()

    max_pts = 20
    worst_ratio = float("inf")

    for ads_i in ads_indices:
        for ref_j in range(n_ref):
            diff = positions[ads_i] - positions[ref_j]
            if any(pbc):
                scaled = np.linalg.solve(cell.T, diff)
                scaled -= np.round(scaled)
                diff = cell.T @ scaled
            dist = np.linalg.norm(diff)

            vdw_sum = (get_vdw_radius(symbols[ads_i]) +
                       get_vdw_radius(symbols[ref_j]))
            ratio = dist / vdw_sum

            if ratio < worst_ratio:
                worst_ratio = ratio

    # Hard floor at 0.85× VdW sum
    if worst_ratio < 0.85:
        return RuleResult(
            "R3", "Steric feasibility", False, False, 0, max_pts,
            f"Steric clash: min ratio = {worst_ratio:.3f} < 0.85"
        )

    # Score: 1.0× → full points, 0.85× → 0 points, linear interpolation
    if worst_ratio >= 1.0:
        score = max_pts
    else:
        score = max_pts * (worst_ratio - 0.85) / 0.15

    return RuleResult(
        "R3", "Steric feasibility", True, False, score, max_pts,
        f"Min VdW ratio: {worst_ratio:.3f}"
    )


def _check_bond_distance(combined, tm_index, binding_indices, tm_symbol):
    """R4: TM-adsorbate bond distance in chemically reasonable range. Max 25 pts."""
    positions = combined.get_positions()
    symbols = combined.get_chemical_symbols()
    max_pts = 25

    # For each binding atom, check distance to TM
    scores = []
    for bi in binding_indices:
        dist = np.linalg.norm(positions[bi] - positions[tm_index])
        bind_element = symbols[bi]
        rmin, rideal, rmax = get_bond_range(tm_symbol, bind_element)

        if dist < rmin or dist > rmax:
            return RuleResult(
                "R4", "TM-adsorbate bond distance", False, False, 0, max_pts,
                f"Bond {tm_symbol}-{bind_element} = {dist:.3f} Å "
                f"outside [{rmin:.2f}, {rmax:.2f}]"
            )

        # Score: ideal → full, at min/max → half
        if abs(rideal - rmin) < 1e-6 and abs(rideal - rmax) < 1e-6:
            scores.append(max_pts)
        else:
            deviation = abs(dist - rideal) / max(rideal - rmin, rmax - rideal)
            scores.append(max_pts * (1.0 - 0.5 * min(deviation, 1.0)))

    avg_score = np.mean(scores) if scores else max_pts

    return RuleResult(
        "R4", "TM-adsorbate bond distance", True, False, avg_score, max_pts,
        f"Bond distance(s) within range"
    )


def _check_coordination(combined, reference, tm_index, n_ref, tm_symbol):
    """R5: TM coordination after adsorption ≤ max. Max 15 pts."""
    max_pts = 15
    max_cn = get_max_coordination(tm_symbol)

    # Count existing coordination in reference
    cutoffs_ref = natural_cutoffs(reference, mult=1.2)
    nl_ref = NeighborList(cutoffs_ref, self_interaction=False, bothways=True)
    nl_ref.update(reference)
    existing_cn = len(nl_ref.get_neighbors(tm_index)[0])

    # Count coordination in combined
    cutoffs_comb = natural_cutoffs(combined, mult=1.2)
    nl_comb = NeighborList(cutoffs_comb, self_interaction=False, bothways=True)
    nl_comb.update(combined)
    new_cn = len(nl_comb.get_neighbors(tm_index)[0])

    if new_cn > max_cn + 1:
        return RuleResult(
            "R5", "Coordination number", False, False, 0, max_pts,
            f"CN = {new_cn} exceeds max+1 = {max_cn + 1} for {tm_symbol}"
        )

    if new_cn > max_cn:
        score = max_pts * 0.3  # Penalty but not zero
        msg = f"CN = {new_cn} exceeds max = {max_cn} for {tm_symbol}"
    elif new_cn == max_cn:
        score = max_pts * 0.7
        msg = f"CN = {new_cn} at max = {max_cn} for {tm_symbol}"
    else:
        score = max_pts
        msg = f"CN = {new_cn} below max = {max_cn} for {tm_symbol}"

    return RuleResult("R5", "Coordination number", True, False, score, max_pts, msg)


def _check_bond_angle(combined, tm_index, binding_indices, ads_indices, molecule_def):
    """R6: TM-binding-next atom angle check. Max 15 pts."""
    max_pts = 15
    angle_info = molecule_def.get("tm_binding_angle")

    if angle_info is None:
        return RuleResult("R6", "TM-binding-next angle", True, False, max_pts, max_pts,
                          "N/A (no angle constraint)")

    positions = combined.get_positions()
    tm_pos = positions[tm_index]

    # binding_indices[0] is the primary binding atom in combined
    bind_idx = binding_indices[0]
    bind_pos = positions[bind_idx]

    # next_atom_index is relative to adsorbate
    next_local = angle_info["next_atom_index"]
    n_ref = len(combined) - len(ads_indices)
    next_idx = n_ref + next_local
    next_pos = positions[next_idx]

    # Compute TM-binding-next angle
    v1 = tm_pos - bind_pos
    v2 = next_pos - bind_pos
    cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
    cos_a = np.clip(cos_a, -1, 1)
    angle = np.degrees(np.arccos(cos_a))

    min_angle = angle_info["min_angle"]
    max_angle = angle_info["max_angle"]
    ideal_angle = (min_angle + max_angle) / 2

    if angle < min_angle - 10 or angle > max_angle + 10:
        score = 0
    elif angle < min_angle or angle > max_angle:
        # Slightly outside range
        deviation = min(abs(angle - min_angle), abs(angle - max_angle))
        score = max_pts * max(0, 1.0 - deviation / 10.0) * 0.5
    else:
        # Inside range — score by proximity to ideal
        range_half = (max_angle - min_angle) / 2
        if range_half > 0:
            deviation = abs(angle - ideal_angle) / range_half
        else:
            deviation = 0
        score = max_pts * (1.0 - 0.3 * deviation)

    return RuleResult(
        "R6", "TM-binding-next angle", True, False, score, max_pts,
        f"Angle = {angle:.1f}° (range: {min_angle}-{max_angle}°)"
    )


def _check_channel_blockage(combined, n_ref, ads_indices):
    """R8: Adsorbate doesn't reduce channel free diameter below 2.0 Å. Max 10 pts.
    Zeolite-only."""
    max_pts = 10
    positions = combined.get_positions()
    symbols = combined.get_chemical_symbols()
    cell = combined.get_cell()
    pbc = combined.get_pbc()

    # Approximate check: find the minimum distance between any adsorbate atom
    # and framework atoms on the opposite side of the channel.
    # This is simplified — full channel analysis would need Voronoi decomposition.

    ads_positions = positions[ads_indices]
    ads_com = np.mean(ads_positions, axis=0)

    # For each adsorbate atom, find the nearest framework atom
    # that is roughly on the opposite side (angle > 90° from COM-TM vector)
    min_clearance = float("inf")

    for ads_i in ads_indices:
        for ref_j in range(n_ref):
            diff = positions[ref_j] - positions[ads_i]
            if any(pbc):
                scaled = np.linalg.solve(cell.T, diff)
                scaled -= np.round(scaled)
                diff = cell.T @ scaled
            dist = np.linalg.norm(diff)

            # Subtract VdW radii to get free space
            vdw_sum = (get_vdw_radius(symbols[ads_i]) +
                       get_vdw_radius(symbols[ref_j]))
            clearance = dist - vdw_sum

            if clearance < min_clearance:
                min_clearance = clearance

    if min_clearance < 0:
        score = 0
        msg = f"Channel blocked: clearance = {min_clearance:.2f} Å"
    elif min_clearance < 2.0:
        score = max_pts * (min_clearance / 2.0)
        msg = f"Reduced channel: clearance = {min_clearance:.2f} Å"
    else:
        score = max_pts
        msg = f"Channel clear: {min_clearance:.2f} Å"

    return RuleResult("R8", "Channel blockage", True, False, score, max_pts, msg)


def _check_framework_competition(combined, reference, tm_index, binding_indices, n_ref):
    """R9: Angle between TM-adsorbate vector and existing TM-framework bonds > 30°.
    Max 15 pts."""
    max_pts = 15
    positions = combined.get_positions()
    tm_pos = positions[tm_index]

    # Get existing TM-framework bonds from reference
    cutoffs = natural_cutoffs(reference, mult=1.2)
    nl = NeighborList(cutoffs, self_interaction=False, bothways=True)
    nl.update(reference)
    neighbor_indices, offsets = nl.get_neighbors(tm_index)

    if len(neighbor_indices) == 0:
        return RuleResult("R9", "TM-framework competition", True, False, max_pts,
                          max_pts, "No existing TM-framework bonds")

    cell = reference.get_cell()

    # Compute TM-adsorbate vector(s)
    ads_vectors = []
    for bi in binding_indices:
        vec = positions[bi] - tm_pos
        norm = np.linalg.norm(vec)
        if norm > 1e-8:
            ads_vectors.append(vec / norm)

    if not ads_vectors:
        return RuleResult("R9", "TM-framework competition", True, False, max_pts,
                          max_pts, "No binding vectors")

    # Check angle with each framework bond
    min_angle = float("inf")
    for idx, offset in zip(neighbor_indices, offsets):
        fw_pos = reference.positions[idx] + offset @ cell
        fw_vec = fw_pos - tm_pos
        fw_norm = np.linalg.norm(fw_vec)
        if fw_norm < 1e-8:
            continue
        fw_vec /= fw_norm

        for ads_vec in ads_vectors:
            cos_a = np.dot(ads_vec, fw_vec)
            cos_a = np.clip(cos_a, -1, 1)
            angle = np.degrees(np.arccos(cos_a))
            if angle < min_angle:
                min_angle = angle

    if min_angle < 20:
        score = 0
        msg = f"Competing with framework bond: min angle = {min_angle:.1f}°"
    elif min_angle < 30:
        score = max_pts * (min_angle - 20) / 10
        msg = f"Near framework bond: min angle = {min_angle:.1f}°"
    else:
        # Score proportional to angle, capped at full points for > 60°
        score = max_pts * min(1.0, (min_angle - 30) / 30)
        score = max(score, max_pts * 0.7)  # At least 70% if > 30°
        msg = f"Clear of framework bonds: min angle = {min_angle:.1f}°"

    return RuleResult("R9", "TM-framework competition", True, False, score,
                      max_pts, msg)
