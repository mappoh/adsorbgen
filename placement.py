"""Core geometry operations for adsorbate placement.

Handles approach direction detection, TM perturbation, and adsorbate
placement in both end-on and side-on binding modes.
"""

import numpy as np
from ase import Atoms
from ase.neighborlist import natural_cutoffs, NeighborList
from ase.data import atomic_numbers, covalent_radii

from chemistry import get_covalent_radius


def list_tm_sites(structure, element):
    """Find all atoms of the given element and print their info.

    Returns list of (index, position) tuples.
    """
    sites = []
    symbols = structure.get_chemical_symbols()
    positions = structure.get_positions()
    for i, sym in enumerate(symbols):
        if sym == element:
            sites.append((i, positions[i]))

    if not sites:
        print(f"No {element} atoms found in structure.")
        return sites

    print(f"Found {len(sites)} {element} site(s):")
    print(f"  {'Index':<8} {'X':>10} {'Y':>10} {'Z':>10}")
    print("  " + "-" * 40)
    for idx, pos in sites:
        print(f"  {idx:<8} {pos[0]:>10.4f} {pos[1]:>10.4f} {pos[2]:>10.4f}")

    return sites


def _get_neighbor_list(structure, cutoff_mult=1.2):
    """Build neighbor list using natural cutoffs scaled by cutoff_mult."""
    cutoffs = natural_cutoffs(structure, mult=cutoff_mult)
    nl = NeighborList(cutoffs, self_interaction=False, bothways=True)
    nl.update(structure)
    return nl


def detect_approach_direction(structure, tm_index):
    """Detect open approach direction for adsorbate at a TM site.

    Algorithm:
    1. Find TM neighbors within natural_cutoffs × 1.2
    2. Average vectors FROM neighbors TO TM → raw open direction
    3. Normalize → candidate approach direction
    4. Validate: no framework atoms within 15° cone along direction within 4 Å
    5. If fails, try perpendicular candidates

    Returns normalized direction vector (3,).
    """
    nl = _get_neighbor_list(structure)
    indices, offsets = nl.get_neighbors(tm_index)

    tm_pos = structure.positions[tm_index]
    cell = structure.get_cell()

    if len(indices) == 0:
        # No neighbors found — default to +z
        print("  Warning: no neighbors found for TM site, defaulting to +z direction")
        return np.array([0.0, 0.0, 1.0])

    # Compute vectors from each neighbor to TM
    neighbor_vecs = []
    for idx, offset in zip(indices, offsets):
        neighbor_pos = structure.positions[idx] + offset @ cell
        vec = tm_pos - neighbor_pos
        norm = np.linalg.norm(vec)
        if norm > 1e-8:
            neighbor_vecs.append(vec / norm)

    if not neighbor_vecs:
        return np.array([0.0, 0.0, 1.0])

    # Average the normalized vectors → points away from neighbors
    raw_dir = np.mean(neighbor_vecs, axis=0)
    norm = np.linalg.norm(raw_dir)

    if norm < 1e-6:
        # Symmetric coordination — try to find least crowded direction
        # Use SVD to find direction with least variance
        vecs = np.array(neighbor_vecs)
        _, _, Vt = np.linalg.svd(vecs)
        raw_dir = Vt[-1]  # Direction of least variance
        norm = np.linalg.norm(raw_dir)

    approach = raw_dir / norm

    # Validate: check for atoms in a 15° cone within 4 Å
    if _validate_approach(structure, tm_index, approach, cone_angle=15.0, radius=4.0):
        return approach

    # Try flipped direction
    if _validate_approach(structure, tm_index, -approach, cone_angle=15.0, radius=4.0):
        print("  Note: flipped approach direction (original blocked)")
        return -approach

    # Try perpendicular candidates
    for perp in _perpendicular_candidates(approach):
        if _validate_approach(structure, tm_index, perp, cone_angle=15.0, radius=4.0):
            print("  Note: using perpendicular approach direction (primary blocked)")
            return perp

    # Fall back to original with warning
    print("  Warning: could not validate approach direction — using best guess")
    return approach


def _validate_approach(structure, tm_index, direction, cone_angle=15.0, radius=4.0):
    """Check that no framework atoms are in a cone along direction."""
    tm_pos = structure.positions[tm_index]
    cos_threshold = np.cos(np.radians(cone_angle))

    positions = structure.get_positions()
    cell = structure.get_cell()
    pbc = structure.get_pbc()

    for i in range(len(structure)):
        if i == tm_index:
            continue

        # Handle PBC with mic
        diff = positions[i] - tm_pos
        if any(pbc):
            # Use minimum image convention
            scaled = np.linalg.solve(cell.T, diff)
            scaled -= np.round(scaled)
            diff = cell.T @ scaled

        dist = np.linalg.norm(diff)
        if dist < 1e-8 or dist > radius:
            continue

        cos_angle = np.dot(diff / dist, direction)
        if cos_angle > cos_threshold:
            return False

    return True


def _perpendicular_candidates(direction, n=6):
    """Generate n evenly-spaced perpendicular directions."""
    # Find a vector not parallel to direction
    if abs(direction[0]) < 0.9:
        aux = np.array([1.0, 0.0, 0.0])
    else:
        aux = np.array([0.0, 1.0, 0.0])

    perp1 = np.cross(direction, aux)
    perp1 /= np.linalg.norm(perp1)
    perp2 = np.cross(direction, perp1)
    perp2 /= np.linalg.norm(perp2)

    candidates = []
    for angle in np.linspace(0, 2 * np.pi, n, endpoint=False):
        candidates.append(np.cos(angle) * perp1 + np.sin(angle) * perp2)
    return candidates


def rodrigues_rotation(v, k, theta):
    """Rotate vector v around axis k by angle theta (radians) using Rodrigues' formula."""
    k = k / np.linalg.norm(k)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    return v * cos_t + np.cross(k, v) * sin_t + k * np.dot(k, v) * (1 - cos_t)


def perturb_tm(structure, tm_index, displacement, direction):
    """Displace a TM atom along a direction.

    Args:
        structure: ASE Atoms object (will be copied)
        tm_index: index of TM atom
        displacement: distance in Å
        direction: displacement direction vector (will be normalized)

    Returns:
        New Atoms object with perturbed TM position.
    """
    if abs(displacement) < 1e-8:
        return structure.copy()

    new_struct = structure.copy()
    direction = np.array(direction, dtype=float)
    direction = direction / np.linalg.norm(direction)
    new_struct.positions[tm_index] += displacement * direction
    return new_struct


def place_end_on(reference, tm_index, adsorbate_atoms, binding_atom_index,
                 approach_direction, bond_distance, tilt_angle, rotation_angle):
    """Place adsorbate in end-on binding mode.

    The adsorbate is oriented so that the binding atom is placed at
    bond_distance from TM along the approach direction, then tilted
    and rotated.

    Args:
        reference: ASE Atoms (possibly perturbed)
        tm_index: index of TM in reference
        adsorbate_atoms: list of (element, [x,y,z]) — binding atom at origin, bond along +z
        binding_atom_index: index of binding atom in adsorbate
        approach_direction: unit vector
        bond_distance: TM to binding atom distance (Å)
        tilt_angle: degrees from approach direction
        rotation_angle: azimuthal rotation (degrees)

    Returns:
        (combined_atoms, adsorbate_indices): combined ASE Atoms and list of
        adsorbate atom indices in the combined structure.
    """
    approach = np.array(approach_direction, dtype=float)
    approach = approach / np.linalg.norm(approach)

    # Build rotation matrix to transform from canonical (+z) to approach direction
    z_axis = np.array([0.0, 0.0, 1.0])

    # Get adsorbate positions relative to binding atom
    ads_positions = []
    ads_symbols = []
    for sym, pos in adsorbate_atoms:
        ads_symbols.append(sym)
        ads_positions.append(np.array(pos, dtype=float))
    ads_positions = np.array(ads_positions)

    # Shift so binding atom is at origin (should already be, but ensure)
    binding_pos = ads_positions[binding_atom_index].copy()
    ads_positions -= binding_pos

    # Step 1: Apply azimuthal rotation around +z (canonical axis)
    rot_rad = np.radians(rotation_angle)
    for i in range(len(ads_positions)):
        ads_positions[i] = rodrigues_rotation(ads_positions[i], z_axis, rot_rad)

    # Step 2: Apply tilt from +z axis
    if abs(tilt_angle) > 1e-6:
        # Tilt axis is perpendicular to z in the xz plane (after rotation)
        tilt_axis = _get_tilt_axis(z_axis, rotation_angle)
        tilt_rad = np.radians(tilt_angle)
        for i in range(len(ads_positions)):
            ads_positions[i] = rodrigues_rotation(ads_positions[i], tilt_axis, tilt_rad)

    # Step 3: Rotate from +z frame to approach direction frame
    rotation_matrix = _rotation_matrix_z_to_v(approach)
    ads_positions = (rotation_matrix @ ads_positions.T).T

    # Step 4: Translate so binding atom is at bond_distance from TM along approach
    tm_pos = reference.positions[tm_index]
    binding_point = tm_pos + approach * bond_distance
    ads_positions += binding_point

    # Build combined structure
    return _build_combined(reference, ads_symbols, ads_positions)


def place_side_on(reference, tm_index, adsorbate_atoms, binding_atom_indices,
                  approach_direction, bond_distance, tilt_angle, rotation_angle):
    """Place adsorbate in side-on (η2) binding mode.

    The midpoint of the two binding atoms is placed at bond_distance from TM.
    The binding face (plane of the two atoms) is perpendicular to the approach.

    Args:
        reference: ASE Atoms
        tm_index: index of TM
        adsorbate_atoms: list of (element, [x,y,z])
        binding_atom_indices: [i, j] indices of the two binding atoms
        approach_direction: unit vector
        bond_distance: TM to midpoint distance (Å)
        tilt_angle: degrees
        rotation_angle: degrees

    Returns:
        (combined_atoms, adsorbate_indices)
    """
    approach = np.array(approach_direction, dtype=float)
    approach = approach / np.linalg.norm(approach)

    ads_positions = []
    ads_symbols = []
    for sym, pos in adsorbate_atoms:
        ads_symbols.append(sym)
        ads_positions.append(np.array(pos, dtype=float))
    ads_positions = np.array(ads_positions)

    # Compute midpoint of binding atoms and center there
    i, j = binding_atom_indices
    midpoint = (ads_positions[i] + ads_positions[j]) / 2.0
    ads_positions -= midpoint

    # The binding axis (between the two atoms) should be perpendicular to approach
    # In canonical frame, C-C axis is along z, approach will be along -y (pointing down to TM)
    # We want binding axis perpendicular to approach direction

    # Step 1: Apply azimuthal rotation around canonical approach (-y → we use +z as canonical)
    z_axis = np.array([0.0, 0.0, 1.0])
    rot_rad = np.radians(rotation_angle)
    for k_idx in range(len(ads_positions)):
        ads_positions[k_idx] = rodrigues_rotation(ads_positions[k_idx], z_axis, rot_rad)

    # Step 2: Apply tilt
    if abs(tilt_angle) > 1e-6:
        tilt_axis = _get_tilt_axis(z_axis, rotation_angle)
        tilt_rad = np.radians(tilt_angle)
        for k_idx in range(len(ads_positions)):
            ads_positions[k_idx] = rodrigues_rotation(ads_positions[k_idx], tilt_axis, tilt_rad)

    # Step 3: Rotate to approach frame
    # For side-on, we want the binding atom pair perpendicular to approach
    # First rotate the whole molecule so approach aligns properly
    rotation_matrix = _rotation_matrix_z_to_v(approach)
    ads_positions = (rotation_matrix @ ads_positions.T).T

    # The binding axis after rotation should be approximately perpendicular to approach
    # (since it started along z in canonical and we rotated z→approach, the binding
    #  axis is now along approach; we need to fix this)
    # Actually for side-on, the C=C bond in canonical is along z (the molecule def).
    # We want C=C perpendicular to approach. So we need a different mapping.
    # Let's rotate 90° around a perpendicular axis first.
    if abs(np.dot(approach, z_axis)) < 0.99:
        perp = np.cross(approach, z_axis)
        perp /= np.linalg.norm(perp)
    else:
        perp = np.array([1.0, 0.0, 0.0])
    ads_positions_corrected = []
    for pos in ads_positions:
        ads_positions_corrected.append(rodrigues_rotation(pos, perp, np.pi / 2))
    ads_positions = np.array(ads_positions_corrected)

    # Step 4: Translate midpoint to bond_distance from TM along approach
    tm_pos = reference.positions[tm_index]
    midpoint_target = tm_pos + approach * bond_distance
    ads_positions += midpoint_target

    return _build_combined(reference, ads_symbols, ads_positions)


def _get_tilt_axis(approach, rotation_angle_deg):
    """Get tilt axis perpendicular to approach, oriented by rotation angle."""
    if abs(approach[0]) < 0.9:
        aux = np.array([1.0, 0.0, 0.0])
    else:
        aux = np.array([0.0, 1.0, 0.0])

    perp1 = np.cross(approach, aux)
    perp1 /= np.linalg.norm(perp1)
    perp2 = np.cross(approach, perp1)
    perp2 /= np.linalg.norm(perp2)

    # Tilt axis rotates with azimuthal angle
    rot_rad = np.radians(rotation_angle_deg)
    return np.cos(rot_rad) * perp1 + np.sin(rot_rad) * perp2


def _rotation_matrix_z_to_v(v):
    """Compute rotation matrix that maps +z to vector v."""
    v = v / np.linalg.norm(v)
    z = np.array([0.0, 0.0, 1.0])

    if np.allclose(v, z):
        return np.eye(3)
    if np.allclose(v, -z):
        return np.diag([1.0, -1.0, -1.0])

    axis = np.cross(z, v)
    axis_norm = np.linalg.norm(axis)
    axis = axis / axis_norm

    cos_a = np.dot(z, v)
    sin_a = axis_norm

    # Skew-symmetric matrix
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0],
    ])

    R = np.eye(3) + sin_a * K + (1 - cos_a) * (K @ K)
    return R


def _build_combined(reference, ads_symbols, ads_positions):
    """Combine reference and adsorbate into a single Atoms object."""
    n_ref = len(reference)

    # Build combined atoms
    all_symbols = list(reference.get_chemical_symbols()) + ads_symbols
    all_positions = np.vstack([reference.get_positions(), ads_positions])

    combined = Atoms(
        symbols=all_symbols,
        positions=all_positions,
        cell=reference.get_cell(),
        pbc=reference.get_pbc(),
    )

    adsorbate_indices = list(range(n_ref, n_ref + len(ads_symbols)))
    return combined, adsorbate_indices
