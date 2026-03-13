#!/usr/bin/env python3
"""Adsorption Configuration Generator (adsorbgen).

Generates diverse initial guess structures for VASP geometry optimization
by placing adsorbate molecules onto transition metal sites at various
bond distances, angles, and orientations.

Usage:
    adsorbgen init                          # Create config.yaml interactively
    adsorbgen init --non-interactive        # Copy example config
    adsorbgen run [config.yaml]             # Generate (default: ./config.yaml)
    adsorbgen run [config.yaml] --dry-run   # Preview
    adsorbgen sites [config.yaml]           # List TM sites
    adsorbgen molecules                     # List built-in molecules
"""

import argparse
import csv
import os
import shutil
import sys
from importlib import resources
from itertools import product

import numpy as np
import yaml
from ase.io import read, write

from .molecules import get_molecule, list_molecules, MOLECULES
from .placement import (
    list_tm_sites,
    detect_approach_direction,
    perturb_tm,
    place_end_on,
    place_side_on,
)
from .validation import validate_config


def _get_example_config_path():
    """Get the path to the bundled config_example.yaml."""
    if sys.version_info >= (3, 9):
        return resources.files("adsorbgen").joinpath("config_example.yaml")
    else:
        with resources.path("adsorbgen", "config_example.yaml") as p:
            return p


def load_config(config_path):
    """Load and validate YAML config."""
    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: config file '{config_path}' not found")
        print("Run 'adsorbgen init' to create one.")
        sys.exit(1)

    # Validate required fields
    required = ["reference", "site", "adsorbate", "parameters"]
    for key in required:
        if key not in config:
            print(f"Error: missing required config section '{key}'")
            sys.exit(1)

    if "poscar" not in config["reference"]:
        print("Error: reference.poscar is required")
        sys.exit(1)

    # Set defaults
    config.setdefault("perturbation", {"enabled": False})
    config.setdefault("validation", {})
    config["validation"].setdefault("min_score", 70)
    config["validation"].setdefault("min_distance", 1.0)
    config.setdefault("output", {})
    config["output"].setdefault("directory", "./adsorption_configs")
    config["output"].setdefault("naming", "config_{:04d}")
    config["output"].setdefault("summary_file", "summary.csv")
    config["reference"].setdefault("system_type", "pom")
    config["site"].setdefault("site_index", None)
    config["site"].setdefault("approach_direction", None)
    config["adsorbate"].setdefault("binding_mode", "end-on")
    config["adsorbate"].setdefault("binding_atom_index", None)

    return config


def load_adsorbate(config):
    """Load adsorbate from built-in molecule or external file."""
    source = config["adsorbate"]["source"]
    mol_def = get_molecule(source)

    if mol_def is not None:
        # Built-in molecule
        binding_mode = config["adsorbate"].get("binding_mode", mol_def["binding_mode"])

        # Override binding atom if specified
        if config["adsorbate"].get("binding_atom") and binding_mode == "end-on":
            # Find the atom matching the specified element
            target_el = config["adsorbate"]["binding_atom"]
            target_idx = config["adsorbate"].get("binding_atom_index")

            if target_idx is not None:
                mol_def = dict(mol_def)
                mol_def["binding_atom_index"] = target_idx
            else:
                # Find first matching element
                for i, (el, _) in enumerate(mol_def["atoms"]):
                    if el == target_el:
                        mol_def = dict(mol_def)
                        mol_def["binding_atom_index"] = i
                        break

        return mol_def, binding_mode

    # External file
    if not os.path.isfile(source):
        print(f"Error: adsorbate source '{source}' not found as built-in or file")
        sys.exit(1)

    try:
        atoms = read(source)
    except Exception as e:
        print(f"Error reading adsorbate file: {e}")
        sys.exit(1)

    # Build molecule definition from external file
    atom_list = []
    for sym, pos in zip(atoms.get_chemical_symbols(), atoms.get_positions()):
        atom_list.append((sym, pos.tolist()))

    binding_mode = config["adsorbate"].get("binding_mode", "end-on")
    binding_atom = config["adsorbate"].get("binding_atom")
    binding_atom_index = config["adsorbate"].get("binding_atom_index")

    if binding_atom and binding_atom_index is None:
        for i, (el, _) in enumerate(atom_list):
            if el == binding_atom:
                binding_atom_index = i
                break

    if binding_atom_index is None:
        binding_atom_index = 0

    mol_def = {
        "atoms": atom_list,
        "binding_atom": atom_list[binding_atom_index][0],
        "binding_atom_index": binding_atom_index,
        "binding_mode": binding_mode,
        "internal_bonds": [],
        "internal_angles": [],
        "tm_binding_angle": None,
    }

    if binding_mode == "side-on":
        mol_def["binding_atoms"] = config["adsorbate"].get("binding_atoms", [0, 1])

    return mol_def, binding_mode


def resolve_tm_site(structure, config):
    """Resolve which TM atom to use."""
    element = config["site"]["tm_element"]
    site_index = config["site"].get("site_index")

    if site_index is not None:
        sym = structure.get_chemical_symbols()[site_index]
        if sym != element:
            print(f"Warning: atom {site_index} is {sym}, not {element}")
        return site_index

    sites = list_tm_sites(structure, element)
    if not sites:
        sys.exit(1)

    if len(sites) == 1:
        print(f"  Using only {element} site: index {sites[0][0]}")
        return sites[0][0]

    # Multiple sites — prompt user
    while True:
        try:
            choice = input(f"\nSelect {element} site index: ").strip()
            idx = int(choice)
            if any(s[0] == idx for s in sites):
                return idx
            print(f"  Invalid index. Choose from: {[s[0] for s in sites]}")
        except (ValueError, EOFError):
            print("  Please enter a valid integer index.")
            if not sys.stdin.isatty():
                sys.exit(1)


def run_generation(config):
    """Main generation pipeline."""
    # Load reference structure
    ref_path = config["reference"]["poscar"]
    try:
        reference = read(ref_path)
    except Exception as e:
        print(f"Error reading reference structure: {e}")
        sys.exit(1)

    print(f"Reference: {ref_path} ({len(reference)} atoms)")
    print(f"System type: {config['reference']['system_type']}")

    # Resolve TM site
    tm_index = resolve_tm_site(reference, config)
    tm_symbol = reference.get_chemical_symbols()[tm_index]
    print(f"TM site: {tm_symbol} (index {tm_index})")

    # Detect or set approach direction
    if config["site"]["approach_direction"] is not None:
        approach = np.array(config["site"]["approach_direction"], dtype=float)
        approach /= np.linalg.norm(approach)
        print(f"Approach direction (manual): [{approach[0]:.3f}, {approach[1]:.3f}, {approach[2]:.3f}]")
    else:
        print("Detecting approach direction...")
        approach = detect_approach_direction(reference, tm_index)
        print(f"Approach direction (auto): [{approach[0]:.3f}, {approach[1]:.3f}, {approach[2]:.3f}]")

    # Load adsorbate
    mol_def, binding_mode = load_adsorbate(config)
    print(f"Adsorbate: {config['adsorbate']['source']} ({binding_mode})")

    # Parameters
    perturbation = config.get("perturbation", {})
    perturb_enabled = perturbation.get("enabled", False)
    displacements = perturbation.get("displacements", [0.0]) if perturb_enabled else [0.0]
    perturb_dir = perturbation.get("direction")
    if perturb_dir is not None:
        perturb_dir = np.array(perturb_dir, dtype=float)
        perturb_dir /= np.linalg.norm(perturb_dir)
    else:
        perturb_dir = approach

    bond_distances = config["parameters"]["bond_distances"]
    tilt_angles = config["parameters"].get("tilt_angles", [0])
    rotation_angles = config["parameters"].get("rotation_angles", [0])

    total = len(displacements) * len(bond_distances) * len(tilt_angles) * len(rotation_angles)
    print(f"\nTotal configurations to try: {total}")
    print(f"  Displacements: {displacements}")
    print(f"  Bond distances: {bond_distances}")
    print(f"  Tilt angles: {tilt_angles}")
    print(f"  Rotation angles: {rotation_angles}")
    print(f"  Min score: {config['validation']['min_score']}")

    # Create output directory
    out_dir = config["output"]["directory"]
    os.makedirs(out_dir, exist_ok=True)

    # Summary CSV
    summary_path = os.path.join(out_dir, config["output"]["summary_file"])
    summary_rows = []

    config_count = 0
    written_count = 0
    skipped_count = 0

    # Phase 1 & 2: Perturbation + Placement + Validation
    for disp in displacements:
        # Phase 1: Perturb TM
        ref_perturbed = perturb_tm(reference, tm_index, disp, perturb_dir)

        for dist, tilt, rot in product(bond_distances, tilt_angles, rotation_angles):
            config_count += 1

            # Phase 2: Place adsorbate
            if binding_mode == "end-on":
                binding_atom_index = mol_def["binding_atom_index"]
                combined, ads_indices = place_end_on(
                    ref_perturbed, tm_index, mol_def["atoms"],
                    binding_atom_index, approach, dist, tilt, rot,
                )
                # Binding indices in combined structure
                n_ref = len(ref_perturbed)
                binding_indices_combined = [n_ref + binding_atom_index]
            else:
                binding_atom_indices = mol_def.get("binding_atoms", [0, 1])
                combined, ads_indices = place_side_on(
                    ref_perturbed, tm_index, mol_def["atoms"],
                    binding_atom_indices, approach, dist, tilt, rot,
                )
                n_ref = len(ref_perturbed)
                binding_indices_combined = [n_ref + bi for bi in binding_atom_indices]

            # Validate
            result = validate_config(
                combined, ref_perturbed, mol_def["atoms"],
                tm_index, binding_indices_combined, mol_def,
                config["reference"]["system_type"], config,
            )

            # Folder name
            folder_name = config["output"]["naming"].format(config_count)

            row = {
                "config_id": config_count,
                "folder": folder_name,
                "displacement": disp,
                "bond_distance": dist,
                "tilt_angle": tilt,
                "rotation_angle": rot,
                "score": result.score,
                "passed": result.passed,
                "rejection_reason": "; ".join(result.rejection_reasons) if not result.passed else "",
            }
            summary_rows.append(row)

            if result.passed:
                # Write POSCAR
                config_dir = os.path.join(out_dir, folder_name)
                os.makedirs(config_dir, exist_ok=True)
                poscar_path = os.path.join(config_dir, "POSCAR")
                write(poscar_path, combined, format="vasp", vasp5=True)
                written_count += 1

                # Print progress for written configs
                if written_count % 10 == 0 or written_count == 1:
                    print(f"  Written: {written_count} (score={result.score})")
            else:
                skipped_count += 1

    # Write summary CSV
    if summary_rows:
        fieldnames = ["config_id", "folder", "displacement", "bond_distance",
                      "tilt_angle", "rotation_angle", "score", "passed",
                      "rejection_reason"]
        with open(summary_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)

    print(f"\nDone!")
    print(f"  Total tried: {config_count}")
    print(f"  Written: {written_count}")
    print(f"  Skipped: {skipped_count}")
    print(f"  Output: {out_dir}")
    print(f"  Summary: {summary_path}")


def run_dry_run(config):
    """Show parameter count and validation preview without writing."""
    perturbation = config.get("perturbation", {})
    perturb_enabled = perturbation.get("enabled", False)
    displacements = perturbation.get("displacements", [0.0]) if perturb_enabled else [0.0]

    bond_distances = config["parameters"]["bond_distances"]
    tilt_angles = config["parameters"].get("tilt_angles", [0])
    rotation_angles = config["parameters"].get("rotation_angles", [0])

    total = len(displacements) * len(bond_distances) * len(tilt_angles) * len(rotation_angles)

    print("=== Dry Run ===")
    print(f"Reference: {config['reference']['poscar']}")
    print(f"System type: {config['reference']['system_type']}")
    print(f"TM element: {config['site']['tm_element']}")
    print(f"Adsorbate: {config['adsorbate']['source']}")
    print(f"Binding mode: {config['adsorbate'].get('binding_mode', 'end-on')}")
    print()
    print(f"Displacements ({len(displacements)}): {displacements}")
    print(f"Bond distances ({len(bond_distances)}): {bond_distances}")
    print(f"Tilt angles ({len(tilt_angles)}): {tilt_angles}")
    print(f"Rotation angles ({len(rotation_angles)}): {rotation_angles}")
    print()
    print(f"Total configurations: {total}")
    print(f"Min score threshold: {config['validation']['min_score']}")
    print(f"Min distance threshold: {config['validation']['min_distance']} Å")
    print(f"Output directory: {config['output']['directory']}")


def run_list_sites(config):
    """List all TM sites in the reference structure."""
    ref_path = config["reference"]["poscar"]
    reference = read(ref_path)
    element = config["site"]["tm_element"]
    list_tm_sites(reference, element)


def _prompt(text, default=""):
    """Prompt user with a default value. Return the response or the default."""
    if default:
        response = input(f"{text} [{default}]: ").strip()
    else:
        response = input(f"{text}: ").strip()
    return response if response else default


def _parse_float_list(text):
    """Parse a comma-separated string of floats."""
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def run_init(non_interactive=False):
    """Create a config.yaml in the current directory."""
    output_path = os.path.join(os.getcwd(), "config.yaml")

    if os.path.exists(output_path):
        print(f"config.yaml already exists in {os.getcwd()}")
        overwrite = input("Overwrite? (y/n) [n]: ").strip().lower()
        if overwrite != "y":
            print("Aborted.")
            return

    if non_interactive:
        # Just copy the example config
        example_path = _get_example_config_path()
        shutil.copy2(str(example_path), output_path)
        print(f"Written: {output_path}")
        print("Edit this file to match your system, then run: adsorbgen run")
        return

    # Interactive prompts
    print("adsorbgen config generator")
    print("-" * 40)

    poscar = _prompt("Reference POSCAR path", "POSCAR")
    system_type = _prompt("System type (pom/zeolite/slab)", "pom")
    tm_element = _prompt("TM element", "Ni")
    site_index_str = _prompt("Site index (leave blank to pick later)", "")
    site_index = int(site_index_str) if site_index_str else None

    # Adsorbate
    available = ", ".join(MOLECULES.keys())
    source = _prompt(f"Adsorbate ({available}, or file path)", "H2O")

    # Determine default binding mode from molecule if built-in
    mol = get_molecule(source)
    default_mode = mol["binding_mode"] if mol else "end-on"
    binding_mode = _prompt("Binding mode (end-on/side-on)", default_mode)

    # Bond distances
    bond_str = _prompt("Bond distances (comma-separated)", "1.8, 2.0, 2.2")
    bond_distances = _parse_float_list(bond_str)

    tilt_str = _prompt("Tilt angles (comma-separated)", "0, 30, 45")
    tilt_angles = _parse_float_list(tilt_str)

    rot_str = _prompt("Rotation angles (comma-separated)", "0, 90, 180, 270")
    rotation_angles = _parse_float_list(rot_str)

    # Perturbation
    perturb_yn = _prompt("Enable TM perturbation? (y/n)", "n").lower()
    perturb_enabled = perturb_yn == "y"
    if perturb_enabled:
        disp_str = _prompt("Perturbation displacements (comma-separated)", "0.0, 0.1, 0.2")
        displacements = _parse_float_list(disp_str)
    else:
        displacements = [0.0]

    min_score_str = _prompt("Min validation score (0-100)", "70")
    min_score = int(min_score_str)

    out_dir = _prompt("Output directory", "./adsorption_configs")

    # Build config dict
    config = {
        "reference": {
            "poscar": poscar,
            "system_type": system_type,
        },
        "site": {
            "tm_element": tm_element,
            "site_index": site_index,
            "approach_direction": None,
        },
        "perturbation": {
            "enabled": perturb_enabled,
            "displacements": displacements,
            "direction": None,
        },
        "adsorbate": {
            "source": source,
            "binding_mode": binding_mode,
        },
        "parameters": {
            "bond_distances": bond_distances,
            "tilt_angles": [int(x) if x == int(x) else x for x in tilt_angles],
            "rotation_angles": [int(x) if x == int(x) else x for x in rotation_angles],
        },
        "validation": {
            "min_score": min_score,
            "min_distance": 1.0,
        },
        "output": {
            "directory": out_dir,
            "naming": "config_{:04d}",
            "summary_file": "summary.csv",
        },
    }

    # Add binding atom info for end-on built-in molecules
    if mol and binding_mode == "end-on":
        config["adsorbate"]["binding_atom"] = mol["binding_atom"]

    with open(output_path, "w") as f:
        # Write a header comment
        f.write("# adsorbgen configuration\n")
        f.write("# Generated by: adsorbgen init\n\n")
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"\nWritten: {output_path}")
    print("Run 'adsorbgen run' to generate configurations.")


def _is_subcommand(arg):
    """Check if an argument is a known subcommand."""
    return arg in ("init", "run", "sites", "molecules")


def main():
    # Check for legacy usage before argparse gets a chance to error
    # Legacy: adsorbgen config.yaml [--dry-run|--list-sites|--list-molecules]
    if len(sys.argv) > 1 and not _is_subcommand(sys.argv[1]) and not sys.argv[1].startswith("-"):
        # First arg looks like a config file path (not a subcommand, not a flag)
        legacy_parser = argparse.ArgumentParser(description="Adsorption Configuration Generator (legacy mode)")
        legacy_parser.add_argument("config")
        legacy_parser.add_argument("--dry-run", action="store_true")
        legacy_parser.add_argument("--list-sites", action="store_true")
        legacy_args = legacy_parser.parse_args()

        config = load_config(legacy_args.config)
        if legacy_args.list_sites:
            run_list_sites(config)
        elif legacy_args.dry_run:
            run_dry_run(config)
        else:
            run_generation(config)
        return

    if len(sys.argv) > 1 and sys.argv[1] == "--list-molecules":
        list_molecules()
        return

    parser = argparse.ArgumentParser(
        description="Adsorption Configuration Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest="command")

    # init subcommand
    init_parser = subparsers.add_parser("init", help="Create config.yaml interactively")
    init_parser.add_argument("--non-interactive", action="store_true",
                             help="Copy example config without prompting")

    # run subcommand
    run_parser = subparsers.add_parser("run", help="Generate adsorption configurations")
    run_parser.add_argument("config", nargs="?", default="config.yaml",
                            help="Path to YAML config file (default: config.yaml)")
    run_parser.add_argument("--dry-run", action="store_true",
                            help="Show parameter count without generating")

    # sites subcommand
    sites_parser = subparsers.add_parser("sites", help="List TM sites in reference structure")
    sites_parser.add_argument("config", nargs="?", default="config.yaml",
                              help="Path to YAML config file (default: config.yaml)")

    # molecules subcommand
    subparsers.add_parser("molecules", help="List available built-in molecules")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    if args.command == "init":
        run_init(non_interactive=args.non_interactive)

    elif args.command == "molecules":
        list_molecules()

    elif args.command == "run":
        config = load_config(args.config)
        if args.dry_run:
            run_dry_run(config)
        else:
            run_generation(config)

    elif args.command == "sites":
        config = load_config(args.config)
        run_list_sites(config)


if __name__ == "__main__":
    main()
