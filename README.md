# adsorbgen

Adsorption configuration generator for VASP geometry optimization. Places adsorbate molecules onto transition metal sites in POM clusters, zeolite frameworks, and slabs at various bond distances, angles, and orientations to generate diverse initial guess structures.

A 9-rule validation pipeline filters out chemically nonsensical configurations before writing.

## Install

```bash
git clone https://github.com/mappoh/adsorbgen.git
pip install ase numpy pyyaml
```

## Usage

```bash
# Show available built-in molecules
python adsorbgen.py --list-molecules

# List TM sites in your structure
python adsorbgen.py config.yaml --list-sites

# Preview how many configs will be generated
python adsorbgen.py config.yaml --dry-run

# Generate configurations
python adsorbgen.py config.yaml
```

## Config

Copy `config_example.yaml` and edit to match your system:

```yaml
reference:
  poscar: "POSCAR_ref"
  system_type: "pom"           # "pom", "zeolite", or "slab"

site:
  tm_element: "Ni"
  site_index: null              # null = interactive, integer = specific atom

adsorbate:
  source: "H2O"                # built-in name or path to POSCAR/XYZ/CIF
  binding_mode: "end-on"       # "end-on" or "side-on"
  binding_atom: "O"

parameters:
  bond_distances: [1.8, 2.0, 2.2]
  tilt_angles: [0, 30, 45]
  rotation_angles: [0, 90, 180, 270]

validation:
  min_score: 70

output:
  directory: "./adsorption_configs"
```

## Output

```
adsorption_configs/
  config_0001/POSCAR
  config_0002/POSCAR
  ...
  summary.csv
```

Each POSCAR is a complete structure ready for VASP. The `summary.csv` contains parameters, quality scores, and rejection reasons for every attempted configuration.

## Built-in molecules

H, O, OH, CO, H2O, O2, CO2, NO, NH3, C2H4 (side-on)

Custom molecules can be provided as POSCAR, XYZ, or CIF files via `adsorbate.source`.

## Validation

Configurations are scored 0-100 based on 9 rules:

| Rule | Type | What it checks |
|------|------|----------------|
| Atom overlap | Hard reject | No adsorbate atom too close to framework atoms |
| Adsorbate integrity | Hard reject | Internal bond lengths/angles preserved |
| Not inside framework | Hard reject | Adsorbate COM not embedded in wall |
| Steric feasibility | Scored (20) | VdW clearance between adsorbate and framework |
| TM-adsorbate bond distance | Scored (25) | Within chemically reasonable range |
| Coordination number | Scored (15) | TM coordination doesn't exceed limit |
| TM-binding-next atom angle | Scored (15) | Binding geometry matches expected range |
| Channel blockage | Scored (10) | Zeolite-only: channel not blocked |
| TM-framework competition | Scored (15) | Adsorbate not competing with existing bonds |

Only configurations that pass all hard rules and score >= `min_score` are written.
