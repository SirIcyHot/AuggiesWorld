# Flowsheet versions

Every script in this project that solves (or draws) the whole process, what
each one is for, and how they relate. When you add, retire or change a
version, update this file and add a line to the change log at the bottom.

## The process (all versions)

```
FEED_A --+              A_PUSH --+           WASH_H --+           FEED_H --+
         +--> CR01H --> CR02H ---+--> [F01H] ---> W02H(D02H -> [F02H]) ---+--> D01H --+
FEED_C --+                              |                  |                          |
                                        +--> filtrate 1    +--> wash filtrate H       |
                                                                                      |
   +----------------------------------------------------------------------------------+
   |
   +--> CR02R --> D01R --> [F01R] ---> W02R(D02R -> [F02R]) ---> D03R --> product
         ^         ^         |           ^         |             ^
CR02R makeup   FEED_B        |        WASH_B       |           FEED_L
                             +--> filtrate 2       +--> wash filtrate B
```

Eight vessels (CR01H, CR02H, D02H, D01H, CR02R, D01R, D02R, D03R) and four
steady-state filters (F01H, F02H, F01R, F02R). Every vessel comes from its
own unit folder's builder with that unit's settings; the flowsheet scripts
only connect them. The filters are never vessels: PharmaPy's
ConcurrentVessel only accepts MultiPhaseVessel members, so each filter is an
algebraic step inside the feed of the unit after it (checked against a real
`ContinuousFilter` at the final time).

## The two choices that define a version

| | **Coupled: one ConcurrentVessel** | **Sequential: one ConcurrentVessel per unit** |
|---|---|---|
| **Population-balance dissolution** (fed crystals keep their size distribution and dissolve or grow by kinetics) | `Flowsheet/Flowsheet.py` (main) | `Flowsheet_sequential/Flowsheet_sequential.py` |
| **DirectTransfer** (fed crystals go onto a "feed solid" that PharmaPy's `DirectTransfer.first_order` turns into liquid; dissolution rates zero) | `Flowsheet/Flowsheet_simplified.py` | `Flowsheet_sequential/Flowsheet_sequential_direct.py` |

- **Coupled:** all eight vessels are members of one ConcurrentVessel on one
  CVode integrator, so every unit advances through each time step together and
  reads the live outlet of the unit feeding it.
- **Sequential:** each vessel is wrapped in its own one-member
  ConcurrentVessel with its own integrator, and the units are solved one after
  another in process order, each over the full 16 h. A unit's feed is the
  upstream unit's recorded outlet, interpolated linearly between the saved
  points, then passed through the same filter and species steps as the coupled
  version. ConcurrentVessels cannot be nested, which is why the per-unit
  vessels are chained rather than placed inside one big one.

The plant has no recycle, so the coupled and sequential versions solve the
same model and should agree; any difference beyond ~0.1 % means something is
wrong.

## Versions

**At a glance: how each script uses ConcurrentVessel**

| Script | ConcurrentVessel setup |
|---|---|
| `Flowsheet/Flowsheet.py` | **Whole flowsheet:** all 8 vessels in one |
| `Flowsheet/Flowsheet_simplified.py` | **Whole flowsheet:** all 8 vessels in one |
| `Flowsheet/Flowsheet_dae.py` | **Whole flowsheet:** all 8 vessels in one (IDA solver) |
| `Flowsheet/Flowsheet_preconditioned_archive.py` | **Whole flowsheet:** all 8 vessels in one (archived) |
| `Flowsheet/flowsheet_visual.py` | **Whole flowsheet:** solves `Flowsheet.py`'s, then draws it |
| `Flowsheet_sequential/Flowsheet_sequential.py` | **Individual:** one per unit, 8 in sequence |
| `Flowsheet_sequential/Flowsheet_sequential_direct.py` | **Individual:** one per unit, 8 in sequence |
| `Flowsheet_medium/Flowsheet_medium.py` | **Individual:** one per unit, 8 in sequence |
| `Flowsheet_medium/Flowsheet_lite.py` | **Individual:** one per unit, 8 in sequence |
| `Flowsheet_medium/Flowsheet_lite_concurrent.py` | **Whole flowsheet:** the 8 lite vessels in one |
| `CR02R_D01R/CR02R_D01R.py` | **Two units coupled:** CR02R and D01R in one, CR02R's slurry feeding D01R live (a control study; see [CONTROL.md](CONTROL.md)) |
| `R02R/R02R_control.py`, `R02R/R02R_state_space.py`, `CR02R/CR02R_control.py`, `CR02R/CR02R_plant_id.py` | **Individual, one unit only:** a one-member ConcurrentVessel on fixed feeds, for control studies (not flowsheets; see [CONTROL.md](CONTROL.md)) |

### `Flowsheet/Flowsheet.py` — main version
- **Coupling:** one ConcurrentVessel. **Solids:** population balances.
- **Solver:** CVode, plain GMRES (krylov), `maxkrylov` 25; 1024 states.
- **Run time:** ~60 min for 16 h simulated.
- **Results:** `Flowsheet/results/` (`Flowsheet_profiles.png`,
  `Flowsheet_streams.png`).
- **Owner:** the other session ("Pharmapy flowsheet script"); it is the
  reference the other versions copy or import.

### `Flowsheet/Flowsheet_simplified.py` — DirectTransfer, coupled
- **Coupling:** one ConcurrentVessel. **Solids:** DirectTransfer.
- Every solid fed to CR02H, D02H, D01H, CR02R, D01R and D03R goes onto an
  extra feed-solid phase that `DirectTransfer.first_order(0.1 /s)` turns into
  liquid; each vessel keeps its own nucleation and growth, so anything beyond
  saturation crystallizes again. CR02R's F and D01R's J solids are dropped
  (they only ever dissolved).
- **Exception:** D02R (the W02R wash) takes the F01R cake's crystals straight
  into its own D and E population balances, with zero nucleation, growth and
  dissolution. Giving it D01R's kinetics made it extremely stiff (~15 min per
  simulated hour for D02R alone) for a ~0.001 % change.
- **Run time:** ~25-28 min; 965 states.
- **Results:** `Flowsheet/results_simplified/` (figures and
  `flowsheet_simplified_streams.json`).
- **Owner:** the other session.

### `Flowsheet_sequential/Flowsheet_sequential.py` — population balances, sequential
- **Coupling:** one ConcurrentVessel per unit. **Solids:** population balances.
- Imports everything except the coupling from `Flowsheet.py` (builders,
  grids, inlet templates, solver options, report, figures).
- **Run time:** 36.8 min for 16 h (CR02R and D01R ~12 min each).
- **Check (2026-11-17):** every stream within 0.03 % of `Flowsheet.py`;
  plant gap 0.51 %.
- **Results:** `Flowsheet_sequential/results/` (figures, `run_full.log`).

### `Flowsheet_sequential/Flowsheet_sequential_direct.py` — DirectTransfer, sequential
- **Coupling:** one ConcurrentVessel per unit. **Solids:** DirectTransfer.
- Imports the model from `Flowsheet_simplified.py`; only the coupling differs.
- **Run time:** 8.1 min for 16 h (953 states in all; no unit over 2.1 min).
- **Check (2026-11-18):** every stream within 0.006 % of
  `Flowsheet_simplified.py`; W02R balances; plant gap 0.52 %.
- **Results:** `Flowsheet_sequential/results_direct/` (figures,
  `run_full.log`).

### `Flowsheet_medium/Flowsheet_medium.py` — medium resolution, sequential
- **Coupling:** one ConcurrentVessel per unit (`Flowsheet_sequential.py`'s
  coupling). **Solids:** population balances in the crystallizers.
- `Flowsheet.py` with a set of simplifying assumptions (see its docstring):
  - Only the crystallizers (CR01H, CR02H, CR02R, D01R) make crystals;
    nucleation and growth are zero in D02H, D01H, D02R and D03R.
  - D01R dissolves J, and D03R dissolves D and E, by DirectTransfer from a
    feed solid, limited by each unit's solubility law. `--law linear`
    makes this zero order.
- **Run time:** 15.1 min for 16 h (1006 states).
- **Results:** `Flowsheet_medium/results_medium/` (figures,
  `flowsheet_medium_streams.json`, `run_full.log`); the `--law linear`
  results are in `results_medium_linear/`.
- **Owner:** the other session (added 2026-11-18).

### `Flowsheet_medium/Flowsheet_lite.py` — no size distributions, sequential
- **Coupling:** one ConcurrentVessel per unit. **Solids:** no population
  balances at all. Each vessel has one mass-based solid holding every
  species, exchanging with the liquid by a two-way DirectTransfer limited by
  the unit's solubility law.
- Imports `Flowsheet_medium.py`.
- **Run time:** 1.1 min for 16 h (195 states).
- **Results:** `Flowsheet_medium/results_lite/`, and `results_lite_linear/`
  for the linear law.
- **Owner:** the other session (added 2026-11-18).

### `Flowsheet_medium/Flowsheet_lite_concurrent.py` — lite, coupled
- **Coupling:** one ConcurrentVessel (whole flowsheet). **Solids:** the lite
  mass-based solid, as in `Flowsheet_lite.py`.
- The eight `Flowsheet_lite.py` vessels, linked live (`LiveOutlet`,
  `LiveLiteLink`) instead of through recorded outlets. With no recycle it
  solves the same equations as the sequential lite run; its report compares
  the two stream by stream.
- **Results:** `Flowsheet_medium/results_lite_concurrent/`.
- **Owner:** the other session (added 2026-11-20; its first full run was
  still going when this entry was written).

### `CR02R_D01R/CR02R_D01R.py` — two units coupled (control study)
- **Coupling:** one ConcurrentVessel with two members, CR02R and D01R.
  **Solids:** population balances.
- Not a flowsheet version: the two units from their own builders (CR02R on
  its standalone design feed, D01R charged with B), linked by `Flowsheet.py`'s
  `SlurryInlet` exactly as in the full flowsheet. A −2 K move on CR02R at
  10 h shows how a change on CR02R reaches D01R inside one solve.
- Imports `Flowsheet.py` for the link classes, then restores CR02R's
  standalone F grid; the product grid is the flowsheet's 50 um one (or the
  standalone 20 um with `--fine`).
- **Results:** `CR02R_D01R/results/`. Details in [CONTROL.md](CONTROL.md).

### `Flowsheet/Flowsheet_dae.py` — main version on the DAE solver
- **Coupling:** one ConcurrentVessel (whole flowsheet). **Solids:**
  population balances.
- A copy of `Flowsheet.py` that integrates with `AssimuloDAEBackend`
  (Sundials IDA, residual form) instead of CVode. Same model; with no recycle
  and no algebraic states IDA solves the same ODE. GMRES subspace is
  Sundials' default 5 (IDA has no `maxkrylov` in Assimulo); `LINEAR_SOLVER`
  picks krylov or dense.
- **Results:** `Flowsheet/results_dae/` (figures,
  `flowsheet_dae_streams.json`).
- A full copy, not an import: it does not follow changes to `Flowsheet.py`.

### `Flowsheet/Flowsheet_preconditioned_archive.py` — archived
- **Coupling:** one ConcurrentVessel (whole flowsheet).
- `Flowsheet.py` as of 2026-11-14 with the per-vessel block preconditioner
  (`BlockPreconditionedCVode`) switched on. It handled very fast dissolution
  (k_d = 5e5 um/s) where plain GMRES stalled, but at the current settings it
  was slower than plain GMRES.
- **Not maintained.** It predates CR02R (it still has R02R) and does not
  follow `Flowsheet.py`. Results would go to `results_preconditioned/`.

### `Flowsheet/flowsheet_visual.py` — the flowsheet map (not a model)
- **Coupling:** whatever `Flowsheet.py` uses, currently one ConcurrentVessel
  for the whole flowsheet.
- Solves `Flowsheet.build_flowsheet` (or reads its cache) and draws the
  process: units, each unit op's class, and every stream's phases and
  contents; also the stream table. Reads classes and couplings from the
  solved objects, so it follows the code.
- **Outputs:** `Flowsheet/results/flowsheet_map.{png,svg,pdf}`,
  `flowsheet_stream_table.{png,svg,pdf}` (SVG/PDF have editable text),
  `flowsheet_map_cache.json`.

## Which files import which

```
Flowsheet.py ──imported by──> Flowsheet_sequential.py
             ──imported by──> flowsheet_visual.py
             ──imported by──> CR02R_D01R/CR02R_D01R.py (link classes only)
             ──copied into──> Flowsheet_dae.py, Flowsheet_preconditioned_archive.py,
                              Flowsheet_simplified.py (copies don't follow later edits)

Flowsheet_simplified.py ──imported by──> Flowsheet_sequential_direct.py

Flowsheet_sequential.py ──imported by──> Flowsheet_medium.py ──imported by──> Flowsheet_lite.py
                                                                    ──imported by──> Flowsheet_lite_concurrent.py
```

**Watch out:**
- The imported model is read once, when a run starts. If
  `Flowsheet.py` or `Flowsheet_simplified.py` is edited during a run, the run
  keeps the version it loaded. (This caused a false 1.3 % D02R mass gain on
  2026-11-17: the sequential run loaded an older `Flowsheet_simplified.py`
  with D01R's kinetics on D02R, and was compared against results from the
  newer one.)
- Never import `Flowsheet.py` and `Flowsheet_simplified.py` into the same
  Python process: both change settings on the shared unit modules (size grids,
  dissolution rates, solid species) when imported.
- A change to a unit folder (e.g. `D01R/D01R.py`) affects every version.

## Running

From the activated environment (otherwise LAPACK fails to load), in the
script's own folder:

```bash
conda activate PharmaPy
```

```bash
python Flowsheet.py --no-show
```

Every model script takes the same options: `--no-show` (save figures without
opening windows), `--runtime 600` (a short wiring check, in seconds),
`--points N` (saved time points, default 481). A full 16 h run stalls if the
PC sleeps, so keep it awake.

## Change log

- **2026-11-14** — Block preconditioner removed from `Flowsheet.py`; kept as
  `Flowsheet_preconditioned_archive.py`.
- **2026-11-16** — `flowsheet_visual.py` added (map and stream table,
  editable SVG/PDF export).
- **2026-11-16/17** — R02R replaced by CR02R in the main flowsheet (other
  session; exact date not recorded).
- **2026-11-17** — `Flowsheet_sequential.py` added (one ConcurrentVessel per
  unit, population balances); matches `Flowsheet.py` within 0.03 %.
- **2026-11-17** — `Flowsheet_simplified.py`: D02R back to zero kinetics
  (other session).
- **2026-11-18** — `Flowsheet_sequential_direct.py` added (one ConcurrentVessel
  per unit, DirectTransfer); matches `Flowsheet_simplified.py` within 0.006 %.
- **2026-11-18** — `Flowsheet_medium.py` and `Flowsheet_lite.py` added (other
  session; one ConcurrentVessel per unit).
- **2026-11-19** — Single-unit control scripts added (`R02R_control.py`,
  `CR02R_control.py`, `CR02R_plant_id.py`); see `CONTROL.md`.
- **2026-11-20** — `Flowsheet_lite_concurrent.py` added (other session; the
  lite flowsheet in one ConcurrentVessel).
- **2026-11-20** — `CR02R_D01R/CR02R_D01R.py` (CR02R and D01R in one
  ConcurrentVessel) and `R02R/R02R_state_space.py` (+ notebook) added; see
  `CONTROL.md`.
