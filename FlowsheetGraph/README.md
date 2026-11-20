# FlowsheetGraph — parallel lines and flowsheet charts for PharmaPy 0.3.3

Works with `pharmapy-org` 0.3.3 as installed; nothing in PharmaPy is changed.

## Why 0.3.3 can't run parallel lines

| Piece | What goes wrong |
|---|---|
| `convert_str_flowsheet` / dict flowsheet | Only reads `A --> B --> C`; every dict key must be one unit. |
| `SimulationExec.SolveFlowsheet` | Connects each unit to the **next unit in run order** (`execution_names[ind + 1]`), not to its successors in the graph. `R01 -> (R02a, R02b) -> R03` gets wired R01 → R02a → R02b → R03. Run order of parallel units can also change between runs (`topological_bfs` pops from a set). |
| `Connection.PassPhases` | Sets `destination.Inlet`, which **replaces** every inlet. A unit where two lines merge keeps only the last one. Nothing splits the flow, so each branch would get the full upstream flow. |
| `ConcurrentVessel` | Already handles splits and merges itself (`StreamLink(..., fraction=)`, several links into one member). What it lacked was a way to build the links from the flowsheet definition. |

## What's here

`flowsheet_graph.py`

- **`FlowsheetGraph(flowsheet, splits=None)`** reads the definition. All of these are the same flowsheet:

  ```python
  {'R01': ('R02a', 'R02b'), ('R02a', 'R02b'): 'R03'}
  [{'R01': ('R02a', 'R02b')}, {('R02a', 'R02b'): 'R03'}]
  'R01 --> (R02a, R02b) --> R03'
  ```

  A tuple value is a split and a tuple key is a merge. Fractions go in the value (`{'R01': {'R02a': 0.3, 'R02b': 0.7}}`) or in `splits=` (`{'R01': {'R02a': 0.3}}`; the rest of the outlet is shared equally among the other branches). If a source's fractions add up to less than 1, the remainder leaves the flowsheet. A recycle raises an error, as in `SimulationExec`. Put a recycle inside a ConcurrentVessel instead.
- **`ParallelSimExec(path, flowsheet, splits=None)`** is a drop-in `SimulationExec`. You assign units and call `SolveFlowsheet` the same way. It connects every edge of the graph, gives a merging unit one inlet per upstream line, and scales each branch's `vol_flow`/`mass_flow`/`mole_flow` by its split. A plain chain gives results identical to `SimulationExec`. Extra argument: `SolveFlowsheet(..., keep_feeds=['R03'])` keeps R03's own feed as well as the upstream lines.
- **`concurrent_from_flowsheet(flowsheet, members, splits=None, placeholder_inlets=None, **cv_kwargs)`** builds a `ConcurrentVessel` from the same definition, with one `StreamLink` per edge.
- **`plot_flowsheet(sim_or_definition)`** (also `sim.plot_flowsheet()`) draws one box per unit with its class, coloured by operating mode. Each stream is an arrow labelled with its split and final flow. A ConcurrentVessel is drawn as a dashed frame around its members, and recycles are drawn as orange arcs.

## Limits

- Where lines merge, the receiving unit must be a refactored (MultiPhaseVessel-based) unit or a ConcurrentVessel: the old units hold one inlet. Merging into a Batch unit is refused.
- Only a continuous outlet can be split.
- A ConcurrentVessel takes every incoming line on its `entry` member and sends out only its `exit` member's outlet.

## Example and tests

From this folder, in the PharmaPy environment:

```bash
python example_parallel_lines.py --no-show
```

```bash
python -m pytest test_flowsheet_graph.py -q
```

The example solves `R01 --> (R02a, R02b) --> R03` (30/70 split) with `ParallelSimExec` and as one ConcurrentVessel. The two agree within 1e-8 on every outlet. Figures are in `results/`.
