"""Semibatch reactor with selectable kinetics (R02R, first or second order),
on PharmaPy's refactored SemiBatchReactor.

The setup is::

    FEED_A ------+
    FEED_G ------+--> SBR (4 g I made up to 17 mL with H), 70 C isothermal
    FEED_C_in_H -+

``SBR`` is ``PharmaPy.Reactors_Refactored.SemiBatchReactor``. It is charged
with 4 g of I made up to 17 mL with the solvent H; the H mass is solved so
that PharmaPy's own liquid density gives exactly ``CHARGE_VOLUME``. Three
streams are fed in separately, each at its own constant mass flow for
``DOSE_TIME`` (37 min), so each is added linearly in time. After the dose
the vessel is held for ``HOLD_TIME``. Nothing leaves the vessel, and the
vessel is held at 70 C (``ISOTHERMAL``).

The reactions and heats of reaction are imported from
``Project work/R02R_onemethod/R02R.py`` rather than copied::

    A + C + G + I --> D + H
    A + C + G + I --> E + H
    A + C + G + I --> J + H

``KINETICS_MODEL`` (or ``--kinetics``) picks the rate law. Every option is
a power law::

    r_i = k_i * exp(-Ea_i / R * (1/T - 1/T_ref)) * prod_j c_j**a_ij

    R02R          R02R_onemethod's reactions, orders, k and Ea
    first_order   R02R's reactions, first order in C:   r_i = k_i [C]
    second_order  a two-step mechanism through the intermediate F, every
                  step second order overall (SECOND_ORDER_REACTIONS):
                      A + C --> F            r_1 = k_1 [A]^1.5 [C]^0.5
                      F + G + I --> D/E/J + H   r = k [F][G]

with c in mol/L. The first- and second-order k and Ea are starting guesses
for parameter estimation, not fitted values; the second-order k_1 was
picked by guess and check so C runs out ~12 min into the dose.
``RATE_SCALE`` multiplies every k. PharmaPy's ``ParamEstim_Refactored``
fits one block per kinetics object, each laid out as::

    that block's k_i, its Ea_i, then one order per reactant per reaction

(one block for R02R and first order; two for second order, F formation
and F consumption). ``parameter_names`` labels the vector and
``estimation_flags`` picks which entries are fitted (by default k and Ea,
with the orders held at the chosen model).
Build each experiment's unit with ``build_reactor(integrator=
AssimuloBackend())``: the estimation needs a backend that computes
sensitivities.

Charge and feeds:

    charge        4 g I (0.0500 mol), H to 17 mL
    FEED_A        0.444 mol A (18.53 mL)
    FEED_G        0.257 mol G (24.34 mL)
    FEED_C_in_H   0.038 mol C with 0.197 mol H

and the final volume should be about ``EXPECTED_FINAL_VOLUME`` (76.21 mL).
Feeds are specified in moles; PharmaPy computes their volumes from the
database densities, and the run prints both against the quoted volumes.

Every reaction consumes one each of A, C, G and I, so C (0.038 mol, against
0.050 mol of I) caps the total product (D + E + J) at 0.038 mol.

``ISOTHERMAL = False`` switches back to R02R's jacket (``COOLANT_*``); then
the heat of reaction (-622 kJ/mol) makes temperature a solved state.

``semibatch_crystallizer.py`` runs the same charge, dose and kinetics in a
SemiBatchCrystallizer, where D, E and J can also crystallize.

Run it with::

    python semibatch_reactor.py               # solve, save figures, show them
    python semibatch_reactor.py --no-show     # solve and save figures only
    python semibatch_reactor.py --kinetics first_order   # pick the rate law

Figures are written to ``results/`` next to this file (with a
``_first_order`` or ``_second_order`` suffix for those models):

    SBR_profiles.png      species moles, temperature, and vessel mass/volume
    SBR_normalized.png    species (no I) over their peaks, and over C fed
    SBR_settings.png      the settings and initial conditions used
    SBR_mass_balance.png  charged, dosed, reacted and final moles per species

"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from PharmaPy.DataClasses import IntraPhaseProcess, PhaseRef
from PharmaPy.IntegratorBackends import ScipyBackend
from PharmaPy.Kinetics import RxnKinetics
from PharmaPy.Mechanisms import ReactionMechanism
from PharmaPy.Phases_Refactored import LiquidPhase
from PharmaPy.ProcessControl import DynamicInput
from PharmaPy.Reactors_Refactored import SemiBatchReactor
from PharmaPy.Streams_Refactored import LiquidStream
from PharmaPy.ThermoModule import ParseDatabase

# ---------------------------------------------------------------- paths
HERE = Path(__file__).resolve().parent
# Kinetics, helpers and database come from R02R_onemethod (its module is
# also named R02R).
R02R_DIR = HERE.parent / "Project work" / "R02R_onemethod"
RESULTS_DIR = HERE / "results"

sys.path.insert(0, str(R02R_DIR))

import R02R  # noqa: E402

DB_PATH = R02R.DEFAULT_DATABASE_PATH

# ---------------------------------------------------------------- settings
# Initial charge: these solutes [kg], made up to CHARGE_VOLUME with H.
CHARGE_SOLUTES = {"I": 4.0e-3}
CHARGE_VOLUME = 17.0e-6  # m**3
# Feed streams, each fed separately at a constant mass flow over
# DOSE_TIME [mol]. The quoted volumes are for comparison only.
DOSE_STREAMS = {
    "FEED_A": {"A": 0.444},
    "FEED_G": {"G": 0.257},
    "FEED_C_in_H": {"C": 0.038, "H": 0.197},
}
QUOTED_STREAM_VOLUMES = {"FEED_A": 18.53e-6, "FEED_G": 24.34e-6}  # m**3
EXPECTED_FINAL_VOLUME = 76.21e-6  # m**3
DOSE_TIME = 37 * 60  # s
HOLD_TIME = 10 * 60  # s, after the dose ends
RUNTIME = DOSE_TIME + HOLD_TIME

INITIAL_TEMPERATURE = 70 + 273.15  # K
# The feeds enter at the vessel temperature; with ISOTHERMAL this does not
# matter, with the jacket it sets the feeds' sensible heat.
DOSE_TEMPERATURE = INITIAL_TEMPERATURE  # K

ISOTHERMAL = True  # False uses the jacket below instead
# m, jacket area = 4 V / D. Lab scale: 17 mL charged, ~76 mL at the end.
VESSEL_DIAMETER = 0.05
VESSEL_H_CONV = R02R.REACTOR_H_CONV  # W/m**2/K
COOLANT_MASS_FLOW = R02R.COOLANT_MASS_FLOW  # kg/s
COOLANT_TEMPERATURE = R02R.COOLANT_TEMPERATURE  # K
COOLANT_H_CONV = R02R.COOLANT_H_CONV  # W/m**2/K

# ---------------------------------------------------------------- kinetics
# "R02R", "first_order" or "second_order"; see the module docstring.
KINETICS_MODEL = "R02R"
# Multiplies every rate constant of the selected model, here only (R02R's
# own values are untouched). 0.1 = reactions 10x slower; 1.0 = as given.
RATE_SCALE = 0.1
# Second-order model: rate constant of A + C --> F [(L/mol)/s, before
# RATE_SCALE], found by guess and check (see SECOND_ORDER_REACTIONS): at
# RATE_SCALE = 0.1, 0.005 -> C runs out at 16.9 min, 0.006 -> 14.7,
# 0.007 -> 13.1, 0.008 -> 11.8, 0.01 -> 10.1.
F_FORMATION = 8.0e-3

KINETICS_TEMP_REF = R02R.KINETICS_TEMP_REF  # K, k below are quoted here

# The second-order option's mechanism (R02R's reactions are not changed):
# A and C first form the intermediate F, which then reacts with G and I to
# the irreversible products. Every step is second order overall:
#     A + C --> F           r_1 = k_1 [A]^1.5 [C]^0.5
#     F + G + I --> D + H   r_2 = k_2 [F][G]
#     F + G + I --> E + H   r_3 = k_3 [F][G]
#     F + G + I --> J + H   r_4 = k_4 [F][G]
# I keeps its R02R stoichiometry but has order 0. Overall, as in R02R:
#     A + C + G + I --> D/E/J + H
# The half order in C is what lets C run out while it is still being fed:
# with first order in C, the C held settles to feed / (k [A]) and only
# levels off as [A] plateaus. When C runs out is not imposed; it follows
# from k_1, which was picked by guess and check (C_DEPLETION_FRACTION
# defines "runs out" for the printed check).
SECOND_ORDER_REACTIONS = [
    "A + C --> F",
    "F + G + I --> D + H",
    "F + G + I --> E + H",
    "F + G + I --> J + H",
]

# Each model: its reactions and heats of reaction, orders per reaction
# keyed by reactant (a reactant left out gets 0), k at KINETICS_TEMP_REF
# (before RATE_SCALE) and Ea. The first- and second-order k and Ea are
# starting guesses for estimation, not fitted values.
KINETIC_MODELS = {
    "R02R": {
        "reactions": list(R02R.REACTIONS),
        "delta_h": np.asarray(R02R.HEAT_OF_REACTION, dtype=float),
        "orders": [dict(orders) for orders in R02R.REACTION_ORDERS],
        "k": np.asarray(R02R.RATE_CONSTANTS, dtype=float),
        "ea": np.asarray(R02R.ACTIVATION_ENERGIES, dtype=float),
        "k_units": "(L/mol)^(n-1)/s",
    },
    "first_order": {
        "reactions": list(R02R.REACTIONS),
        "delta_h": np.asarray(R02R.HEAT_OF_REACTION, dtype=float),
        "orders": [{"C": 1.0}] * 3,
        "k": np.array([1.0e-2, 1.0e-2, 1.0e-2]),
        "ea": np.asarray(R02R.ACTIVATION_ENERGIES, dtype=float),
        "k_units": "1/s",
    },
    "second_order": {
        "reactions": SECOND_ORDER_REACTIONS,
        # R02R's -622 kJ/mol per path, split evenly over the two steps.
        "delta_h": np.array([-3.11e5, -3.11e5, -3.11e5, -3.11e5]),
        "orders": [{"A": 1.5, "C": 0.5}] + [{"F": 1.0, "G": 1.0}] * 3,
        # k_1 (F formation) is F_FORMATION; k_2..k_4 split F 74/20/6 into
        # D/E/J and are fast enough that F stays small.
        "k": np.array([F_FORMATION, 0.074, 0.020, 0.006]),
        "ea": np.array([5.0e4, 5.0e4, 5.0e4, 5.0e4]),
        "k_units": "L/mol/s",
        # Separate kinetics objects for F formation and F consumption.
        # PharmaPy's RxnKinetics scales every reaction it holds by one
        # factor when any species would go negative; at half order in C,
        # the formation rate stays nonzero at [C] = 0, so in one object it
        # froze F --> D/E/J whenever C ran out.
        "groups": [[0], [1, 2, 3]],
    },
}

# Limiting reagent the normalized plot's yields are taken on.
NORMALIZE_BASIS = "C"
# Species on the normalized plot's left panel (I left out; F is added
# when the model makes it).
NORMALIZED_SPECIES = ("A", "G", "C", "D", "E", "J")
# C counts as run out once it falls below this share of its peak.
C_DEPLETION_FRACTION = 0.05

# Species plotted and tabulated, in order: dosed, charged, products, and
# the intermediate F when the model makes it (report_species).
REPORT_SPECIES = ("A", "G", "C", "I", "D", "E", "J")
# R02R's seven categorical colors plus one for F, fixed per species so a
# species keeps its color across figures and models.
SERIES_COLORS = tuple(R02R.SERIES_COLORS) + ("#a3672c",)
SPECIES_COLORS = dict(zip(REPORT_SPECIES + ("F",), SERIES_COLORS))
TEXT_PRIMARY = R02R.TEXT_PRIMARY
TEXT_MUTED = R02R.TEXT_MUTED
GRID_COLOR = "#e5e4e0"


# ---------------------------------------------------------------- builders
def species_names() -> tuple[str, ...]:
    return R02R.database_species(DB_PATH)


def solvent_name() -> str:
    return R02R.database_solvent(species_names())


def molecular_weights() -> dict[str, float]:
    """Database molecular weights [g/mol], by name."""
    parsed = ParseDatabase(str(DB_PATH), to_arrays=False)
    return dict(zip(parsed["name_species"], parsed["mw"]))


def liquid_densities() -> dict[str, float]:
    """Database pure-liquid densities [kg/m**3], by name."""
    parsed = ParseDatabase(str(DB_PATH), to_arrays=False)
    return dict(zip(parsed["name_species"], parsed["rho_liq"]))


def moles_to_kg(moles: dict[str, float]) -> dict[str, float]:
    mw = molecular_weights()
    return {name: n * mw[name] * 1e-3 for name, n in moles.items()}


def mass_fractions_of(moles: dict[str, float]) -> dict[str, float]:
    """Named mass fractions of a mixture given in moles (solvent included)."""
    masses = moles_to_kg(moles)
    total = sum(masses.values())
    return {name: m / total for name, m in masses.items()}


def _charge_phase(solvent_kg: float, phase_cls=LiquidPhase, **kwargs):
    masses = dict(CHARGE_SOLUTES)
    masses[solvent_name()] = solvent_kg
    total = sum(masses.values())
    return phase_cls(
        str(DB_PATH),
        temp=INITIAL_TEMPERATURE,
        mass_frac=R02R.named_mass_fraction(
            {name: m / total for name, m in masses.items()},
            species_names(), solvent_name()),
        mass=total,
        name_solv=solvent_name(),
        **kwargs,
    )


def charge_solvent_mass() -> float:
    """H [kg] that makes the charge up to CHARGE_VOLUME.

    Solved against PharmaPy's own liquid density, so the vessel starts at
    exactly CHARGE_VOLUME whatever mixing rule PharmaPy applies.
    """
    rho = liquid_densities()
    # Start from ideal (volume-additive) mixing, then correct.
    solvent = (CHARGE_VOLUME - sum(m / rho[name]
                                   for name, m in CHARGE_SOLUTES.items())
               ) * rho[solvent_name()]
    for _ in range(20):
        error = CHARGE_VOLUME - float(_charge_phase(solvent).vol)
        if abs(error) < 1e-12:
            break
        solvent += error * rho[solvent_name()]
    return solvent


def charge_moles() -> dict[str, float]:
    """The charge in moles, H included [mol]."""
    mw = molecular_weights()
    masses = dict(CHARGE_SOLUTES)
    masses[solvent_name()] = charge_solvent_mass()
    return {name: m / (mw[name] * 1e-3) for name, m in masses.items()}


def charge_mass() -> float:
    """Initial charge [kg]."""
    return sum(CHARGE_SOLUTES.values()) + charge_solvent_mass()


def dose_mass_flows() -> dict[str, float]:
    """Mass flow of each feed stream while dosing [kg/s]."""
    return {label: sum(moles_to_kg(moles).values()) / DOSE_TIME
            for label, moles in DOSE_STREAMS.items()}


def dose_mass_flow() -> float:
    """Total feed mass flow while dosing [kg/s]."""
    return sum(dose_mass_flows().values())


def dose_profile(flow: float):
    """A mass-flow schedule [kg/s]: ``flow`` until DOSE_TIME, then off."""
    def profile(time: float) -> float:
        return flow if time < DOSE_TIME else 0.0
    return profile


def dosed_moles(time) -> dict[str, np.ndarray]:
    """Moles of each species fed by ``time``, all streams summed [mol]."""
    fraction = np.clip(np.asarray(time, dtype=float) / DOSE_TIME, 0.0, 1.0)
    totals: dict[str, float] = {}
    for moles in DOSE_STREAMS.values():
        for name, n in moles.items():
            totals[name] = totals.get(name, 0.0) + n
    return {name: n * fraction for name, n in totals.items()}


def make_charge(phase_cls=LiquidPhase, **kwargs):
    """The initial charge as a liquid phase."""
    return _charge_phase(charge_solvent_mass(), phase_cls, **kwargs)


def make_dose() -> list[LiquidStream]:
    """One stream per DOSE_STREAMS entry, each on its own schedule."""
    names = species_names()
    streams = []
    for label, flow in dose_mass_flows().items():
        stream = LiquidStream(
            str(DB_PATH),
            temp=DOSE_TEMPERATURE,
            mass_frac=R02R.named_mass_fraction(
                mass_fractions_of(DOSE_STREAMS[label]), names,
                solvent_name()),
            mass_flow=flow,
            name_solv=solvent_name(),
        )
        schedule = DynamicInput()
        schedule.add_variable("mass_flow", dose_profile(flow))
        stream.DynamicInlet = schedule
        streams.append(stream)
    return streams


def stream_volumes() -> dict[str, float]:
    """Volume of each feed stream over the dose, by PharmaPy's density
    [m**3]."""
    return {label: float(stream.vol_flow) * DOSE_TIME
            for label, stream in zip(DOSE_STREAMS, make_dose())}


def reactants(reaction: str) -> list[str]:
    return [term.strip().split()[-1]
            for term in reaction.split("-->")[0].split("+")]


def kinetic_model(model: str | None = None) -> dict:
    model = KINETICS_MODEL if model is None else model
    try:
        return KINETIC_MODELS[model]
    except KeyError:
        raise KeyError(f"Unknown kinetics model {model!r}; choose from "
                       + ", ".join(KINETIC_MODELS)) from None


def kinetics_groups(model: str | None = None) -> list[list[int]]:
    """Reaction indices held by each RxnKinetics object; one object holding
    every reaction unless the model sets ``groups``."""
    spec = kinetic_model(model)
    return spec.get("groups", [list(range(len(spec["reactions"])))])


def order_matrix(model: str | None = None, group=None) -> list[list[float]]:
    """The model's orders in PharmaPy's ``params_f`` layout.

    One row per reaction (those in ``group``, default all), one order per
    reactant of that reaction in database order (as
    R02R.reaction_order_matrix does). Reactants the model gives no order
    get 0.
    """
    names = species_names()
    spec = kinetic_model(model)
    if group is None:
        group = range(len(spec["reactions"]))
    matrix = []
    for i in group:
        reaction, orders = spec["reactions"][i], spec["orders"][i]
        unknown = set(orders) - set(reactants(reaction))
        if unknown:
            raise KeyError(f"Orders given for non-reactants of '{reaction}': "
                           + ", ".join(sorted(unknown)))
        ordered = sorted(reactants(reaction), key=names.index)
        matrix.append([float(orders.get(name, 0.0)) for name in ordered])
    return matrix


def build_kinetics(model: str | None = None) -> list[RxnKinetics]:
    """The selected rate law, one RxnKinetics per ``kinetics_groups``."""
    spec = kinetic_model(model)
    return [
        RxnKinetics(
            path=str(DB_PATH),
            rxn_list=[spec["reactions"][i] for i in group],
            k_params=spec["k"][group] * RATE_SCALE,
            ea_params=spec["ea"][group],
            params_f=order_matrix(model, group),
            temp_ref=KINETICS_TEMP_REF,
            delta_hrxn=spec["delta_h"][group],
        )
        for group in kinetics_groups(model)
    ]


def reaction_processes(liquid, model: str | None = None
                       ) -> list[IntraPhaseProcess]:
    """One liquid-phase reaction process per kinetics object."""
    return [
        IntraPhaseProcess(phaseref=PhaseRef("liquid", 0),
                          mechanism=ReactionMechanism(kinetics, liquid))
        for kinetics in build_kinetics(model)
    ]


def report_species(model: str | None = None) -> tuple[str, ...]:
    """REPORT_SPECIES, plus the intermediate F if the model makes it."""
    made = {term.strip().split()[-1]
            for reaction in kinetic_model(model)["reactions"]
            for term in reaction.split("-->")[1].split("+")}
    return REPORT_SPECIES + (("F",) if "F" in made else ())


def parameter_names(model: str | None = None) -> list[str]:
    """Readable names for the estimated parameter vector, in its order.

    One block per kinetics object (``kinetics_groups``), each laid out as
    PharmaPy concatenates it: that block's ``k_i``, its ``Ea_i``, then
    ``a_<reaction>_<species>``. Reaction numbers are the model's.
    """
    names = species_names()
    rxns = kinetic_model(model)["reactions"]
    labels = []
    for group in kinetics_groups(model):
        labels += [f"k_{i + 1}" for i in group]
        labels += [f"Ea_{i + 1}" for i in group]
        for i in group:
            labels += [f"a_{i + 1}_{name}" for name in
                       sorted(reactants(rxns[i]), key=names.index)]
    return labels


def estimation_flags(fit=("k", "Ea"), model: str | None = None) -> list[bool]:
    """``optimize_flags`` for ParameterEstimation: True for each parameter
    whose kind is in ``fit`` ("k", "Ea", "orders")."""
    kinds = {"k": "k_", "Ea": "Ea_", "orders": "a_"}
    prefixes = tuple(kinds[kind] for kind in fit)
    return [name.startswith(prefixes) for name in parameter_names(model)]


def c_depletion_time(reactor) -> float:
    """When C runs out [s]: the first time after its peak that the C held
    drops below C_DEPLETION_FRACTION of that peak (nan if it never does)."""
    time = np.asarray(reactor.result.time, dtype=float)
    n_c = mole_history(reactor)["C"]
    peak = int(np.argmax(n_c))
    below = np.nonzero(n_c[peak:] < C_DEPLETION_FRACTION * n_c[peak])[0]
    if below.size == 0:
        return float("nan")
    i = peak + int(below[0])
    # Interpolate the crossing between the two bracketing solver points.
    level = C_DEPLETION_FRACTION * n_c[peak]
    return float(np.interp(level, [n_c[i], n_c[i - 1]],
                           [time[i], time[i - 1]]))


def parameter_table(reactor) -> list[tuple[str, float]]:
    """(name, value) of every parameter as the estimation sees it."""
    from PharmaPy.Parameters import Parameters
    values = Parameters.kinetics(reactor).values()
    return list(zip(parameter_names(), np.asarray(values, dtype=float)))


def vessel_kwargs() -> dict:
    """Thermal settings shared with the semibatch crystallizer."""
    if ISOTHERMAL:
        return {"isothermal": True, "h_conv": 0.0, "diam": VESSEL_DIAMETER}
    return {"h_conv": VESSEL_H_CONV, "diam": VESSEL_DIAMETER}


def make_utility():
    """Jacket coolant, or None when isothermal."""
    if ISOTHERMAL:
        return None
    from PharmaPy.Utilities import CoolingWater
    return CoolingWater(mass_flow=COOLANT_MASS_FLOW,
                        temp_in=COOLANT_TEMPERATURE, h_conv=COOLANT_H_CONV)


def build_reactor(integrator=None, model: str | None = None
                  ) -> SemiBatchReactor:
    """Build the reactor. Pass an AssimuloBackend for parameter estimation;
    the default ScipyBackend computes no sensitivities."""
    reactor = SemiBatchReactor(integrator=integrator or ScipyBackend(),
                               **vessel_kwargs())
    reactor.Phases = make_charge()
    reactor.intraphase_processes = reaction_processes(
        reactor.Phases.Liquids[0], model)
    reactor.nomenclature(overwrite=True)
    utility = make_utility()
    if utility is not None:
        reactor.Utility = utility
    reactor.Inlet = make_dose()
    return reactor


def run_reactor(runtime: float = RUNTIME, model: str | None = None
                ) -> SemiBatchReactor:
    """Solve the semibatch reactor and return it with ``result`` populated."""
    reactor = build_reactor(model=model)
    reactor.solve_unit(runtime=runtime, verbose=False)
    return reactor


# ---------------------------------------------------------------- results
def mole_history(reactor) -> dict[str, np.ndarray]:
    """Moles of every species in the liquid over time [mol]."""
    mw = molecular_weights()
    mass_j = np.asarray(reactor.result.mass_j_liquid0, dtype=float)
    return {name: mass_j[:, j] / (mw[name] * 1e-3)
            for j, name in enumerate(reactor.name_species)}


def mole_balance(reactor) -> SimpleNamespace:
    """Charged, dosed, reacted and final moles per species at the end [mol].

    ``reacted`` is final - charged - dosed (negative = consumed). Every
    reaction path consumes one C and makes one of D, E or J (via F in the
    second-order model), so ``closure`` compares the C consumed with the
    D + E + J + F formed; it should be ~0.
    """
    time = float(np.asarray(reactor.result.time)[-1])
    final = {name: float(n[-1]) for name, n in mole_history(reactor).items()}
    charge = charge_moles()
    charged = {name: charge.get(name, 0.0) for name in final}
    dosed_now = dosed_moles(time)
    dosed = {name: float(dosed_now.get(name, 0.0)) for name in final}
    reacted = {name: final[name] - charged[name] - dosed[name]
               for name in final}
    # F (the second-order intermediate) has taken its C but is not yet
    # product; it is 0 in the other models.
    products = reacted["D"] + reacted["E"] + reacted["J"] + reacted["F"]
    return SimpleNamespace(time=time, charged=charged, dosed=dosed,
                           reacted=reacted, final=final,
                           closure=-reacted["C"] - products)


# ---------------------------------------------------------------- plotting
def style(axis):
    axis.grid(True, color=GRID_COLOR, lw=0.8)
    axis.spines[["top", "right"]].set_visible(False)


def mark_dose_end(axis):
    axis.axvline(DOSE_TIME / 60.0, color=TEXT_MUTED, lw=1, ls="--")
    axis.annotate("dose ends", (DOSE_TIME / 60.0, 1.0),
                  xycoords=("data", "axes fraction"), xytext=(4, -4),
                  textcoords="offset points", va="top", color=TEXT_MUTED,
                  fontsize=9)


def plot_profiles(reactor, title=None):
    """Species moles, temperature and vessel mass/volume vs time."""
    import matplotlib.pyplot as plt

    if title is None:
        title = f"Semibatch reactor ({KINETICS_MODEL} kinetics)"

    result = reactor.result
    time_min = np.asarray(result.time) / 60.0
    moles = mole_history(reactor)
    temp = np.ravel(result.global_temp)

    fig, (ax_n, ax_t, ax_m) = plt.subplots(1, 3, figsize=(15, 4.5))
    for name in report_species():
        ax_n.plot(time_min, moles[name], color=SPECIES_COLORS[name], lw=2,
                  label=name)
    ax_n.set_xlabel("Time [min]")
    ax_n.set_ylabel("Amount [mol]")
    ax_n.set_title("Species in the vessel")
    ax_n.legend(frameon=False, ncol=2)

    ax_t.plot(time_min, temp, color=TEXT_PRIMARY, lw=2)
    if not ISOTHERMAL:
        ax_t.axhline(COOLANT_TEMPERATURE, color=TEXT_MUTED, lw=1, ls=":")
    ax_t.set_xlabel("Time [min]")
    ax_t.set_ylabel("$T$ [K]")
    ax_t.set_title(f"Temperature (peak {temp.max():.1f} K)")

    ax_m.plot(time_min, np.ravel(result.vessel_vol) * 1e6,
              color=SERIES_COLORS[0], lw=2, label="Volume [mL]")
    ax_m.plot(time_min, np.ravel(result.Total_m_in_vessel) * 1e3,
              color=SERIES_COLORS[1], lw=2, label="Mass [g]")
    ax_m.set_xlabel("Time [min]")
    ax_m.set_title("Vessel contents")
    ax_m.legend(frameon=False, loc="lower right")

    for axis in (ax_n, ax_t, ax_m):
        style(axis)
        mark_dose_end(axis)

    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_normalized(reactor, moles=None, title=None):
    """Normalized species profiles, two ways.

    Left: each species over its own peak over the run (0-1), to compare
    shapes. Right: C and the products over the total moles of
    NORMALIZE_BASIS fed, i.e. conversion and yield on the limiting
    reagent. ``moles`` defaults to ``mole_history`` (the crystallizer
    passes liquid + crystal totals).
    """
    import matplotlib.pyplot as plt

    if moles is None:
        moles = mole_history(reactor)
    # A consumed species can dip a hair below zero (solver overshoot, ~1e-5
    # mol); over a small peak that reads as a real negative, so floor it.
    moles = {name: np.maximum(n, 0.0) for name, n in moles.items()}
    if title is None:
        title = f"Semibatch reactor ({KINETICS_MODEL} kinetics), normalized"
    time_min = np.asarray(reactor.result.time) / 60.0
    basis = float(dosed_moles(DOSE_TIME)[NORMALIZE_BASIS])
    colors = SPECIES_COLORS
    extra = ("F",) if "F" in report_species() else ()

    fig, (ax_peak, ax_basis) = plt.subplots(1, 2, figsize=(12, 4.5))
    for name in NORMALIZED_SPECIES + extra:
        peak = np.max(np.abs(moles[name]))
        if peak > 0:
            ax_peak.plot(time_min, moles[name] / peak, color=colors[name],
                         lw=2, label=name)
    ax_peak.set_ylabel("$n_i / n_{i,\\max}$")
    ax_peak.set_title("Each species over its peak")

    for name in (NORMALIZE_BASIS,) + extra + ("D", "E", "J"):
        ax_basis.plot(time_min, moles[name] / basis, color=colors[name],
                      lw=2, label=name)
    products = sum(moles[name] for name in ("D", "E", "J"))
    ax_basis.plot(time_min, products / basis, color=TEXT_PRIMARY, lw=1.5,
                  ls="--", label="D + E + J")
    ax_basis.set_ylabel(f"$n_i / n_{{{NORMALIZE_BASIS},\\mathrm{{fed}}}}$")
    ax_basis.set_title(f"Over total {NORMALIZE_BASIS} fed ({basis:g} mol)")

    for axis in (ax_peak, ax_basis):
        axis.set_xlabel("Time [min]")
        axis.legend(frameon=False, ncol=2)
        style(axis)
        mark_dose_end(axis)

    fig.suptitle(title)
    fig.tight_layout()
    return fig


def table_figure(rows, columns, title, col_widths=None, note=None):
    """A matplotlib table in the Project work style."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 0.3 * len(rows) + 1.0))
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=columns, colWidths=col_widths,
                     cellLoc="left", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.4)
    for (row, _), cell in table.get_celld().items():
        cell.set_edgecolor(GRID_COLOR)
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#f0efec")
    ax.set_title(title, pad=12)
    if note:
        fig.text(0.5, 0.02, note, ha="center", color=TEXT_MUTED, fontsize=9)
    fig.tight_layout(rect=(0, 0.04 if note else 0, 1, 1))
    return fig


def settings_rows(reactor) -> list[tuple[str, str, str]]:
    """(group, setting, value) rows describing this run."""
    mw = molecular_weights()

    def amounts(moles):
        return ", ".join(f"{k} {v:g} mol ({v * mw[k]:.2f} g)"
                         for k, v in moles.items())

    temp = np.ravel(reactor.result.global_temp)
    spec = kinetic_model()
    volumes = stream_volumes()
    rows = [
        ("Charge", "Contents", amounts(charge_moles())),
        ("Charge", "Mass / volume",
         f"{charge_mass() * 1e3:.2f} g, "
         f"{float(np.ravel(reactor.result.vessel_vol)[0]) * 1e6:.1f} mL"),
        ("Charge", "Temperature", f"{INITIAL_TEMPERATURE:g} K"),
        *[("Feeds", label,
           f"{amounts(DOSE_STREAMS[label])}; {flow * 1e6:.3f} mg/s, "
           f"{volumes[label] * 1e6:.2f} mL")
          for label, flow in dose_mass_flows().items()],
        ("Feeds", "Profile",
         f"each linear over {DOSE_TIME:g} s "
         f"({DOSE_TIME // 60:.0f} min {DOSE_TIME % 60:.0f} s)"),
        ("Feeds", "Temperature", f"{DOSE_TEMPERATURE:g} K"),
        *[(f"Kinetics ({KINETICS_MODEL})", f"Rxn {i + 1}: {rxn}",
           "orders " + ", ".join(f"{k} {v:g}" for k, v in orders.items())
           + f"; k {k_i:g} {spec['k_units']}; Ea {ea:.1f} J/mol;"
           f" ΔH {dh:g} J/mol")
          for i, (rxn, orders, k_i, ea, dh) in enumerate(zip(
              spec["reactions"], spec["orders"], spec["k"] * RATE_SCALE,
              spec["ea"], spec["delta_h"]))],
        (f"Kinetics ({KINETICS_MODEL})", "Rate scale",
         f"{RATE_SCALE:g} × the model's k (k shown already scaled)"),
        (f"Kinetics ({KINETICS_MODEL})", "T_ref",
         f"{KINETICS_TEMP_REF:g} K"),
    ]
    if ISOTHERMAL:
        rows.append(("Thermal", "Mode", "isothermal"))
    else:
        rows.append(("Thermal", "Jacket",
                     f"{COOLANT_MASS_FLOW:g} kg/s at {COOLANT_TEMPERATURE:g} K;"
                     f" h {VESSEL_H_CONV:g} / {COOLANT_H_CONV:g} W/m²/K;"
                     f" D {VESSEL_DIAMETER:g} m"))
    rows += [
        ("Run", "Runtime", f"{RUNTIME:g} s (dose + {HOLD_TIME:g} s hold)"),
        ("Run", "T peak / final", f"{temp.max():.2f} K / {temp[-1]:.2f} K"),
    ]
    return rows


def plot_settings(reactor, title="Semibatch reactor settings"):
    rows, previous = [], None
    for group, setting, value in settings_rows(reactor):
        rows.append([group if group != previous else "", setting, value])
        previous = group
    return table_figure(rows, ["Group", "Setting", "Value"], title,
                        col_widths=[0.13, 0.25, 0.62])


def plot_mass_balance(reactor, title="Semibatch reactor mole balance"):
    balance = mole_balance(reactor)

    def cell(value):
        return "0" if abs(value) < 5e-10 else f"{value:.6f}"

    rows = [[name, cell(balance.charged[name]), cell(balance.dosed[name]),
             cell(balance.reacted[name]), cell(balance.final[name])]
            for name in reactor.name_species
            if any(abs(getattr(balance, term)[name]) > 5e-10
                   for term in ("charged", "dosed", "final"))]
    return table_figure(
        rows, ["Species", "Charged", "Dosed", "Reacted", "Final"],
        f"{title} at t = {balance.time:g} s  [mol]",
        note=(f"Reacted = final - charged - dosed. C consumed minus "
              f"D + E + J (+ F) formed: {balance.closure:.2g} mol."))


def save_figures(reactor, out_dir: Path = RESULTS_DIR):
    out_dir.mkdir(parents=True, exist_ok=True)
    figures = {
        "SBR_profiles": plot_profiles(reactor),
        "SBR_normalized": plot_normalized(reactor),
        "SBR_settings": plot_settings(reactor),
        "SBR_mass_balance": plot_mass_balance(reactor),
    }
    for name, fig in figures.items():
        fig.savefig(out_dir / f"{name}{figure_suffix()}.png", dpi=150,
                    bbox_inches="tight")
    return figures


def figure_suffix() -> str:
    """'' for R02R, '_first_order' etc. so model runs don't overwrite."""
    return "" if KINETICS_MODEL == "R02R" else f"_{KINETICS_MODEL}"


def add_kinetics_argument(parser):
    parser.add_argument("--kinetics", choices=list(KINETIC_MODELS),
                        default=KINETICS_MODEL,
                        help=f"rate law (default {KINETICS_MODEL})")


def print_parameters(reactor):
    flags = estimation_flags()
    print(f"Kinetic parameters ({KINETICS_MODEL}), as ParameterEstimation "
          "orders them (* = fitted by estimation_flags()):")
    for (name, value), free in zip(parameter_table(reactor), flags):
        print(f"  {'*' if free else ' '} {name:<10}{value:>14.6g}")


def print_summary(reactor):
    balance = mole_balance(reactor)
    temp = np.ravel(reactor.result.global_temp)
    print(f"Moles at t = {balance.time:g} s (mol):")
    print(f"  {'species':<8}{'charged':>10}{'dosed':>10}{'reacted':>10}"
          f"{'final':>10}")
    for name in report_species() + (solvent_name(),):
        print(f"  {name:<8}{balance.charged[name] :>10.6f}"
              f"{balance.dosed[name] :>10.6f}"
              f"{balance.reacted[name] :>10.6f}"
              f"{balance.final[name] :>10.6f}")
    print(f"  C consumed - (D + E + J + F) formed: {balance.closure:.2g} mol")
    print(f"Temperature: peak {temp.max():.2f} K, final {temp[-1]:.2f} K")
    vol = np.ravel(reactor.result.vessel_vol)
    print(f"Final mass "
          f"{float(np.ravel(reactor.result.Total_m_in_vessel)[-1]) * 1e3:.2f} g")
    print_volumes(vol)


def print_volumes(vessel_vol):
    """Charge, feed and final volumes as PharmaPy computes them, against
    the quoted ones [mL]."""
    print("Volumes (mL), PharmaPy vs quoted:")
    print(f"  {'charge':<14}{vessel_vol[0] * 1e6:>8.2f}"
          f"{CHARGE_VOLUME * 1e6:>8.2f}"
          f"   ({charge_solvent_mass() * 1e3:.3f} g H, "
          f"{charge_moles()[solvent_name()]:.6f} mol)")
    volumes = stream_volumes()
    for label, volume in volumes.items():
        quoted = QUOTED_STREAM_VOLUMES.get(label)
        print(f"  {label:<14}{volume * 1e6:>8.2f}"
              + (f"{quoted * 1e6:>8.2f}" if quoted else f"{'-':>8}"))
    print(f"  {'charge + feeds':<14}"
          f"{(vessel_vol[0] + sum(volumes.values())) * 1e6:>8.2f}")
    print(f"  {'final':<14}{vessel_vol[-1] * 1e6:>8.2f}"
          f"{EXPECTED_FINAL_VOLUME * 1e6:>8.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-show", action="store_true",
                        help="save the figures without opening windows")
    add_kinetics_argument(parser)
    args = parser.parse_args()
    KINETICS_MODEL = args.kinetics

    if args.no_show:
        import matplotlib
        matplotlib.use("Agg")

    sbr = run_reactor()
    print_parameters(sbr)
    if "F" in report_species():
        print(f"C runs out at {c_depletion_time(sbr) / 60:.2f} min "
              f"(below {C_DEPLETION_FRACTION:.0%} of its peak)")
    print_summary(sbr)
    save_figures(sbr)
    print(f"Figures saved to {RESULTS_DIR}")

    if not args.no_show:
        import matplotlib.pyplot as plt
        plt.show()
