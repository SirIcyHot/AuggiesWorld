# -*- coding: utf-8 -*-
"""
Steady-state continuous filter, standalone from the PharmaPy package.

The filter is a splitter. All feeds are summed into one slurry and split into
two streams:

- cake:     the retained crystals plus the mother liquor held in the cake
- filtrate: the remaining mother liquor plus any crystals that pass the cloth

Two settings decide the split:

- ``solids_loss``: the fraction of crystals that pass into the filtrate. A
  scalar applies to every size; an array gives one fraction per size bin,
  which is how a fines cut-off is modelled.
- ``cake_moisture``: how much liquid the cake keeps, either as loss on drying
  (kg liquid / kg wet cake, ``moisture_basis='wet'``, the default) or as
  kg liquid / kg dry solid (``moisture_basis='dry'``). The retained liquid has
  the same composition as the mixed mother liquor: there is no washing.

The filter is isothermal and holds nothing, so it needs no solver. Each call
to ``solve`` is one steady state.

Crystal bookkeeping
-------------------
The refactored PharmaPy stack stores a crystal population as a number density
per m3 of slurry. A filter is the one unit where liquid leaves without the
crystals, so a density that is carried through unchanged loses crystals.
This module therefore converts every feed to absolute flows on the way in:

- liquid:   kg/s of each species
- crystals: #/(um s) in each size bin

It converts back to a per-m3-of-slurry density only when it builds the
output streams. The cake's density is the crystal flow divided by the
cake's own slurry flow, so the count is conserved by construction.

Units
-----
Mass flow [kg/s], volumetric flow [m3/s], temperature [K], size [um], crystal
number density [#/(um m3 slurry)], crystal number flow [#/(um s)].
"""

import warnings
from dataclasses import dataclass, field

import numpy as np

from PharmaPy.DataClasses import PhaseRef
from PharmaPy.MixedPhases_Refactored import MixedStream
from PharmaPy.Mechanisms import OneDFVMMechanism
from PharmaPy.Phases_Refactored import SolidPhase
from PharmaPy.Streams_Refactored import LiquidStream

# A micron-based third moment times this is m**3.
VOLUME_UNIT_FACTOR = 1e-18


@dataclass
class SlurryFlow:
    """A slurry as absolute flows, the form the filter works in.

    Attributes
    ----------
    path_thermo : str
        Compound database the PharmaPy phases are built from.
    name_species : list of str
        Species names, in database order.
    temp : float
        Temperature [K].
    liquid_mass_j : numpy.ndarray
        Liquid mass flow of each species [kg/s].
    name_solv : str, optional
        Solvent name, passed through to the liquid streams.
    x_grid : numpy.ndarray, optional
        Crystal size grid [um]. None for a liquid-only flow.
    crystal_flow : numpy.ndarray, optional
        Crystal number flow in each size bin [#/(um s)].
    solid_mass_frac : numpy.ndarray, optional
        Composition of the crystals [-].
    solid_density : float, optional
        Crystal density [kg/m3].
    kv : float, optional
        Volumetric shape factor [-].
    target_components, solvent_name : optional
        Mechanism settings, passed to the cake's population balance so that
        a downstream crystallizer receives a solid configured like its own.
    """

    path_thermo: str
    name_species: list
    temp: float
    liquid_mass_j: np.ndarray
    name_solv: str = None
    x_grid: np.ndarray = None
    crystal_flow: np.ndarray = None
    solid_mass_frac: np.ndarray = None
    solid_density: float = None
    kv: float = 1.0
    target_components: object = None
    solvent_name: str = None
    dx: object = field(default=None, repr=False)

    def __post_init__(self):
        self.liquid_mass_j = np.asarray(self.liquid_mass_j, dtype=float)

        if self.x_grid is not None:
            self.x_grid = np.asarray(self.x_grid, dtype=float)
            self.crystal_flow = np.asarray(self.crystal_flow, dtype=float)

            if self.dx is None:
                # Same cell widths as the FVM mechanism, so a moment taken
                # here matches the one the downstream vessel takes.
                self.dx = OneDFVMMechanism._cell_widths(self.x_grid)

    @property
    def has_solids(self):
        return self.x_grid is not None

    @property
    def liquid_mass_flow(self):
        """Total liquid mass flow [kg/s]."""
        return float(np.sum(self.liquid_mass_j))

    @property
    def solid_mass_flow(self):
        """Crystal mass flow [kg/s], from the third moment of the flow."""
        if not self.has_solids:
            return 0.0

        mu_3 = np.sum(self.crystal_flow * self.x_grid**3 * self.dx)

        return float(self.solid_density * self.kv * mu_3
                     * VOLUME_UNIT_FACTOR)


# ----------------------------------------------------------------------
# Feeds into absolute flows
# ----------------------------------------------------------------------

def _mechanism_of(phase):
    for mechanism in getattr(phase, 'mechanisms', None) or ():
        if isinstance(mechanism, OneDFVMMechanism):
            return mechanism

    return None


def flow_from_stream(stream):
    """A PharmaPy stream as a `SlurryFlow`.

    Accepts a `LiquidStream`, or a `MixedStream` of one liquid and at most
    one solid. The solid must carry a `OneDFVMMechanism`.

    The crystal flow is read exactly the way `MultiPhaseVessel` reads a
    static slurry feed: the number density times the stream's total
    volumetric flow (the sum of its phases' ``vol_flow``). The filter therefore
    sees the same crystals a vessel fed this stream directly would see.
    """
    phases = list(stream.Phases) if isinstance(stream, MixedStream) \
        else [stream]

    liquids = [p for p in phases if p.phase_family == 'liquid']
    solids = [p for p in phases if p.phase_family == 'solid']

    if len(liquids) > 1 or len(solids) > 1:
        raise ValueError(
            'A feed can hold one liquid and one solid. Pass several feeds '
            'instead of one stream with several phases of the same kind.')

    if not liquids and not solids:
        raise ValueError('The feed has neither a liquid nor a solid phase.')

    reference = liquids[0] if liquids else solids[0]

    kwargs = dict(
        path_thermo=reference.path_thermo,
        name_species=list(reference.name_species),
        temp=float(reference.temp),
    )

    if liquids:
        liquid = liquids[0]
        kwargs['liquid_mass_j'] = (float(liquid.mass_flow)
                                   * np.asarray(liquid.mass_frac, dtype=float))
        kwargs['name_solv'] = liquid.name_solv
    else:
        kwargs['liquid_mass_j'] = np.zeros(len(reference.name_species))

    if solids:
        solid = solids[0]
        mechanism = _mechanism_of(solid)

        if mechanism is None:
            raise ValueError(
                'The solid in this feed has no OneDFVMMechanism, so it has '
                'no size distribution to filter.')

        slurry_vol_flow = sum(float(p.vol_flow or 0.0) for p in phases)

        kwargs.update(
            x_grid=mechanism.x_grid,
            crystal_flow=np.asarray(mechanism.distrib, dtype=float)
            * slurry_vol_flow,
            solid_mass_frac=np.asarray(solid.mass_frac, dtype=float),
            solid_density=float(mechanism.getDensity()),
            kv=float(mechanism.kv),
            target_components=mechanism.target_components,
            solvent_name=mechanism.solvent_name,
        )

    return SlurryFlow(**kwargs)


def flow_from_vessel(vessel, time_index=-1):
    """The product stream of a solved continuous vessel, at one time point.

    Reads ``vessel.outputs``, the trajectory PharmaPy itself hands between
    units: ``vol_flow`` for the whole stream [m3/s], ``mole_conc`` of the
    liquid [kmol/m3], ``temp`` [K] and the crystal density ``distrib``
    [#/(um m3 slurry)]. The default time point is the last one, which is the
    steady state when the vessel was run long enough.
    """
    outputs = vessel.outputs

    if outputs is None:
        raise ValueError('The vessel has not been solved yet.')

    if 'vol_flow' not in outputs:
        raise ValueError(
            'The vessel publishes no vol_flow, so it is not a continuous '
            'unit. A batch vessel discharges a quantity, not a flow.')

    liquid = vessel.Phases.get_phase_from_ref(PhaseRef('liquid', 0))

    vol_flow = float(np.ravel(outputs['vol_flow'])[time_index])
    temp = float(np.ravel(outputs['temp'])[time_index]) \
        if 'temp' in outputs else float(liquid.temp)

    solid = None
    mechanism = None

    for phase in vessel.Phases:
        if phase.phase_family == 'solid':
            solid = phase
            mechanism = _mechanism_of(phase)

    distrib = None

    if mechanism is not None:
        name = mechanism.distribution_state_name

        if name in outputs:
            distrib = np.atleast_2d(outputs[name])[time_index]

    if distrib is not None:
        solid_fraction = mechanism.solid_volume_fraction(distrib)
    else:
        solid_fraction = 0.0

    liquid_vol_flow = (1.0 - solid_fraction) * vol_flow

    mole_conc = np.atleast_2d(outputs['mole_conc'])[time_index]
    liquid_mass_j = mole_conc * np.asarray(liquid.mw) * liquid_vol_flow

    kwargs = dict(
        path_thermo=liquid.path_thermo,
        name_species=list(liquid.name_species),
        temp=temp,
        liquid_mass_j=liquid_mass_j,
        name_solv=liquid.name_solv,
    )

    if distrib is not None:
        kwargs.update(
            x_grid=mechanism.x_grid,
            crystal_flow=np.asarray(distrib, dtype=float) * vol_flow,
            solid_mass_frac=np.asarray(solid.mass_frac, dtype=float),
            solid_density=float(mechanism.getDensity()),
            kv=float(mechanism.kv),
            target_components=mechanism.target_components,
            solvent_name=mechanism.solvent_name,
        )

    return SlurryFlow(**kwargs)


def _as_flow(feed):
    if isinstance(feed, SlurryFlow):
        return feed

    return flow_from_stream(feed)


def mix(flows):
    """Sum several `SlurryFlow`s into one.

    Species and crystal flows add. The temperature is the mass-weighted
    average, which ignores differences in heat capacity: adequate for feeds
    at similar temperatures, which is the usual case at a filter.
    """
    flows = [_as_flow(f) for f in flows]

    if not flows:
        raise ValueError('The filter has no feeds.')

    first = flows[0]

    for other in flows[1:]:
        if list(other.name_species) != list(first.name_species):
            raise ValueError('All feeds must use the same species list.')

    liquid_mass_j = np.sum([f.liquid_mass_j for f in flows], axis=0)

    masses = np.array([f.liquid_mass_flow + f.solid_mass_flow
                       for f in flows])
    temps = np.array([f.temp for f in flows])
    temp = (float(np.dot(masses, temps) / masses.sum())
            if masses.sum() > 0 else float(temps.mean()))

    name_solv = next((f.name_solv for f in flows if f.name_solv), None)

    solid_flows = [f for f in flows if f.has_solids]

    kwargs = dict(path_thermo=first.path_thermo,
                  name_species=list(first.name_species), temp=temp,
                  liquid_mass_j=liquid_mass_j, name_solv=name_solv)

    if solid_flows:
        ref = solid_flows[0]

        for other in solid_flows[1:]:
            if not np.allclose(other.x_grid, ref.x_grid):
                raise ValueError(
                    'Every slurry feed must use the same size grid. Mixing '
                    'distributions across grids needs interpolation, which '
                    'is not implemented.')

            if not np.isclose(other.solid_density, ref.solid_density):
                raise ValueError('Slurry feeds carry crystals of different '
                                 'densities; they cannot share one solid.')

        solid_mass = np.array([f.solid_mass_flow for f in solid_flows])
        fracs = np.array([f.solid_mass_frac for f in solid_flows])
        solid_mass_frac = (solid_mass @ fracs / solid_mass.sum()
                           if solid_mass.sum() > 0 else fracs[0])

        kwargs.update(
            x_grid=ref.x_grid,
            crystal_flow=np.sum([f.crystal_flow for f in solid_flows],
                                axis=0),
            solid_mass_frac=solid_mass_frac,
            solid_density=ref.solid_density,
            kv=ref.kv,
            target_components=ref.target_components,
            solvent_name=ref.solvent_name,
            dx=ref.dx,
        )

    return SlurryFlow(**kwargs)


# ----------------------------------------------------------------------
# Absolute flows back into PharmaPy streams
# ----------------------------------------------------------------------

def stream_from_flow(flow, include_solid=None):
    """A `SlurryFlow` as a PharmaPy `MixedStream`.

    The stream follows the convention `MultiPhaseVessel` reads a static
    slurry feed with: each phase's ``vol_flow`` is its own share, and the
    crystal density is per m3 of the whole stream. The density is therefore
    the crystal flow divided by the stream's total volumetric flow, which
    gives back exactly the crystal flow a vessel fed this stream will read.

    Parameters
    ----------
    include_solid : bool, optional
        Whether to give the stream a solid phase. Defaults to whether it
        carries any crystals. A vessel with no solid phase rejects a stream
        that has one, so a crystal-free filtrate is left liquid only.
    """
    if include_solid is None:
        include_solid = flow.has_solids and flow.solid_mass_flow > 0

    liquid_mass = flow.liquid_mass_flow
    num_species = len(flow.name_species)

    if liquid_mass > 0:
        mass_frac = flow.liquid_mass_j / liquid_mass
    else:
        # A composition is required even for a zero flow; the solvent is
        # the natural placeholder.
        mass_frac = np.zeros(num_species)
        solv = (flow.name_species.index(flow.name_solv)
                if flow.name_solv in flow.name_species else -1)
        mass_frac[solv] = 1.0

    liquid = LiquidStream(flow.path_thermo, mass_flow=liquid_mass,
                          mass_frac=mass_frac, temp=flow.temp,
                          name_solv=flow.name_solv, verbose=False)

    if not include_solid:
        return MixedStream([liquid])

    solid_vol_flow = flow.solid_mass_flow / flow.solid_density
    slurry_vol_flow = float(liquid.vol_flow) + solid_vol_flow

    density = (flow.crystal_flow / slurry_vol_flow if slurry_vol_flow > 0
               else np.zeros_like(flow.crystal_flow))

    solid_phase = SolidPhase(flow.path_thermo, mass=0,
                             mass_frac=flow.solid_mass_frac, temp=flow.temp,
                             verbose=False)

    # The mechanism insists on both names. Feeds read from PharmaPy always
    # carry them; a hand-built SlurryFlow may not, so fall back to the
    # species the crystals are made of and the liquid's solvent.
    target = flow.target_components
    if target is None:
        target = [name for name, frac
                  in zip(flow.name_species, flow.solid_mass_frac) if frac > 0]

    solvent = flow.solvent_name or flow.name_solv
    if solvent is None:
        raise ValueError(
            'The crystal stream needs a solvent name. Set name_solv on the '
            'liquid feed, or solvent_name on the SlurryFlow.')

    mechanism = OneDFVMMechanism(
        solid_phase,
        target_components=target,
        solvent_name=solvent,
        x_grid=flow.x_grid,
        distrib_init=density,
        density=flow.solid_density,
        kv=flow.kv,
    )

    # The solid's mass is derived from its density times a slurry volume.
    # For a stream that "volume" is the slurry flow, so mass reads back as
    # the crystal mass flow and vol_flow as the crystal volume flow.
    mechanism.slurry_volume = slurry_vol_flow
    solid_phase.mechanisms = mechanism

    solid = solid_phase.to_stream()

    return MixedStream([liquid, solid])


# ----------------------------------------------------------------------
# The filter
# ----------------------------------------------------------------------

class ContinuousFilter:
    """Steady-state continuous filter.

    Parameters
    ----------
    solids_loss : float or array_like, optional
        Fraction of crystals passing into the filtrate [-]. A scalar applies
        to every size; an array gives one fraction per size bin.
    cake_moisture : float, optional
        Liquid retained by the cake [-], on the basis below.
    moisture_basis : {'wet', 'dry'}, optional
        ``'wet'``: kg liquid / kg wet cake, i.e. loss on drying (LOD).
        ``'dry'``: kg liquid / kg dry solid.

    Examples
    --------
    >>> filt = ContinuousFilter(solids_loss=0.01, cake_moisture=0.30)
    >>> filt.Inlets = [slurry_stream, flow_from_vessel(msmpr)]
    >>> cake, filtrate = filt.solve()
    >>> redissolver.Inlet = cake
    """

    def __init__(self, solids_loss=0.0, cake_moisture=0.3,
                 moisture_basis='wet'):

        self.solids_loss = solids_loss
        self.cake_moisture = cake_moisture
        self.moisture_basis = moisture_basis

        self._inlets = []
        self.feed = None
        self.cake_flow = None
        self.filtrate_flow = None
        self.Outlet = None

    # ---------------- settings ----------------

    @property
    def moisture_basis(self):
        return self._moisture_basis

    @moisture_basis.setter
    def moisture_basis(self, value):
        if value not in ('wet', 'dry'):
            raise ValueError("moisture_basis must be 'wet' or 'dry'.")

        self._moisture_basis = value

    def _liquid_per_solid(self):
        """kg of liquid the cake holds per kg of crystals."""
        w = float(self.cake_moisture)

        if self.moisture_basis == 'dry':
            if w < 0:
                raise ValueError('cake_moisture must not be negative.')
            return w

        if not 0 <= w < 1:
            raise ValueError(
                "On the 'wet' basis cake_moisture is a fraction of the wet "
                'cake, so it must be in [0, 1).')

        return w / (1.0 - w)

    def _loss_by_bin(self, num_bins):
        loss = np.broadcast_to(np.asarray(self.solids_loss, dtype=float),
                               (num_bins,))

        if np.any(loss < 0) or np.any(loss > 1):
            raise ValueError('solids_loss must lie in [0, 1].')

        return loss

    # ---------------- feeds ----------------

    @property
    def Inlets(self):
        return self._inlets

    @Inlets.setter
    def Inlets(self, inlets):
        """One feed or a list of them.

        Each feed is a PharmaPy `LiquidStream` or `MixedStream`, or a
        `SlurryFlow` (for example from `flow_from_vessel`).
        """
        self._inlets = list(inlets) if isinstance(inlets, (list, tuple)) \
            else [inlets]

    # The vessel spelling, so the filter reads like the other units.
    Inlet = Inlets

    # ---------------- split ----------------

    def solve(self):
        """Split the mixed feed into cake and filtrate.

        Returns
        -------
        cake, filtrate : MixedStream
            Also available as ``self.Outlet['cake']`` and
            ``self.Outlet['filtrate']``. The cake always has a solid phase.
            The filtrate has one only when crystals are lost to it.
        """
        feed = mix(self.Inlets)
        self.feed = feed

        if not feed.has_solids:
            raise ValueError(
                'None of the feeds carries crystals, so there is nothing '
                'to filter.')

        loss = self._loss_by_bin(len(feed.x_grid))

        crystals_filtrate = feed.crystal_flow * loss
        crystals_cake = feed.crystal_flow - crystals_filtrate

        solid_common = dict(
            x_grid=feed.x_grid, solid_mass_frac=feed.solid_mass_frac,
            solid_density=feed.solid_density, kv=feed.kv,
            target_components=feed.target_components,
            solvent_name=feed.solvent_name, dx=feed.dx,
        )

        # The cake's crystal mass sets how much liquid it holds.
        cake_solid = SlurryFlow(
            feed.path_thermo, feed.name_species, feed.temp,
            np.zeros_like(feed.liquid_mass_j), feed.name_solv,
            crystal_flow=crystals_cake, **solid_common,
        ).solid_mass_flow

        retained = cake_solid * self._liquid_per_solid()
        available = feed.liquid_mass_flow

        if retained > available:
            warnings.warn(
                f'The cake would hold {retained:.4g} kg/s of liquid but the '
                f'feed carries only {available:.4g} kg/s. All of it stays '
                'in the cake and the filtrate is empty; the cake is drier '
                'than cake_moisture asks for.')
            retained = available

        # Retained liquid is mother liquor: same composition as the feed.
        split = retained / available if available > 0 else 0.0
        liquid_cake = feed.liquid_mass_j * split
        liquid_filtrate = feed.liquid_mass_j - liquid_cake

        self.cake_flow = SlurryFlow(
            feed.path_thermo, feed.name_species, feed.temp, liquid_cake,
            feed.name_solv, crystal_flow=crystals_cake, **solid_common)

        self.filtrate_flow = SlurryFlow(
            feed.path_thermo, feed.name_species, feed.temp, liquid_filtrate,
            feed.name_solv, crystal_flow=crystals_filtrate, **solid_common)

        cake = stream_from_flow(self.cake_flow, include_solid=True)
        filtrate = stream_from_flow(self.filtrate_flow)

        self.Outlet = {'cake': cake, 'filtrate': filtrate}

        return cake, filtrate

    # ---------------- reporting ----------------

    def summary(self):
        """Flows and closures of the last solve, as a dict."""
        if self.feed is None:
            raise ValueError('Call solve() first.')

        feed, cake, filt = self.feed, self.cake_flow, self.filtrate_flow

        cake_total = cake.solid_mass_flow + cake.liquid_mass_flow

        return {
            'feed_solid_kg_s': feed.solid_mass_flow,
            'feed_liquid_kg_s': feed.liquid_mass_flow,
            'cake_solid_kg_s': cake.solid_mass_flow,
            'cake_liquid_kg_s': cake.liquid_mass_flow,
            'filtrate_solid_kg_s': filt.solid_mass_flow,
            'filtrate_liquid_kg_s': filt.liquid_mass_flow,
            'cake_lod': (cake.liquid_mass_flow / cake_total
                         if cake_total > 0 else 0.0),
            'solids_recovery': (cake.solid_mass_flow / feed.solid_mass_flow
                                if feed.solid_mass_flow > 0 else 0.0),
            'mass_balance_error_kg_s': (
                feed.solid_mass_flow + feed.liquid_mass_flow
                - cake_total
                - filt.solid_mass_flow - filt.liquid_mass_flow),
            'temp_K': feed.temp,
        }
