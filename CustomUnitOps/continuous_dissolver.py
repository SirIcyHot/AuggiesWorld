# -*- coding: utf-8 -*-
"""
Steady-state continuous dissolver (redissolution / resuspension).

Feeds, typically a filter cake plus fresh solvent, are mixed and part of the
crystals dissolve. The outlet is one slurry stream: the enriched liquid plus
whatever crystals are left. The dissolver holds nothing and needs no solver
beyond two scalar root finds.

How much dissolves is set in one of two ways:

- ``solubility``: a function ``solubility(temp, conc) -> mol/L of solution``,
  with ``conc`` the liquid's molar concentrations [kmol/m3 = mol/L] in
  database order. Crystals dissolve until the liquid is saturated, or until
  none are left. ``approach`` (0 to 1) takes only that fraction of the way
  to equilibrium, a stand-in for a residence time too short to get there.
- ``fraction_dissolved``: that fraction of the crystal mass dissolves.

Equations
---------
Mixing. Species and crystal flows add; the temperature is the mass-weighted
average of the feeds unless ``temp`` is given::

    m_j  = sum over feeds of m_j,feed            [kg/s]
    N_i  = sum over feeds of N_i,feed            [#/s in size bin i]

Equilibrium. With D the crystal mass dissolved per second and t the
crystallizing species, the liquid after dissolution has::

    m_t(D) = m_t + D
    Q_L(D) = sum_j m_j(D) / rho_L(D)             [m3/s]
    c_t(D) = m_t(D) / (MW_t * Q_L(D))            [kmol/m3]

and equilibrium is ``c_t(D) = solubility(T, c(D))``. Three cases:

    c_t(M_s) <= S   all crystals dissolve (D = M_s, the feed crystal mass)
    c_t(0)   >= S   nothing dissolves (the feed liquid is already saturated)
    otherwise       solve c_t(D) - S(T, c(D)) = 0 for D in (0, M_s)

then ``D = approach * D_eq``.

Size distribution. Dissolution is taken as size-independent: every crystal
shrinks by the same length delta (the reverse of McCabe's Delta-L law), and
crystals smaller than delta vanish. delta is found from the dissolved mass::

    M_out(delta) = rho_s * kv * 1e-18 * sum_i N_i * max(x_i - delta, 0)**3
    M_s - M_out(delta) = D

A shrunken crystal of size y = x_i - delta generally falls between two grid
nodes x_k <= y < x_k+1. Its count is shared between them with weights that
keep both the number and the mass of crystals::

    w_k + w_k+1 = 1
    w_k * x_k**3 + w_k+1 * x_k+1**3 = y**3

Crystals that shrink below the first node keep their mass in the first bin.

Mass balance. The dissolved crystals join the liquid with the crystals'
composition, so the total mass flow out equals the total in exactly.

The dissolver is isothermal: there is no heat of dissolution.

Units
-----
Mass flow [kg/s], temperature [K], size [um], concentration [kmol/m3 =
mol/L of solution], crystal number flow [#/(um s)].
"""

import warnings

import numpy as np
from scipy.optimize import brentq

from continuous_filter import SlurryFlow, mix, stream_from_flow
from PharmaPy.Streams_Refactored import LiquidStream

# A micron-based third moment times this is m**3.
VOLUME_UNIT_FACTOR = 1e-18


# ----------------------------------------------------------------------
# Size distribution under uniform shrinkage
# ----------------------------------------------------------------------

def _bin_counts(flow):
    """Crystals per second in each bin [#/s]."""
    return flow.crystal_flow * flow.dx


def _mass_after_shrink(counts, x_grid, delta, density, kv):
    """Crystal mass flow [kg/s] once every crystal has shrunk by delta."""
    sizes = np.maximum(x_grid - delta, 0.0)
    return float(density * kv * VOLUME_UNIT_FACTOR * np.sum(counts * sizes**3))


def shrink(flow, delta):
    """Crystal number flow [#/(um s)] after every crystal shrinks by delta.

    Each bin's crystals move to size x_i - delta and are shared between the
    two grid nodes around that size so that number and mass are both kept.
    Crystals that end up below the first node keep only their mass, in the
    first bin.
    """
    x = flow.x_grid
    counts = _bin_counts(flow)
    out = np.zeros_like(counts)

    for count, size in zip(counts, x - delta):
        if count == 0 or size <= 0:
            continue

        if size <= x[0]:
            out[0] += count * (size / x[0]) ** 3
            continue

        k = min(np.searchsorted(x, size, side="right") - 1, len(x) - 2)
        lo, hi = x[k] ** 3, x[k + 1] ** 3
        w_hi = (size**3 - lo) / (hi - lo)
        out[k] += count * (1.0 - w_hi)
        out[k + 1] += count * w_hi

    return out / flow.dx


def shrink_for_mass(flow, dissolved):
    """Shrinkage delta [um] that dissolves `dissolved` kg/s of crystals."""
    counts = _bin_counts(flow)
    total = flow.solid_mass_flow

    if dissolved <= 0:
        return 0.0

    if dissolved >= total:
        return float(flow.x_grid[-1])

    def gap(delta):
        remaining = _mass_after_shrink(counts, flow.x_grid, delta,
                                       flow.solid_density, flow.kv)
        return total - remaining - dissolved

    return brentq(gap, 0.0, float(flow.x_grid[-1]), xtol=1e-12, rtol=1e-12)


# ----------------------------------------------------------------------
# The dissolver
# ----------------------------------------------------------------------

class ContinuousDissolver:
    """Steady-state continuous dissolver.

    Parameters
    ----------
    solubility : callable, optional
        ``solubility(temp, conc) -> mol/L of solution`` for the
        crystallizing species. Gives an equilibrium dissolution.
    fraction_dissolved : float, optional
        Fraction of the crystal mass that dissolves [-]. Use instead of
        ``solubility``, e.g. for a mixed crystal population.
    approach : float, optional
        Fraction of the way to equilibrium reached [-], with ``solubility``.
    temp : float, optional
        Dissolver temperature [K]. Defaults to the mixed feed temperature.

    Examples
    --------
    >>> diss = ContinuousDissolver(solubility=f_solubility)
    >>> diss.Inlets = [cake, solvent]
    >>> outlet = diss.solve()
    """

    def __init__(self, solubility=None, fraction_dissolved=None,
                 approach=1.0, temp=None):

        if (solubility is None) == (fraction_dissolved is None):
            raise ValueError(
                "Give exactly one of solubility or fraction_dissolved.")

        if fraction_dissolved is not None and not 0 <= fraction_dissolved <= 1:
            raise ValueError("fraction_dissolved must lie in [0, 1].")

        if not 0 <= approach <= 1:
            raise ValueError("approach must lie in [0, 1].")

        self.solubility = solubility
        self.fraction_dissolved = fraction_dissolved
        self.approach = approach
        self.temp = temp

        self._inlets = []
        self.feed = None
        self.outlet_flow = None
        self.dissolved = None
        self.delta = None
        self.Outlet = None

    @property
    def Inlets(self):
        return self._inlets

    @Inlets.setter
    def Inlets(self, inlets):
        """One feed or a list: PharmaPy streams or `SlurryFlow`s."""
        self._inlets = list(inlets) if isinstance(inlets, (list, tuple)) \
            else [inlets]

    Inlet = Inlets

    # ---------------- equilibrium ----------------

    def _target_index(self, feed):
        species = np.flatnonzero(feed.solid_mass_frac > 0)

        if len(species) != 1:
            names = [feed.name_species[j] for j in species]
            raise ValueError(
                "An equilibrium dissolution needs crystals of one species, "
                f"but these are a mix of {names}. Use fraction_dissolved, "
                "or give each species its own solid.")

        return int(species[0])

    def _liquid_conc(self, feed, liquid_mass_j, temp):
        total = float(np.sum(liquid_mass_j))
        liquid = LiquidStream(feed.path_thermo, mass_flow=total,
                              mass_frac=liquid_mass_j / total, temp=temp,
                              name_solv=feed.name_solv, verbose=False)
        return np.asarray(liquid.mole_conc, dtype=float)

    def equilibrium_dissolved(self, feed, temp):
        """Crystal mass flow [kg/s] that dissolves at equilibrium."""
        j = self._target_index(feed)
        available = feed.solid_mass_flow

        def excess(dissolved):
            mass_j = feed.liquid_mass_j.copy()
            mass_j[j] += dissolved
            conc = self._liquid_conc(feed, mass_j, temp)
            return conc[j] - float(self.solubility(temp, conc))

        if excess(available) <= 0:
            return available  # undersaturated even with every crystal gone

        if excess(0.0) >= 0:
            warnings.warn(
                "The feed liquid is already at or above saturation, so "
                "nothing dissolves. This unit does not grow crystals.")
            return 0.0

        return brentq(excess, 0.0, available, xtol=1e-16, rtol=1e-12)

    # ---------------- solve ----------------

    def solve(self):
        """Dissolve and return the outlet slurry as a PharmaPy MixedStream.

        The stream has a solid phase only if crystals remain. Also stored as
        ``self.Outlet``; ``self.outlet_flow`` holds the same in absolute
        flows, ``self.dissolved`` the crystal mass dissolved [kg/s] and
        ``self.delta`` how far every crystal shrank [um].
        """
        feed = mix(self.Inlets)

        if self.temp is not None:
            feed.temp = float(self.temp)

        self.feed = feed

        if not feed.has_solids or feed.solid_mass_flow <= 0:
            raise ValueError("None of the feeds carries crystals.")

        if self.solubility is not None:
            dissolved = self.approach * self.equilibrium_dissolved(feed,
                                                                   feed.temp)
        else:
            dissolved = self.fraction_dissolved * feed.solid_mass_flow

        delta = shrink_for_mass(feed, dissolved)
        crystals = shrink(feed, delta)

        remaining = SlurryFlow(
            feed.path_thermo, feed.name_species, feed.temp,
            np.zeros_like(feed.liquid_mass_j), feed.name_solv,
            x_grid=feed.x_grid, crystal_flow=crystals,
            solid_mass_frac=feed.solid_mass_frac,
            solid_density=feed.solid_density, kv=feed.kv,
            target_components=feed.target_components,
            solvent_name=feed.solvent_name, dx=feed.dx,
        )

        # What actually left the crystals, so the balance closes to roundoff
        # whatever the root finder's tolerance.
        dissolved = feed.solid_mass_flow - remaining.solid_mass_flow

        remaining.liquid_mass_j = (feed.liquid_mass_j
                                   + dissolved * feed.solid_mass_frac)

        self.outlet_flow = remaining
        self.dissolved = dissolved
        self.delta = delta
        self.Outlet = stream_from_flow(remaining)

        return self.Outlet

    # ---------------- reporting ----------------

    def summary(self):
        """Flows and closures of the last solve, as a dict."""
        if self.feed is None:
            raise ValueError("Call solve() first.")

        feed, out = self.feed, self.outlet_flow
        total_in = feed.liquid_mass_flow + feed.solid_mass_flow
        total_out = out.liquid_mass_flow + out.solid_mass_flow

        return {
            "feed_solid_kg_s": feed.solid_mass_flow,
            "feed_liquid_kg_s": feed.liquid_mass_flow,
            "dissolved_kg_s": self.dissolved,
            "fraction_dissolved": self.dissolved / feed.solid_mass_flow,
            "outlet_solid_kg_s": out.solid_mass_flow,
            "outlet_liquid_kg_s": out.liquid_mass_flow,
            "shrinkage_um": self.delta,
            "mass_balance_error_kg_s": total_in - total_out,
            "temp_K": feed.temp,
        }
