"""Physical conditions of the chamber (or of a simulation).

You describe the setting with four everyday quantities (temperature, pressure,
dry particle size, chamber height). From these, ``Conditions`` computes the four
constants the model needs:

    a    growth constant         [um^2/s]      how fast droplets grow per unit supersaturation
    A_k  curvature constant      [um]          small droplets are harder to grow (Kelvin effect)
    B_k  solute constant         [um^3]        dissolved salt helps small droplets grow (Raoult effect)
    k    fallout constant        [1/(s um^2)]  droplets of radius r fall out at rate k * r^2

Optionally, two accommodation coefficients (alpha_c, alpha_T) add a fifth constant:

    ell  kinetic length          [um]          small droplets grow more slowly, by a factor r / (r + ell)

and a fallout factor f changes the fallout rate to k * r^2 * f(r) (droplets are
removed faster than k * r^2 if, for example, they crowd toward the floor).

The formulas follow Steve Krueger's MATLAB solver (xi1_func.m, esat_Pa.m and the
benchmark script); the kinetic length follows Fukuta and Walter (1970), as in
C-ODT. The aerosol is sodium chloride (NaCl).
"""
from dataclasses import dataclass
from typing import Callable, Optional, Union
import math
import numpy as np

# fixed physical constants (SI units)
RHO_W = 1000.0        # density of liquid water [kg/m^3]
RHO_S = 2160.0        # density of NaCl [kg/m^3]
M_W = 18e-3           # molar mass of water [kg/mol]
M_S = 58.5e-3         # molar mass of NaCl [kg/mol]
NU_S = 2.0            # van 't Hoff factor of NaCl
SIGMA_W = 0.075       # surface tension of water [N/m]
R_UNIV = 8.314        # universal gas constant [J/(mol K)]
R_AIR = 287.0         # gas constant of dry air [J/(kg K)]
R_VAP = 461.5         # gas constant of water vapor [J/(kg K)]
L_V = 2.5e6           # latent heat of vaporization [J/kg]
G = 9.81              # gravitational acceleration [m/s^2]
MU_AIR = 1.8e-5       # dynamic viscosity of air [Pa s]
C_V_AIR = 718.0       # specific heat of dry air at constant volume [J/(kg K)]
T0 = 273.15           # 0 deg C in kelvin


def saturation_vapor_pressure(T):
    """Saturation vapor pressure over liquid water [Pa] at temperature T [K]."""
    return 100.0 * 10.0 ** (23.832241 - 5.02808 * math.log10(T)
                            - 1.3816e-7 * 10.0 ** (11.344 - 0.0303998 * T)
                            + 8.1328e-3 * 10.0 ** (3.49149 - 1302.8844 / T)
                            - 2949.076 / T)


@dataclass(frozen=True)
class Conditions:
    """Physical setting of the chamber or simulation.

    Parameters
    ----------
    T : float
        Air temperature [K]. Default 288 K (15 deg C).
    p : float
        Air pressure [Pa]. Default 1.0e5 Pa (1000 hPa).
    d_dry : float
        Diameter of the dry salt particle each droplet forms on [um]. Default 0.13 um.
    H : float
        Height of the chamber [m]; sets how quickly droplets fall out. Default 1 m.
    alpha_c : float or None
        Condensation (mass) accommodation coefficient: the fraction of water vapor
        molecules hitting a droplet that stick (0 < alpha_c <= 1). Default None,
        meaning no kinetic correction (ell = 0), as in the original model.
        C-ODT uses alpha_c = 0.04.
    alpha_T : float
        Thermal accommodation coefficient (0 < alpha_T <= 1). Only used when alpha_c
        is given. Default 1, as in C-ODT.
    fallout_factor : float, or a function of radius
        Multiplies the fallout rate: droplets of radius r [um] are removed at rate
        k * r^2 * fallout_factor. Default 1, which assumes droplets are spread evenly
        in height. Give a number for the same change at every size, or a function of r
        for a change that depends on size (for example ``lambda r: 1 + 0.005 * r**2``).
        Values above 1 mean droplets are removed faster, for example because they
        crowd toward the floor.

    The model constants are available as ``cond.a``, ``cond.A_k``, ``cond.B_k``, ``cond.k``
    and ``cond.ell``, and ``cond.fallout_rate(r)`` gives the fallout rate [1/s].
    Use ``print(cond)`` to see all of them.

    Examples
    --------
    >>> cond = Conditions()                  # defaults (no kinetic correction)
    >>> cond = Conditions(T=285.0, H=1.2)    # change any of the inputs
    >>> cond = Conditions(alpha_c=0.04)      # include the kinetic correction, as in C-ODT
    >>> cond = Conditions(fallout_factor=1.3)                        # 30% faster fallout at every size
    >>> cond = Conditions(fallout_factor=lambda r: 1 + 0.005 * r**2)  # faster for bigger droplets
    """
    T: float = 288.0
    p: float = 1.0e5
    d_dry: float = 0.13
    H: float = 1.0
    alpha_c: Optional[float] = None
    alpha_T: float = 1.0
    fallout_factor: Union[float, Callable] = 1.0

    def __post_init__(self):
        for name in ("T", "p", "d_dry", "H"):
            value = getattr(self, name)
            if not (isinstance(value, (int, float)) and math.isfinite(value) and value > 0):
                raise ValueError(f"Conditions: {name} must be a positive number (got {value!r}).")
        if not 233.0 <= self.T <= 313.0:
            raise ValueError(f"Conditions: T = {self.T} K is outside the range the formulas are meant "
                             "for (about -40 to +40 deg C, i.e. 233-313 K).")
        for name in ("alpha_c", "alpha_T"):
            value = getattr(self, name)
            if name == "alpha_c" and value is None:
                continue
            if not (isinstance(value, (int, float)) and math.isfinite(value) and 0 < value <= 1):
                raise ValueError(f"Conditions: {name} must be a number between 0 and 1 (got {value!r}).")
        f = self.fallout_factor
        if not callable(f) and not (isinstance(f, (int, float)) and math.isfinite(f) and f > 0):
            raise ValueError("Conditions: fallout_factor must be a positive number or a function of "
                             f"radius (got {f!r}).")

    # ------------------------------------------------------------------ air properties
    def _conductivity(self):
        """Thermal conductivity of air [W/(m K)]."""
        return 7.7e-5 * (self.T - T0) + 0.02399

    def _diffusivity(self):
        """Diffusivity of water vapor in air [m^2/s]."""
        return (1.57e-7 * (self.T - T0) + 2.211e-5) * 1e5 / self.p

    def _resistances(self):
        """The two resistances to growth: heat conduction (F_k) and vapor diffusion (F_d) [s/m^2]."""
        T = self.T
        F_k = (L_V / (R_VAP * T)) * L_V / (self._conductivity() * T)
        F_d = R_VAP * T / (saturation_vapor_pressure(T) * self._diffusivity())
        return F_k, F_d

    # ------------------------------------------------------------------ model constants
    @property
    def a(self):
        """Growth constant [um^2/s]: a droplet grows as d(r^2)/dt = a * (s - s_eq)
        (or d(r + ell)^2/dt = a * (s - s_eq) with the kinetic correction)."""
        F_k, F_d = self._resistances()
        xi1 = 1e12 / (RHO_W * (F_k + F_d))                    # dr/dt = xi1 * s / r  [um^2/s]
        return 2.0 * xi1

    @property
    def ell(self):
        """Kinetic length [um]: with it, droplets grow as dr/dt = (a/2) (s - s_eq) / (r + ell).

        It is a weighted average of the vapor and heat "jump lengths" set by the
        accommodation coefficients (Fukuta and Walter 1970). It is 0 when alpha_c is None.
        """
        if self.alpha_c is None:
            return 0.0
        T, p = self.T, self.p
        jump_vapor = math.sqrt(2.0 * math.pi / (R_VAP * T)) * self._diffusivity() / self.alpha_c
        jump_heat = (self._conductivity() * math.sqrt(2.0 * math.pi * R_AIR * T)
                     / (self.alpha_T * p * (C_V_AIR + R_AIR / 2.0)))
        F_k, F_d = self._resistances()
        return (F_d * jump_vapor + F_k * jump_heat) / (F_d + F_k) * 1e6

    @property
    def A_k(self):
        """Curvature (Kelvin) constant [um]."""
        return 2.0 * SIGMA_W * M_W / (RHO_W * R_UNIV * self.T) * 1e6

    @property
    def B_k(self):
        """Solute (Raoult) constant [um^3]."""
        r_dry = 0.5 * self.d_dry * 1e-6                       # m
        m_s = (4.0 / 3.0) * math.pi * r_dry**3 * RHO_S        # kg of salt per droplet
        return 3.0 * NU_S * m_s * M_W / (4.0 * math.pi * RHO_W * M_S) * 1e18

    @property
    def k(self):
        """Fallout constant [1/(s um^2)]: droplets of radius r are removed at rate k * r^2."""
        rho_air = self.p / (R_AIR * self.T)
        stokes = 2.0 * (RHO_W - rho_air) * G / (9.0 * MU_AIR)  # fall speed / r^2 [1/(m s)]
        return stokes / self.H * 1e-12

    def fallout_rate(self, r):
        """Rate [1/s] at which droplets of radius r [um] fall out: k * r^2 * fallout_factor."""
        r = np.asarray(r, float)
        f = self.fallout_factor
        factor = np.broadcast_to(np.asarray(f(r) if callable(f) else f, float), r.shape)
        if not np.all(np.isfinite(factor) & (factor > 0)):
            raise ValueError("Conditions: fallout_factor must be positive at every droplet size "
                             f"(it is not between r = {r.min():.3g} and {r.max():.3g} um).")
        return self.k * r**2 * factor

    @property
    def critical_radius(self):
        """Radius [um] a droplet must pass to keep growing at the critical supersaturation."""
        return math.sqrt(3.0 * self.B_k / self.A_k)

    @property
    def critical_supersaturation(self):
        """Supersaturation (as a fraction) needed to activate a droplet on this particle."""
        return math.sqrt(4.0 * self.A_k**3 / (27.0 * self.B_k))

    def __str__(self):
        if self.alpha_c is None:
            accommodation = ["  accommodation: none given (no kinetic correction)"]
        else:
            accommodation = [f"  accommodation alpha_c = {self.alpha_c:g}, alpha_T = {self.alpha_T:g}"]
        f = self.fallout_factor
        if callable(f):
            try:
                example = f" ({float(np.asarray(f(np.array(5.0)))):.3g} at r = 5 um)"
            except Exception:
                example = ""
            fallout = [f"  fallout factor = a function of r{example}"]
        else:
            fallout = [f"  fallout factor = {f:g}"]
        return "\n".join([
            "Conditions",
            f"  temperature  T     = {self.T:g} K",
            f"  pressure     p     = {self.p:g} Pa",
            f"  dry diameter d_dry = {self.d_dry:g} um",
            f"  chamber      H     = {self.H:g} m",
            *accommodation,
            *fallout,
            "Model constants",
            f"  growth       a     = {self.a:.4g} um^2/s",
            f"  curvature    A_k   = {self.A_k:.4g} um",
            f"  solute       B_k   = {self.B_k:.4g} um^3",
            f"  fallout      k     = {self.k:.4g} 1/(s um^2)",
            f"  kinetic      ell   = {self.ell:.4g} um",
            f"  critical supersaturation = {self.critical_supersaturation:.4g} "
            f"({100 * self.critical_supersaturation:.3g}%)",
        ])
