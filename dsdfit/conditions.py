"""Physical conditions of the chamber (or of a simulation).

You describe the setting with four everyday quantities (temperature, pressure,
dry particle size, chamber height). From these, ``Conditions`` computes the four
constants the model needs:

    a    growth constant         [um^2/s]      how fast droplets grow per unit supersaturation
    A_k  curvature constant      [um]          small droplets are harder to grow (Kelvin effect)
    B_k  solute constant         [um^3]        dissolved salt helps small droplets grow (Raoult effect)
    k    fallout constant        [1/(s um^2)]  droplets of radius r fall out at rate k * r^2

The formulas follow Steve Krueger's MATLAB solver (xi1_func.m, esat_Pa.m and the
benchmark script). The aerosol is sodium chloride (NaCl).
"""
from dataclasses import dataclass
import math

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

    The model constants are available as ``cond.a``, ``cond.A_k``, ``cond.B_k`` and ``cond.k``.
    Use ``print(cond)`` to see all of them.

    Examples
    --------
    >>> cond = Conditions()                  # defaults
    >>> cond = Conditions(T=285.0, H=1.2)    # change any of the four inputs
    """
    T: float = 288.0
    p: float = 1.0e5
    d_dry: float = 0.13
    H: float = 1.0

    def __post_init__(self):
        for name in ("T", "p", "d_dry", "H"):
            value = getattr(self, name)
            if not (isinstance(value, (int, float)) and math.isfinite(value) and value > 0):
                raise ValueError(f"Conditions: {name} must be a positive number (got {value!r}).")
        if not 233.0 <= self.T <= 313.0:
            raise ValueError(f"Conditions: T = {self.T} K is outside the range the formulas are meant "
                             "for (about -40 to +40 deg C, i.e. 233-313 K).")

    # ------------------------------------------------------------------ model constants
    @property
    def a(self):
        """Growth constant [um^2/s]: a droplet grows as d(r^2)/dt = a * (s - s_eq)."""
        T, p = self.T, self.p
        K = 7.7e-5 * (T - T0) + 0.02399                      # thermal conductivity of air [W/(m K)]
        D_v = (1.57e-7 * (T - T0) + 2.211e-5) * 1e5 / p      # diffusivity of water vapor [m^2/s]
        denom = RHO_W * ((L_V / (R_VAP * T)) * L_V / (K * T)
                         + R_VAP * T / (saturation_vapor_pressure(T) * D_v))
        xi1 = 1e12 / denom                                    # dr/dt = xi1 * s / r  [um^2/s]
        return 2.0 * xi1

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

    @property
    def critical_radius(self):
        """Radius [um] a droplet must pass to keep growing at the critical supersaturation."""
        return math.sqrt(3.0 * self.B_k / self.A_k)

    @property
    def critical_supersaturation(self):
        """Supersaturation (as a fraction) needed to activate a droplet on this particle."""
        return math.sqrt(4.0 * self.A_k**3 / (27.0 * self.B_k))

    def __str__(self):
        return "\n".join([
            "Conditions",
            f"  temperature  T     = {self.T:g} K",
            f"  pressure     p     = {self.p:g} Pa",
            f"  dry diameter d_dry = {self.d_dry:g} um",
            f"  chamber      H     = {self.H:g} m",
            "Model constants",
            f"  growth       a     = {self.a:.4g} um^2/s",
            f"  curvature    A_k   = {self.A_k:.4g} um",
            f"  solute       B_k   = {self.B_k:.4g} um^3",
            f"  fallout      k     = {self.k:.4g} 1/(s um^2)",
            f"  critical supersaturation = {self.critical_supersaturation:.4g} "
            f"({100 * self.critical_supersaturation:.3g}%)",
        ])
