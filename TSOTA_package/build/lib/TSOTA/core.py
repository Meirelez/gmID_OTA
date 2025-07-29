import numpy as np
from scipy import constants
from dataclasses import dataclass
from typing import Optional

# Constants
k = constants.Boltzmann
TEMP = 300  # Kelvin

@dataclass
class Specs:
    ts: float
    fu1: float
    fp2: float
    vod_noise: float
    G: float
    FO: float
    L0: float

@dataclass
class DesignParams:
    beta: float = None
    beta_max: float = None
    slew_frac: float = None
    rself1: float = None
    rself2: float = None
    cgs2_cc: float = None
    cltot_cc: float = None
    gm3_gm1: float = None
    gm4_gm2: float = None
    gam1: float = None
    gam2: float = None
    gam3: float = None
    gam4: float = None
    L1: float = None
    L2: float = None
    L3: float = None
    L4: float = None

@dataclass
class Transistor:
    L: float
    gm: Optional[float] = None
    id: Optional[float] = None
    gm_id: Optional[float] = None
    w: Optional[float] = None
    ft: Optional[float] = None
    cgg: Optional[float] = None
    cgs: Optional[float] = None
    cgd: Optional[float] = None
    cdd: Optional[float] = None

@dataclass
class Circuit:
    cc: Optional[float] = None
    cltot: Optional[float] = None
    cf: Optional[float] = None
    cs: Optional[float] = None
    cl: Optional[float] = None
    c1: Optional[float] = None
    rz: Optional[float] = None
    cn: Optional[float] = None
    cc_add: Optional[float] = None

def calculate_capacitances(specs: Specs, params: DesignParams, m2: Transistor) -> Circuit:
    circuit = Circuit()
    kT = k * TEMP

    if params.beta_max is None:
        params.beta_max = 1 / (1 + specs.G)
    if params.beta is None:
        params.beta = 0.75 * params.beta_max

    numerator = (2 / params.beta * kT * params.gam1 * (1 + params.gam3 / params.gam1 * params.gm3_gm1) +
                 1 / params.cltot_cc * kT * (1 + params.gam2 * (1 + params.gam4 / params.gam2 * params.gm4_gm2)))

    circuit.cc = numerator / (specs.vod_noise ** 2)
    circuit.cltot = circuit.cc * params.cltot_cc
    circuit.cf = circuit.cltot / ((1 + params.rself2) * (1 - params.beta + specs.FO * specs.G))
    circuit.cs = circuit.cf * specs.G
    circuit.cl = circuit.cs * specs.FO

    m2.cgs = circuit.cc * params.cgs2_cc
    circuit.c1 = m2.cgs * (1 + params.rself1)

    return circuit

# Placeholder functions requiring external model access
def calculate_m1_parameters(m1: Transistor, specs: Specs, params: DesignParams, circuit: Circuit) -> Transistor:
    """Calculate M1 parameters 
    
    Args:
        m1: Pre-initialized transistor with L and PDK parameters
        specs: Design specifications
        params: Design parameters
        circuit: Calculated circuit components
        
    Returns:
        Configured M1 transistor with all calculated parameters
    """
    # Calculate gmR 
    gmR = np.sqrt(specs.L0) / params.beta #possivelmente os valores de teste sao demasiado elevados (rever)
    # Transconductance calculation 
    m1.gm = (2 * np.pi * specs.fu1 * circuit.cc / params.beta * 
            (1 + (1 + circuit.c1/circuit.cc)/gmR + 
             (1 + circuit.cltot/circuit.cc)/gmR))
    
    # Gate capacitance 
    m1.cgg = circuit.cf * ((1/params.beta) - 1 - specs.G)
    
    # Transition frequency
    m1.ft = m1.gm / m1.cgg / (2 * np.pi)
    
    # gm/ID lookup 
    m1.gm_id=pmos.interpolate(
        x_expression=pmos.length_expression,
        x_value=m1.L,
        y_expression = Expression(
        variables=["gm", "cgg"],
        function=lambda x, y: x / y,
        ),
        y_value=m1.gm/m1.cgg,
        z_expression=pmos.gmid_expression
    ).item() 

    # Drain current
    m1.id = m1.gm / m1.gm_id
    
    return m1

def calculate_m2_parameters(m2: Transistor, specs: Specs, params: DesignParams, circuit: Circuit) -> Transistor:
    """Calculate M2 parameters 
    
    Args:
        m2: Pre-initialized transistor with L and cgs parameters
        specs: Design specifications
        params: Design parameters
        circuit: Calculated circuit components
        
    Returns:
        Configured M2 transistor with all calculated parameters
    """
    # 1. Transconductance calculation 
    m2.gm = (2 * np.pi * specs.fp2 * 
            (1 + circuit.cltot/circuit.cc + circuit.cltot/circuit.c1) * 
            circuit.c1)
    
    # 2. Transition frequency 
    m2.ft = m2.gm / m2.cgs / (2 * np.pi)
    
    # 3. gm/ID lookup 
    m2.gm_id=nmos.interpolate(
        x_expression=nmos.length_expression,
        x_value=m2.L,
        y_expression = Expression(
        variables=["gm", "cgg"],#Rever isto 
        function=lambda x, y: x / (y),
        ),
        y_value=m2.gm/m2.cgs,
        z_expression=nmos.gmid_expression
    ).item()
    
    # 4. Drain current
    m2.id = m2.gm / m2.gm_id
    
    # 5. Nulling resistor (MATLAB: p.rz = 1./m2.gm)
    circuit.rz = 1 / m2.gm
    
    return m2
  
def calculate_transistor_widths(m1: Transistor, m2: Transistor, 
                               m3: Transistor, m4: Transistor,
                               params: DesignParams) -> None:
    """Calculate transistor widths 
    
    Args:
        m1: Input pair transistor
        m2: Second stage transistor
        m3: PMOS mirror for M1
        m4: PMOS mirror for M2
        params: Design parameters with gm ratios
    """
    # 1. Set current mirror gm/ID ratios 
    m3.gm_id = m1.gm_id * params.gm3_gm1  
    m4.gm_id = m2.gm_id * params.gm4_gm2  
    
    # 2. Calculate widths using current density (ID/W)
    m1.w = m1.id / pmos.interpolate(
        x_expression=pmos.length_expression,
        x_value=m1.L,
        y_expression=pmos.gmid_expression,
        y_value=m1.gm_id,
        z_expression=pmos.current_density_expression
    ).item() 

    m2.w = m2.id / nmos.interpolate(
        x_expression=nmos.length_expression,
        x_value=m2.L,
        y_expression=nmos.gmid_expression,
        y_value=m2.gm_id,
        z_expression=nmos.current_density_expression
    ).item() 

    # 3. Mirror devices inherit currents but have their own gm/ID
    m3.w = m1.id / nmos.interpolate(
        x_expression=nmos.length_expression,
        x_value=m3.L,
        y_expression=nmos.gmid_expression,
        y_value=m3.gm_id,
        z_expression=nmos.current_density_expression
    ).item() 

    m4.w = m2.id / pmos.interpolate(
        x_expression=pmos.length_expression,
        x_value=m4.L,
        y_expression=pmos.gmid_expression,
        y_value=m4.gm_id,
        z_expression=pmos.current_density_expression
    ).item() 
    
    
def calculate_compensation_caps(m1: Transistor, m2: Transistor, circuit: Circuit) -> None:
    """Calculate neutralization and compensation caps
    
    Args:
        m1: Input pair transistor
        m2: Second stage transistor
        circuit: Circuit components (updates cn and cc_add)
    """
    # 1. M1 gate-drain capacitance (neutralization cap)
    m1.cgd =m1.w*pmos.interpolate(
        x_expression=pmos.length_expression,
        x_value=m1.L,
        y_expression=pmos.gmid_expression,
        y_value=m1.gm_id,
        z_expression=Expression(
            variables=["cgd", "weff"],
            function=lambda x, y: x / (y),
        ),
    ).item()
    
    # 2. Set neutralization cap 
    circuit.cn = m1.cgd
    
    # 3. M2 gate-drain capacitance
    m2.cgd =m2.w*nmos.interpolate(
        x_expression=nmos.length_expression,
        x_value=m2.L,
        y_expression=nmos.gmid_expression,
        y_value=m2.gm_id,
        z_expression=Expression(
            variables=["cgd", "weff"],
            function=lambda x, y: x / (y),
        ),
    ).item() 
    
    # 4. Additional compensation cap 
    circuit.cc_add = circuit.cc - m2.cgd
    
def calculate_self_loading(m1: Transistor, m2: Transistor, 
                         m3: Transistor, m4: Transistor,
                         circuit: Circuit, params: DesignParams) -> None:
    """Calculate self-loading effects 
    
    Args:
        m1-m4: All transistors with calculated widths
        circuit: Circuit components (updates rself1/rself2)
        params: Design parameters (beta)
    """
    # 1. Drain capacitances
    m1.cdd = m1.w*pmos.interpolate(
        x_expression=pmos.length_expression,
        x_value=m1.L,
        y_expression=pmos.gmid_expression,
        y_value=m1.gm_id,
        z_expression=Expression(
            variables=["cdd", "weff"],
            function=lambda x, y: x / (y),
        ),
    ).item()
    
    m2.cdd = m2.w*nmos.interpolate(
        x_expression=nmos.length_expression,
        x_value=m2.L,
        y_expression=nmos.gmid_expression,
        y_value=m2.gm_id,
        z_expression=Expression(
            variables=["cdd", "weff"],
            function=lambda x, y: x / (y),
        ),
    ).item()
    
    m3.cdd = m3.w*nmos.interpolate(
        x_expression=nmos.length_expression,
        x_value=m3.L,
        y_expression=nmos.gmid_expression,
        y_value=m3.gm_id,
        z_expression=Expression(
            variables=["cdd", "weff"],
            function=lambda x, y: x / (y),
        ),
    ).item() 
    
    m4.cdd = m4.w*pmos.interpolate(
        x_expression=pmos.length_expression,
        x_value=m4.L,
        y_expression=pmos.gmid_expression,
        y_value=m4.gm_id,
        z_expression=Expression(
            variables=["cdd", "weff"],
            function=lambda x, y: x / (y),
        ),
    ).item() 
    
    # 2. Stage 1 self-loading 
    circuit.rself1 = (m1.cdd + m3.cdd) / m2.cgs
    
    # 3. Stage 2 self-loading 
    numerator = ((m2.cdd - m2.cgd) + m4.cdd)
    denominator = (circuit.cl + (1 - params.beta) * circuit.cf)
    circuit.rself2 = numerator / denominator
    
def run_full_opamp_design(
    specs: Specs,
    params: DesignParams,
    *,
    m1: Transistor | None = None,
    m2: Transistor | None = None,
    m3: Transistor | None = None,
    m4: Transistor | None = None,
    verbose: bool = True,
):
    """
    End-to-end op-amp design helper that executes every major
    calculation step and gathers the results.

    Returns
    -------
    dict
        {
            "circuit":  CircuitCaps,
            "m1":       Transistor,
            "m2":       Transistor,
            "m3":       Transistor,
            "m4":       Transistor,
        }
    """
    m1 = m1 or Transistor(L=150e-9)   # input pair
    m2 = m2 or Transistor(L=200e-9)   # second stage
    m3 = m3 or Transistor(L=200e-9)
    m4 = m4 or Transistor(L=150e-9)

    circuit = calculate_capacitances(specs, params, m2)
    
    m1 = calculate_m1_parameters(m1, specs, params, circuit)
    m2 = calculate_m2_parameters(m2, specs, params, circuit)
    
    calculate_transistor_widths(m1, m2, m3, m4, params)
    calculate_compensation_caps(m1, m2, circuit)
    calculate_self_loading(m1, m2, m3, m4, circuit, params)

    # ------------------------------------------------------------------
    # Optional console read-out (mirrors your test bench prints)
    # ------------------------------------------------------------------
    if verbose:
        print("=== Op-amp Design Summary ===")
        print(f"β (beta):              {params.beta:.3f}")
        print(f"CC:                    {circuit.cc:.3e} F")
        print(f"CLtot:                 {circuit.cltot:.3e} F")
        print(f"CF:                    {circuit.cf:.3e} F")
        print(f"CS:                    {circuit.cs:.3e} F")
        print(f"C1:                    {circuit.c1:.3e} F")
        print(f"Zero-nulling Rz:       {circuit.rz:,.1f} Ω")
        print()
        print("— Widths (µm) —")
        print(f"M1: {m1.w*1e6:.3f} | M2: {m2.w*1e6:.3f} "
              f"| M3: {m3.w*1e6:.3f} | M4: {m4.w*1e6:.3f}")
        print()
        print("— Compensation —")
        print(f"CN (neutralization):   {circuit.cn*1e15:.2f} fF")
        print(f"ΔCC (extra):           {circuit.cc_add*1e12:.3f} pF")
        print()
        print("— Self-loading —")
        print(f"Stage 1 (actual / target): "
              f"{circuit.rself1:.3f} / {params.rself1}")
        print(f"Stage 2 (actual / target): "
              f"{-circuit.rself2:.3f} / {params.rself2}")
        print("==============================\n")

    # ------------------------------------------------------------------
    # 7. Return everything so the caller can poke around programmatically
    # ------------------------------------------------------------------
    return {
        "circuit": circuit,
        "m1": m1,
        "m2": m2,
        "m3": m3,
        "m4": m4,
    }

