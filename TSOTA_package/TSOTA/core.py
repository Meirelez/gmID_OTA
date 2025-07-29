import numpy as np
from scipy import constants
from dataclasses import dataclass
from typing import Optional
from mosplot.plot import Expression, Mosfet, load_lookup_table
import ipywidgets as widgets
from IPython.display import display
import re

# Constants
k = constants.Boltzmann
TEMP = 300  # Kelvin
_last_design_results = None 

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
        z_expression=pmos.gmid_expression,
        fast=True
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
        z_expression=nmos.gmid_expression,
        fast=True
    ).item()
    
    # 4. Drain current
    m2.id = m2.gm / m2.gm_id
    
    # 5. Nulling resistor 
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
        z_expression=pmos.current_density_expression,
        fast=True
    ).item() 

    m2.w = m2.id / nmos.interpolate(
        x_expression=nmos.length_expression,
        x_value=m2.L,
        y_expression=nmos.gmid_expression,
        y_value=m2.gm_id,
        z_expression=nmos.current_density_expression,
        fast=True
    ).item() 

    # 3. Mirror devices inherit currents but have their own gm/ID
    m3.w = m1.id / nmos.interpolate(
        x_expression=nmos.length_expression,
        x_value=m3.L,
        y_expression=nmos.gmid_expression,
        y_value=m3.gm_id,
        z_expression=nmos.current_density_expression,
        fast=True
    ).item() 

    m4.w = m2.id / pmos.interpolate(
        x_expression=pmos.length_expression,
        x_value=m4.L,
        y_expression=pmos.gmid_expression,
        y_value=m4.gm_id,
        z_expression=pmos.current_density_expression,
        fast=True
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
        fast=True
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
        fast=True
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
        fast=True
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
        fast=True
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
        fast=True
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
        fast=True
    ).item() 
    
    # 2. Stage 1 self-loading 
    circuit.rself1 = (m1.cdd + m3.cdd) / m2.cgs
    
    # 3. Stage 2 self-loading 
    numerator = abs(((m2.cdd - m2.cgd) + m4.cdd))
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
        print(f"CL:                    {circuit.cl:.3e} F")
        print(f"Zero-nulling Rz:       {circuit.rz:,.1f} Ω")
        print()
        print("— Widths (µm) —")
        print(f"M1: {m1.w*1e6:.3f} | M2: {m2.w*1e6:.3f} "
              f"| M3: {m3.w*1e6:.3f} | M4: {m4.w*1e6:.3f}")
        print("— Gm/ID (µm) —")
        print(f"M1: {m1.gm_id:.3f} | M2: {m2.gm_id:.3f} "
              f"| M3: {m3.gm_id:.3f} | M4: {m4.gm_id:.3f}")
        print("— ID (µm) —")
        print(f"ID1:              {m1.id*1e6:.3f}")
        print()
        print("— Compensation —")
        print(f"CN :            {circuit.cn*1e15:.2f} fF")
        print(f"Cadd:           {circuit.cc_add*1e12:.3f} pF")
        print(f"Cgd2:           {m2.cgd*1e12:.3f} pF")
        print()
        print("— Self-loading —")
        print(f"Stage 1 (actual / target):"
              f"{circuit.rself1:.3f} / {params.rself1}")
        print(f"Stage 2 (actual / target): "
              f"{circuit.rself2:.3f} / {params.rself2}")
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

def optimize_opamp_design(
    specs: Specs,
    base_params: DesignParams = None,
    target_cgs2_cc: float = 0.29,
    optimization_level: str = "balanced",
    verbose: bool = False
):
    """
    Complete automatic op-amp optimization function.
    Runs the full optimization and returns the final optimized design.
    
    Args:
        specs (Specs): Design specifications
        base_params (DesignParams, optional): Base design parameters. Uses defaults if None.
        target_cgs2_cc (float): Target cgs2_cc value for final design selection (default: 0.29)
        optimization_level (str): Optimization thoroughness
            - "fast": Quick optimization 
            - "balanced": Good balance of speed/accuracy 
            - "thorough": Comprehensive search 
            - "ultra": Maximum accuracy
        verbose (bool): Print detailed progress and results
    
    Returns:
        dict: Complete results containing:
            - 'final_design': Final optimized circuit results from run_full_opamp_design()
            - 'optimal_params': Optimal DesignParams object
            - 'optimization_data': Full optimization results (for analysis/plotting)
            - 'summary': Summary statistics
    """
    
    if verbose:
        print("="*60)
        print(" AUTOMATIC OP-AMP OPTIMIZATION")
        print("="*60)
        print(f"Specifications:")
        print(f"  • fu1 = {specs.fu1/1e6:.0f} MHz")
        print(f"  • fp2 = {specs.fp2/1e6:.0f} MHz") 
        print(f"  • G = {specs.G}")
        print(f"  • vod_noise = {specs.vod_noise*1e6:.0f} µV")
        print(f"  • FO = {specs.FO}")
        print(f"Optimization level: {optimization_level}")
        print(f"Target cgs2_cc: {target_cgs2_cc}")
        print("-"*60)
    
    # Set optimization parameters based on level
    optimization_configs = {
        "fast": {
            "cltot_cc_points": 20,
            "beta_points": 20, 
            "cgs2_cc_points": 6,
            "self_loading_iterations": 2
        },
        "balanced": {
            "cltot_cc_points": 30,
            "beta_points": 30,
            "cgs2_cc_points": 8, 
            "self_loading_iterations": 3
        },
        "thorough": {
            "cltot_cc_points": 50,
            "beta_points": 50,
            "cgs2_cc_points": 12,
            "self_loading_iterations": 4
        },
        "ultra": {
            "cltot_cc_points": 80,
            "beta_points": 80,
            "cgs2_cc_points": 16,
            "self_loading_iterations": 5
        }
    }
    
    if optimization_level not in optimization_configs:
        raise ValueError(f"optimization_level must be one of: {list(optimization_configs.keys())}")
    
    config = optimization_configs[optimization_level]
    
    try:
        # Step 1: Run full optimization
        if verbose:
            print("Running optimization...")
        
        optimization_data = find_minimum_current_design_fast(
            specs=specs,
            base_params=base_params,
            **config,
            verbose_internal=verbose
        )
        
        # Step 2: Extract optimal design
        if verbose:
            print("\nExtracting optimal design...")
        
        final_design, optimal_params = get_final_design_from_results(
            optimization_data, 
            target_cgs2_cc=target_cgs2_cc
        )
        
        # Step 3: Calculate summary statistics
        min_current_idx = np.argmin(optimization_data['IDtot_k'])
        absolute_minimum = optimization_data['IDtot_k'][min_current_idx]
        selected_current = optimization_data['IDtot_k'][np.argmax(optimization_data['cgs2_cc'] >= target_cgs2_cc)]
        
        summary = {
            'absolute_minimum_current_uA': absolute_minimum * 1e6,
            'absolute_minimum_cgs2_cc': optimization_data['cgs2_cc'][min_current_idx],
            'selected_current_uA': selected_current * 1e6,
            'selected_cgs2_cc': target_cgs2_cc,
            'current_penalty_percent': ((selected_current - absolute_minimum) / absolute_minimum) * 100,
            'total_evaluations': len(optimization_data['cgs2_cc']) * config['cltot_cc_points'] * config['beta_points'],
            'beta_max': optimization_data['beta_max'],
            'optimal_beta_ratio': optimal_params.beta / optimization_data['beta_max']
        }
        
        # Step 4: Print results summary
        if verbose:
            print("\n OPTIMAL PARAMETERS:")
            print(f"  • β = {optimal_params.beta:.3f}")
            print(f"  • cltot_cc = {optimal_params.cltot_cc:.3f}")
            print(f"  • cgs2_cc = {optimal_params.cgs2_cc:.3f}")
            print(f"  • rself1 = {optimal_params.rself1:.3f}")
            print(f"  • rself2 = {optimal_params.rself2:.3f}")
            print("="*60)
        
        # Return everything
        return {
            'final_design': final_design,
            'optimal_params': optimal_params,
            'optimization_data': optimization_data,
            'summary': summary
        }
        
    except Exception as e:
        if verbose:
            print(f"\nOPTIMIZATION FAILED: {str(e)}")
        raise

def find_minimum_current_design_fast(
    specs: Specs,
    base_params: DesignParams = None,
    cltot_cc_points: int = 50,
    beta_points: int = 50,
    cgs2_cc_points: int = 10,
    self_loading_iterations: int = 3,
    verbose_internal: bool = False
):
    """
    Internal optimization function (modified to support the auto function above).
    Added verbose_internal parameter to control internal printing.
    """
    
    beta_max = 1 / (1 + specs.G)
    
    # Use provided base_params or create defaults
    if base_params is None:
        base_params = DesignParams(
            L1=0.15e-6, L2=0.2e-6, L3=0.2e-6, L4=0.15e-6,
            gam1=0.8, gam2=0.8, gam3=0.8, gam4=0.8,
            gm3_gm1=1.0, gm4_gm2=0.5
        )
    
    # Search ranges
    cltot_cc_range = np.linspace(0.2, 1.5, cltot_cc_points)
    beta_range = beta_max * np.linspace(0.4, 0.95, beta_points)
    cgs2_cc_range = np.linspace(0.2, 0.5, cgs2_cc_points)
    
    # Storage for results
    IDtot_k = []
    ID1_opt_k = []
    ID2_opt_k = []
    beta_opt_k = []
    cltot_cc_opt_k = []
    rself1_k = []
    rself2_k = []
    
    if verbose_internal:
        print(f"Search space: {cgs2_cc_points} × {cltot_cc_points} × {beta_points} = {cgs2_cc_points * cltot_cc_points * beta_points:,} evaluations")
    
    for k, cgs2_cc_val in enumerate(cgs2_cc_range):
        if verbose_internal:
            print(f"Processing cgs2_cc = {cgs2_cc_val:.3f} ({k+1}/{len(cgs2_cc_range)})")
        
        # Initialize self-loading
        rself1 = 0.0
        rself2 = 0.0
        
        # Self-loading convergence loop
        for iteration in range(self_loading_iterations):
            ID1 = np.full((len(cltot_cc_range), len(beta_range)), 1e-3)
            ID2 = np.full((len(cltot_cc_range), len(beta_range)), 1e-3)
            rself1_out = np.zeros((len(cltot_cc_range), len(beta_range)))
            rself2_out = np.zeros((len(cltot_cc_range), len(beta_range)))
            
            successful_designs = 0
            total_designs = len(cltot_cc_range) * len(beta_range)
            
            # Parameter sweep
            for i, cltot_cc_val in enumerate(cltot_cc_range):
                for j, beta_val in enumerate(beta_range):
                    params = DesignParams(
                        beta=beta_val, beta_max=beta_max,
                        rself1=rself1, rself2=rself2,
                        cgs2_cc=cgs2_cc_val, cltot_cc=cltot_cc_val,
                        gm3_gm1=base_params.gm3_gm1, gm4_gm2=base_params.gm4_gm2,
                        gam1=base_params.gam1, gam2=base_params.gam2,
                        gam3=base_params.gam3, gam4=base_params.gam4,
                        L1=base_params.L1, L2=base_params.L2,
                        L3=base_params.L3, L4=base_params.L4
                    )
                    
                    # Create transistors
                    m1 = Transistor(L=params.L1)
                    m2 = Transistor(L=params.L2)
                    m3 = Transistor(L=params.L3)
                    m4 = Transistor(L=params.L4)
                    
                    try:
                        # Run design
                        results = run_full_opamp_design(
                            specs, params,
                            m1=m1, m2=m2, m3=m3, m4=m4,
                            verbose=False
                        )
                        
                        # Store successful results
                        ID1[i, j] = results["m1"].id
                        ID2[i, j] = results["m2"].id
                        rself1_out[i, j] = results["circuit"].rself1
                        rself2_out[i, j] = results["circuit"].rself2
                        successful_designs += 1
                        
                    except Exception:
                        rself1_out[i, j] = rself1
                        rself2_out[i, j] = rself2
            
            if verbose_internal:
                print(f"Iteration {iteration+1}: {successful_designs}/{total_designs} successful designs")
            
            # Find optimum
            IDtot = ID1 + ID2
            min_idx = np.unravel_index(np.argmin(IDtot), IDtot.shape)
            idx1, idx2 = min_idx
            
            # Update self-loading
            new_rself1 = rself1_out[idx1, idx2]
            new_rself2 = rself2_out[idx1, idx2]
            
            # Check convergence
            if (abs(new_rself1 - rself1) < 0.01 and 
                abs(new_rself2 - rself2) < 0.01):
                if verbose_internal:
                    print(f" Converged after {iteration+1} iterations")
                break
                
            rself1 = new_rself1
            rself2 = new_rself2
        
        # Store results for this cgs2_cc
        IDtot_opt = IDtot[idx1, idx2]
        IDtot_k.append(IDtot_opt)
        ID1_opt_k.append(ID1[idx1, idx2])
        ID2_opt_k.append(ID2[idx1, idx2])
        beta_opt_k.append(beta_range[idx2])
        cltot_cc_opt_k.append(cltot_cc_range[idx1])
        rself1_k.append(rself1)
        rself2_k.append(rself2)
        
        if verbose_internal:
            print(f"  Optimal IDtot: {IDtot_opt*1e6:.1f} µA")
    
    # Package results
    results_dict = {
        'IDtot_k': np.array(IDtot_k),
        'ID1_opt_k': np.array(ID1_opt_k),
        'ID2_opt_k': np.array(ID2_opt_k),
        'beta_opt_k': np.array(beta_opt_k),
        'cltot_cc_opt_k': np.array(cltot_cc_opt_k),
        'cgs2_cc': cgs2_cc_range,
        'rself1_k': np.array(rself1_k),
        'rself2_k': np.array(rself2_k),
        'beta_max': beta_max,
        'specs': specs
    }
    
    return results_dict

def get_final_design_from_results(results_dict, target_cgs2_cc=0.29):
    """
    Extract final design from optimization results (used internally by optimize_opamp_design).
    """
    # Extract specs from results
    specs = results_dict['specs']
    
    # Find index closest to target cgs2_cc
    cgs2_cc = results_dict['cgs2_cc']
    idx = np.argmax(cgs2_cc >= target_cgs2_cc)
    
    # Extract optimal parameters for final design
    final_params = DesignParams(
        cltot_cc=results_dict['cltot_cc_opt_k'][idx],
        beta=results_dict['beta_opt_k'][idx],
        rself1=results_dict['rself1_k'][idx],
        rself2=results_dict['rself2_k'][idx],
        cgs2_cc=cgs2_cc[idx],
        gm3_gm1=1.0, gm4_gm2=0.5,
        gam1=0.8, gam2=0.8, gam3=0.8, gam4=0.8,
        L1=0.15e-6, L2=0.2e-6, L3=0.2e-6, L4=0.15e-6
    )
    
    # Run final design
    final_results = run_full_opamp_design(
        specs, final_params,
        m1=Transistor(L=final_params.L1),
        m2=Transistor(L=final_params.L2),
        m3=Transistor(L=final_params.L3),
        m4=Transistor(L=final_params.L4),
        verbose=True  # Controlled by main function
    )
    
    return final_results, final_params



###UI-PART ##################      
def launch_ui():
    # Create an output widget to capture prints
    output = widgets.Output()
    
    # Specs input widgets
    ts = widgets.FloatText(value=5e-9, description="ts")
    fu1 = widgets.FloatText(value=220e6, description="fu1")
    fp2 = widgets.FloatText(value=6*220e6, description="fp2")
    vod_noise = widgets.FloatText(value=400e-6, description="vod_noise")
    G = widgets.FloatText(value=2.0, description="G")
    FO = widgets.FloatText(value=0.5, description="FO")
    L0 = widgets.FloatText(value=50, description="L0")
    
    # Button to trigger design
    button = widgets.Button(description="Run Design")
    
    # Layout for input fields
    specs_box = widgets.VBox([ts, fu1, fp2, vod_noise, G, FO, L0])
    
    # Display UI
    display(specs_box, button, output)

    def on_button_click(b):
        output.clear_output()  # Clear previous outputs
        with output:
            specs = Specs(
                ts=ts.value,
                fu1=fu1.value,
                fp2=fp2.value,
                vod_noise=vod_noise.value,
                G=G.value,
                FO=FO.value,
                L0=L0.value
            )

            params = DesignParams(
                rself1=0.4, rself2=0.4,
                cgs2_cc=1/3, cltot_cc=1.0,
                gm3_gm1=1.0, gm4_gm2=0.5,
                slew_frac=0.3,
                gam1=0.8, gam2=0.8, gam3=0.8, gam4=0.8
            )

            try:
                print("Running design...\n")
                global _last_design_results
                _last_design_results = optimize_opamp_design(specs,params,optimization_level="fast", verbose=True)
            except Exception as e:
                print(f"Error: {e}")

    button.on_click(on_button_click)
###NETLIST-PART ##################          
def update_scs_file_lw(filepath: str, output_path: str, replacements: dict) -> None:
    """
    Update the L and W values of MOSFETs inside TSOTA-tagged blocks in a Spectre .scs file.

    Args:
        filepath (str): Path to the original .scs file.
        output_path (str): Path to save the modified file.
        replacements (dict): Dictionary like {'M1': {'w': W_in_meters, 'l': L_in_meters}, ...}
    """
    with open(filepath, "r") as f:
        lines = f.readlines()

    inside_block = None
    updated_lines = []

    for line in lines:
        block_start = re.match(r"\s*// === TSOTA_(M\d) ===", line)
        block_end = re.match(r"\s*// === END_TSOTA_(M\d) ===", line)

        if block_start:
            inside_block = block_start.group(1)
            updated_lines.append(line)
            continue
        elif block_end:
            inside_block = None
            updated_lines.append(line)
            continue

        if inside_block and inside_block in replacements:
            mos = replacements[inside_block]
            # Replace w and l values in microns and nanometers, respectively
            line = re.sub(r"w=\s*[\d.eE+-]+", f"w={mos['w'] * 1e6:.4g}", line)
            line = re.sub(r"l=\s*[\d.eE+-]+", f"l={mos['l'] * 1e9:.4g}", line)

        updated_lines.append(line)

    with open(output_path, "w") as f:
        f.writelines(updated_lines)

def create_netlist(input_path: str, output_path: str, results: dict = None) -> None:
    """
    Updates a Spectre netlist (.scs) file with W/L values from the op-amp design results.

    Args:
        input_path (str): Path to the source .scs netlist file.
        output_path (str): Path to save the modified netlist file.
        results (dict, optional): Dictionary from `run_full_opamp_design()`. 
                                  If None, uses the most recent results.
    """
    global _last_design_results
    if results is None:
        if _last_design_results is None:
            raise ValueError("No design results available. Run the design first.")
        results = _last_design_results

    update_scs_file_lw(
        filepath=input_path,
        output_path=output_path,
        replacements={
            "M1": {"w": results["m1"].w, "l": results["m1"].L},
            "M2": {"w": results["m2"].w, "l": results["m2"].L},
            "M3": {"w": results["m3"].w, "l": results["m3"].L},
            "M4": {"w": results["m4"].w, "l": results["m4"].L},
        }
    )

def launch_ui2():
    """
    Launch interactive UI for op-amp optimization with all options.
    """
    # Create an output widget to capture prints
    output = widgets.Output()
    
    # Specs input widgets
    ts = widgets.FloatText(value=5e-9, description="ts (s):", style={'description_width': 'initial'})
    fu1 = widgets.FloatText(value=220e6, description="fu1 (Hz):", style={'description_width': 'initial'})
    fp2 = widgets.FloatText(value=6*220e6, description="fp2 (Hz):", style={'description_width': 'initial'})
    vod_noise = widgets.FloatText(value=400e-6, description="vod_noise (V):", style={'description_width': 'initial'})
    G = widgets.FloatText(value=2.0, description="G:", style={'description_width': 'initial'})
    FO = widgets.FloatText(value=0.5, description="FO:", style={'description_width': 'initial'})
    L0 = widgets.FloatText(value=50, description="L0:", style={'description_width': 'initial'})
    
    # Optimization level dropdown
    optimization_level = widgets.Dropdown(
        options=[
            ('Fast (~5-10 min)', 'fast'),
            ('Balanced (~10-20 min)', 'balanced'), 
            ('Thorough (~20-40 min)', 'thorough'),
            ('Ultra (~40+ min)', 'ultra')
        ],
        value='balanced',
        description='Optimization:',
        style={'description_width': 'initial'}
    )
    
    # Target cgs2_cc selector
    target_cgs2_cc = widgets.FloatSlider(
        value=0.29,
        min=0.2,
        max=0.5,
        step=0.01,
        description='Target cgs2_cc:',
        style={'description_width': 'initial'},
        readout_format='.2f'
    )
    
    # Verbose output checkbox
    verbose_checkbox = widgets.Checkbox(
        value=True,
        description='Show detailed output',
        style={'description_width': 'initial'}
    )
    
    # Buttons
    optimize_button = widgets.Button(
        description="🚀 Run Optimization",
        button_style='success',
        layout=widgets.Layout(width='200px')
    )
    
    clear_button = widgets.Button(
        description="Clear Output",
        button_style='warning',
        layout=widgets.Layout(width='150px')
    )
    
    # Layout sections
    specs_section = widgets.VBox([
        widgets.HTML("<h3>📊 Design Specifications</h3>"),
        widgets.HBox([ts, fu1]),
        widgets.HBox([fp2, vod_noise]),
        widgets.HBox([G, FO]),
        L0
    ])
    
    optimization_section = widgets.VBox([
        widgets.HTML("<h3>⚙️ Optimization Settings</h3>"),
        optimization_level,
        target_cgs2_cc,
        verbose_checkbox
    ])
    
    buttons_section = widgets.HBox([
        optimize_button,
        clear_button
    ])
    
    # Main layout
    ui_layout = widgets.VBox([
        widgets.HTML("<h2>🔧 Automatic Op-Amp Optimizer</h2>"),
        specs_section,
        optimization_section,
        buttons_section,
        output
    ])
    
    # Display UI
    display(ui_layout)
    
    def on_optimize_click(b):
        """Handle optimization button click"""
        output.clear_output()
        
        # Disable button during optimization
        optimize_button.disabled = True
        optimize_button.description = "⏳ Optimizing..."
        
        with output:
            try:
                # Create specs from UI inputs
                specs = Specs(
                    ts=ts.value,
                    fu1=fu1.value,
                    fp2=fp2.value,
                    vod_noise=vod_noise.value,
                    G=G.value,
                    FO=FO.value,
                    L0=L0.value
                )
                
                print(f"🎯 Starting optimization with level: {optimization_level.value}")
                print(f"🎯 Target cgs2_cc: {target_cgs2_cc.value}")
                print("-" * 60)
                
                # Run optimization using the main function
                global _last_design_results
                _last_design_results = optimize_opamp_design(
                    specs=specs,
                    base_params=None,  # Use defaults
                    target_cgs2_cc=target_cgs2_cc.value,
                    optimization_level=optimization_level.value,
                    verbose=verbose_checkbox.value
                )
                
                # Extract results for easy access
                final_design = _last_design_results['final_design']
                optimal_params = _last_design_results['optimal_params']
                summary = _last_design_results['summary']
                
                print("\n" + "=" * 60)
                print("✅ OPTIMIZATION COMPLETE!")
                print("=" * 60)
                print(f"📊 Total current: {summary['selected_current_uA']:.1f} µA")
                print(f"🔧 M1 width: {final_design['m1'].w*1e6:.2f} µm")
                print(f"🔧 M2 width: {final_design['m2'].w*1e6:.2f} µm")
                print(f"🔧 M3 width: {final_design['m3'].w*1e6:.2f} µm")
                print(f"🔧 M4 width: {final_design['m4'].w*1e6:.2f} µm")
                print(f"⚡ Compensation cap: {final_design['circuit'].cc*1e12:.2f} pF")
                print(f"🎯 Nulling resistor: {final_design['circuit'].rz/1000:.1f} kΩ")
                print("=" * 60)
                
                print("\n💡 Use create_netlist('input.scs', 'output.scs') to update your netlist!")
                
            except Exception as e:
                print(f"❌ ERROR: {str(e)}")
                print("\n🔍 Please check your input values and try again.")
                
        # Re-enable button
        optimize_button.disabled = False
        optimize_button.description = "🚀 Run Optimization"
    
    def on_clear_click(b):
        """Handle clear button click"""
        output.clear_output()
    
    # Connect button events
    optimize_button.on_click(on_optimize_click)
    clear_button.on_click(on_clear_click)

