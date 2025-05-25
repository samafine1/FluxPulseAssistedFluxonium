import numpy as np
import qutip as qtp
import matplotlib.pyplot as plt
from scipy.constants import h, e, k # h (Planck), e (electron charge), k (Boltzmann)

# %% --- 0. Constants and Utility Functions ---
hbar = h / (2 * np.pi)
Phi0 = h / (2 * e)  # Magnetic flux quantum

def get_basic_ops(N_levels):
    """
    Returns creation, annihilation, number, identity operators
    for a given number of levels.
    """
    a_op = qtp.destroy(N_levels)
    adag_op = qtp.create(N_levels)
    n_op = qtp.num(N_levels) # n_op = adag_op * a_op
    id_op = qtp.qeye(N_levels)
    return a_op, adag_op, n_op, id_op

# %% --- 1. Hamiltonian Component Functions ---

def hamiltonian_fluxonium(N_q_levels, Ec, Ej, El, phi_ext_value, return_ops=False):
    """
    Constructs the Fluxonium qubit Hamiltonian using harmonic oscillator basis.
    Energies Ec, Ej, El should be in angular frequency units (e.g., 2*pi*GHz).
    phi_ext_value is dimensionless (phi_ext = 2*pi*Phi/Phi0).
    """
    # phi_0_osc is the zero-point fluctuation of phi in the LC oscillator basis
    phi_0_osc = (8 * Ec / El)**0.25 
    
    c_q = qtp.destroy(N_q_levels) # Annihilation operator for fluxonium's LC mode
    cdag_q = qtp.create(N_q_levels)

    phi_op = (phi_0_osc / np.sqrt(2)) * (c_q + cdag_q)
    # n_op is the charge operator conjugate to phi. n = -i * d/dphi
    # In the HO basis, n_op = (1j / (phi_0_osc * np.sqrt(2))) * (cdag_q - c_q)
    # Note: This n_op is dimensionless. For coupling, physical charge is e * n_op.
    # The 4Ec n^2 term implies n is the number of Cooper pairs.
    # The charge operator used for coupling is often scaled differently.
    # Let's use the standard definition for n in 4Ec n^2
    n_op_fluxonium_charge = (1j / (phi_0_osc * np.sqrt(2))) * (cdag_q - c_q)

    H_C = 4 * Ec * (n_op_fluxonium_charge**2)
    H_L = 0.5 * El * (phi_op**2)
    
    # Cosine term: Ensure phi_op and phi_ext_value are correctly combined
    # phi_ext_value is a scalar here.
    cos_arg = phi_op - phi_ext_value * qtp.qeye(N_q_levels)
    H_J = -Ej * cos_arg.cosm() 
    
    H_fluxonium = H_C + H_L + H_J
    
    if return_ops:
        return H_fluxonium, phi_op, n_op_fluxonium_charge, phi_0_osc
    else:
        return H_fluxonium

def hamiltonian_resonator(N_r_levels, omega_r):
    """omega_r in angular frequency units."""
    a_r, _, n_r, _ = get_basic_ops(N_r_levels)
    H_resonator = omega_r * n_r
    return H_resonator

def hamiltonian_jja_mode(N_jja_levels, omega_jja):
    """omega_jja in angular frequency units."""
    b_jja, _, n_jja, _ = get_basic_ops(N_jja_levels)
    H_jja = omega_jja * n_jja
    return H_jja

def hamiltonian_tls(omega_tls):
    """omega_tls in angular frequency units. TLS is 2-level."""
    H_tls = (omega_tls / 2.0) * qtp.sigmaz()
    return H_tls

# %% --- 2. System Parameters (Adjust these carefully!) ---

# Hilbert Space Dimensions
N_q = 10         # Qubit levels (fluxonium)
N_r = 7          # Resonator levels
N_jja = 3        # JJA mode levels (0, 1, 2 excitations)
N_tls = 2        # TLS levels (is always 2)

# Fluxonium Parameters (energies in 2*pi*GHz, i.e., angular frequencies)
Ec_q_GHz = 1.5
Ej_q_GHz = 3.0
El_q_GHz = 0.3    # For "heavy" fluxonium, El is typically small
phi_ext_static_q = np.pi # Example: flux sweet spot or other point of interest

Ec_q = 2 * np.pi * Ec_q_GHz
Ej_q = 2 * np.pi * Ej_q_GHz
El_q = 2 * np.pi * El_q_GHz

# Resonator Parameters
omega_r_GHz = 6.0
kappa_r_MHz = 1.0       # Resonator decay rate (linewidth)
omega_r_res = 2 * np.pi * omega_r_GHz
kappa_r = 2 * np.pi * kappa_r_MHz * 1e-3 # Convert MHz to 2*pi*GHz

# Qubit-Resonator Coupling (capacitive: g_qr * n_q * (a + a_dag))
g_qr_MHz = 70.0
g_qr_coup = 2 * np.pi * g_qr_MHz * 1e-3

# JJA Mode Parameters
omega_jja_GHz = 3.5  # Example: A JJA mode frequency
kappa_jja_MHz = 0.5   # JJA mode decay rate
omega_jja_mode = 2 * np.pi * omega_jja_GHz
kappa_jja = 2 * np.pi * kappa_jja_MHz * 1e-3

# Qubit-JJA Mode Coupling (charge coupling: g_q_jja * n_q * (b + b_dag))
g_q_jja_MHz = 30.0
g_q_jja_coup = 2 * np.pi * g_q_jja_MHz * 1e-3

# TLS Parameters
omega_tls_GHz = Ec_q_GHz + 0.05 # Example: TLS nearly resonant with a qubit transition
gamma_tls1_MHz = 0.1     # TLS T1 decay rate
gamma_tls_phi_MHz = 0.2  # TLS pure dephasing rate
omega_tls_val = 2 * np.pi * omega_tls_GHz
gamma_tls1 = 2 * np.pi * gamma_tls1_MHz * 1e-3
gamma_tls_phi = 2 * np.pi * gamma_tls_phi_MHz * 1e-3

# Qubit-TLS Coupling (charge coupling: g_q_tls * n_q * sigmax_tls)
g_q_tls_MHz = 5.0
g_q_tls_coup = 2 * np.pi * g_q_tls_MHz * 1e-3

# Qubit Intrinsic Decoherence
gamma_q1_MHz = 0.01 # Qubit T1 decay rate (e.g., for 0->1 transition)
gamma_q_phi_MHz = 0.02 # Qubit pure dephasing rate
gamma_q1 = 2 * np.pi * gamma_q1_MHz * 1e-3
gamma_q_phi = 2 * np.pi * gamma_q_phi_MHz * 1e-3

# Drive Parameters
# Readout drive (on resonator)
readout_drive_amp_MHz = 20.0 # Amplitude of the resonator drive field (epsilon)
readout_drive_freq_GHz = omega_r_GHz # Drive resonator on resonance
readout_drive_amp = 2 * np.pi * readout_drive_amp_MHz * 1e-3
readout_drive_freq = 2 * np.pi * readout_drive_freq_GHz

# Experimental pulse (example: direct qubit drive on sigma_x)
# This requires identifying qubit sigma_x in the fluxonium eigenbasis.
# For simplicity here, we'll drive the n_op_fluxonium_charge, which couples to sigma_x like transitions.
# A more rigorous approach would be to diagonalize H_fluxonium and construct sigma_x.
exp_pulse_amp_MHz = 10.0 # Amplitude of the experimental pulse
exp_pulse_freq_GHz = (Ec_q_GHz - El_q_GHz) # Placeholder: Target a qubit transition (NEEDS REFINEMENT)
exp_pulse_amp = 2 * np.pi * exp_pulse_amp_MHz * 1e-3
exp_pulse_freq = 2 * np.pi * exp_pulse_freq_GHz # This needs to be set to an actual transition

# %% --- 3. Construct Static Hamiltonian and Operators ---

# Get fluxonium Hamiltonian and its operators
Hq_bare, phi_q_op, n_q_op, _ = hamiltonian_fluxonium(N_q, Ec_q, Ej_q, El_q, phi_ext_static_q, return_ops=True)

# Bare Hamiltonians for other subsystems
Hr_bare = hamiltonian_resonator(N_r, omega_r_res)
Hjja_bare = hamiltonian_jja_mode(N_jja, omega_jja_mode)
Htls_bare = hamiltonian_tls(omega_tls_val)

# Identity operators for tensor products
id_q = qtp.qeye(N_q)
id_r = qtp.qeye(N_r)
id_jja = qtp.qeye(N_jja)
id_tls = qtp.qeye(N_tls) # TLS is 2-level

# Promote bare Hamiltonians to full Hilbert space
# Order: Qubit x Resonator x JJA_Mode x TLS
Hq_full = qtp.tensor(Hq_bare, id_r, id_jja, id_tls)
Hr_full = qtp.tensor(id_q, Hr_bare, id_jja, id_tls)
Hjja_full = qtp.tensor(id_q, id_r, Hjja_bare, id_tls)
Htls_full = qtp.tensor(id_q, id_r, id_jja, Htls_bare)

# Interaction Terms
# Qubit n_q operator in full space
n_q_full = qtp.tensor(n_q_op, id_r, id_jja, id_tls)

# Resonator a, a_dag in full space
a_r_op = qtp.destroy(N_r)
a_r_full = qtp.tensor(id_q, a_r_op, id_jja, id_tls)
adag_r_full = qtp.tensor(id_q, qtp.create(N_r), id_jja, id_tls)

# JJA b, b_dag in full space
b_jja_op = qtp.destroy(N_jja)
b_jja_full = qtp.tensor(id_q, id_r, b_jja_op, id_tls)
bdag_jja_full = qtp.tensor(id_q, id_r, qtp.create(N_jja), id_tls)

# TLS sigma_x, sigma_y, sigma_z in full space
sigmax_tls_op = qtp.sigmax()
sigmax_tls_full = qtp.tensor(id_q, id_r, id_jja, sigmax_tls_op)
sigmay_tls_op = qtp.sigmay()
sigmay_tls_full = qtp.tensor(id_q, id_r, id_jja, sigmay_tls_op)
sigmaz_tls_op = qtp.sigmaz()
sigmaz_tls_full = qtp.tensor(id_q, id_r, id_jja, sigmaz_tls_op)


# H_qr_coupling: g_qr * n_q * (a_r + adag_r) (charge-position form)
# Note: The n_q from fluxonium is dimensionless. g_qr must have units of energy.
# The n_q_op here is the one from the 4Ec n^2 term.
H_qr_int = g_qr_coup * n_q_full * (a_r_full + adag_r_full)

# H_q_jja_coupling: g_q_jja * n_q * (b_jja + bdag_jja)
H_q_jja_int = g_q_jja_coup * n_q_full * (b_jja_full + bdag_jja_full)

# H_q_tls_coupling: g_q_tls * n_q * sigmax_tls
H_q_tls_int = g_q_tls_coup * n_q_full * sigmax_tls_full

# Static Hamiltonian H0
H0 = Hq_full + Hr_full + Hjja_full + Htls_full + H_qr_int + H_q_jja_int + H_q_tls_int

# Operators for Drives
# Resonator drive operator: (a_r + adag_r)
H_res_drive_op = (a_r_full + adag_r_full)

# Experimental pulse operator (example: drive qubit via n_q, similar to sigma_x for some transitions)
# A better way is to find fluxonium eigenstates |g>,|e> and define sigma_x_q = |g><e| + |e><g|
# For now, let's use n_q_full as a proxy for driving qubit transitions.
H_exp_drive_op = n_q_full # This will drive transitions where <f|n_q|i> is non-zero.
                           # Or, use phi_q_full = qtp.tensor(phi_q_op, id_r, id_jja, id_tls) for flux drive

# To find actual qubit transition frequency for exp_pulse_freq:
evals_H0, _ = H0.eigenstates()
# This is complex as H0 includes couplings. We need bare qubit states first.
evals_q_bare, ekets_q_bare = Hq_bare.eigenstates()
# Assume ground is 0, first excited is 1 for the qubit
if len(evals_q_bare) > 1:
    bare_q_freq_01 = evals_q_bare[1] - evals_q_bare[0]
    exp_pulse_freq = bare_q_freq_01 # Drive resonant with bare 0-1 transition
    print(f"Bare qubit 0-1 transition angular frequency: {bare_q_freq_01 / (2*np.pi):.3f} GHz")
    if exp_pulse_freq <= 0: # Safety check
        print("Warning: Calculated experimental pulse frequency is <= 0. Using arbitrary value.")
        exp_pulse_freq = 2 * np.pi * 0.1 # Arbitrary low freq if calculation failed
else:
    print("Warning: Not enough qubit levels to determine transition frequency. Using arbitrary value.")
    exp_pulse_freq = 2 * np.pi * 0.1


# %% --- 4. Time Sequence and Coefficients for Drives ---
# Times are in ns

# Durations
tau2_duration_ns = 5 * (1 / (kappa_r / (2*np.pi))) if kappa_r > 0 else 200 # e.g., 5 / kappa_resonator (in GHz for kappa_r)
experiment_duration_ns = 50.0
readout2_duration_ns = 200.0

# Calculate end times for each period
t_buffer_end_ns = tau2_duration_ns
t_experiment_end_ns = tau2_duration_ns + experiment_duration_ns
t_total_simulation_ns = tau2_duration_ns + experiment_duration_ns + readout2_duration_ns

# Time list for simulation
num_points = int(t_total_simulation_ns * 5) # 5 points per ns as an example
tlist_ns = np.linspace(0, t_total_simulation_ns, num_points)

# Coefficient for resonator readout drive (active during Final Readout)
def coeff_readout_drive(t, args):
    amp = args['readout_amplitude']
    pulse_start = args['readout_pulse_start_time']
    pulse_end = args['readout_pulse_end_time']
    drive_freq = args['readout_drive_freq']
    
    if pulse_start <= t < pulse_end:
        # Square pulse envelope, with cosine modulation
        return amp * np.cos(drive_freq * t)
    else:
        return 0.0

args_readout_drive = {
    'readout_amplitude': readout_drive_amp, 
    'readout_pulse_start_time': t_experiment_end_ns,
    'readout_pulse_end_time': t_total_simulation_ns,
    'readout_drive_freq': readout_drive_freq
}

# Coefficient for experimental pulse (active during Qubit Pulse Sequence)
def coeff_experimental_pulse(t, args):
    amp = args['exp_pulse_amplitude']
    pulse_start = args['exp_pulse_start_time']
    pulse_end = args['exp_pulse_end_time']
    drive_freq = args['exp_pulse_freq']
    
    if pulse_start <= t < pulse_end:
        # Square pulse envelope, with cosine modulation
        return amp * np.cos(drive_freq * t)
    else:
        return 0.0
            
args_experimental_pulse = {
    'exp_pulse_amplitude': exp_pulse_amp,
    'exp_pulse_start_time': t_buffer_end_ns,
    'exp_pulse_end_time': t_experiment_end_ns,
    'exp_pulse_freq': exp_pulse_freq
}

# Combine args for mesolve
all_args = {**args_readout_drive, **args_experimental_pulse}

# Assemble H_t for mesolve
H_t = [H0, 
       [H_res_drive_op, coeff_readout_drive],
       [H_exp_drive_op, coeff_experimental_pulse]
      ]

# %% --- 5. Initial State and Collapse Operators ---

# Initial state: Qubit in ground state (heralded), others in vacuum
# Find true ground state of Hq_bare
_, q_eigenstates = Hq_bare.eigenstates()
psi_q0 = q_eigenstates[0] # Assumes lowest energy eigenstate is ground

psi_r0 = qtp.basis(N_r, 0)
psi_jja0 = qtp.basis(N_jja, 0)
psi_tls0 = qtp.basis(N_tls, 0) # TLS ground state (e.g., spin down)

psi0 = qtp.tensor(psi_q0, psi_r0, psi_jja0, psi_tls0)

# Collapse Operators
c_ops = []
# Resonator decay
if kappa_r > 0:
    c_ops.append(np.sqrt(kappa_r) * a_r_full)
# JJA mode decay
if kappa_jja > 0:
    c_ops.append(np.sqrt(kappa_jja) * b_jja_full)
# TLS T1 decay and dephasing
if gamma_tls1 > 0:
    tls_sm = qtp.tensor(id_q, id_r, id_jja, qtp.sigmam()) # TLS sigma minus
    c_ops.append(np.sqrt(gamma_tls1) * tls_sm)
if gamma_tls_phi > 0:
    tls_sz = qtp.tensor(id_q, id_r, id_jja, qtp.sigmaz()) # TLS sigma z
    c_ops.append(np.sqrt(gamma_tls_phi/2.0) * tls_sz) # Pure dephasing

# Qubit intrinsic decoherence (simplified: apply to bare qubit states 0 and 1)
# A more rigorous treatment involves Lindblad operators in the energy eigenbasis of H0 or Hq_full
if gamma_q1 > 0 and N_q > 1:
    # Projector onto qubit ground state |g_q>
    P_g_q = qtp.tensor(q_eigenstates[0] * q_eigenstates[0].dag(), id_r, id_jja, id_tls)
    # Projector onto qubit first excited state |e_q>
    P_e_q = qtp.tensor(q_eigenstates[1] * q_eigenstates[1].dag(), id_r, id_jja, id_tls)
    # Sigma minus for the qubit subspace (bare)
    q_sm_bare = q_eigenstates[0] * q_eigenstates[1].dag()
    q_sm_full = qtp.tensor(q_sm_bare, id_r, id_jja, id_tls)
    c_ops.append(np.sqrt(gamma_q1) * q_sm_full)

if gamma_q_phi > 0 and N_q > 1:
    # Pure dephasing on qubit: related to |e_q><e_q|
    q_proj_e_full = qtp.tensor(q_eigenstates[1] * q_eigenstates[1].dag(), id_r, id_jja, id_tls)
    c_ops.append(np.sqrt(gamma_q_phi/2.0) * q_proj_e_full) # Approximation for dephasing of |1> state

# %% --- 6. Observables ---
# Projectors for qubit states (using bare eigenstates for simplicity)
proj_q_states = []
for i in range(min(N_q, 4)): # Look at first few qubit states
    op = qtp.tensor(q_eigenstates[i] * q_eigenstates[i].dag(), id_r, id_jja, id_tls)
    proj_q_states.append(op)

# Resonator photon number
n_r_expect = qtp.tensor(id_q, qtp.num(N_r), id_jja, id_tls)
# JJA mode photon number
n_jja_expect = qtp.tensor(id_q, id_r, qtp.num(N_jja), id_tls)
# TLS excitation (population in TLS |e_tls>)
proj_e_tls = qtp.tensor(id_q, id_r, id_jja, qtp.basis(N_tls,1) * qtp.basis(N_tls,1).dag())

e_ops = proj_q_states + [n_r_expect, n_jja_expect, proj_e_tls]
e_ops_labels = [f'P(q{i})' for i in range(len(proj_q_states))] + ['<n_r>', '<n_jja>', 'P(e_tls)']


# %% --- 7. Run Simulation ---
print("Starting QuTiP simulation...")
# Use progress_bar=True for long simulations
# options = qtp.Options(nsteps=5000, store_final_state=True) # Adjust nsteps if needed
output = qtp.mesolve(H_t, psi0, tlist_ns, c_ops, e_ops, args=all_args, progress_bar=True)
print("Simulation complete.")

# %% --- 8. Plot Results ---
fig, axes = plt.subplots(len(e_ops), 1, figsize=(10, 2 * len(e_ops)), sharex=True)

for i, data in enumerate(output.expect):
    ax = axes[i] if len(e_ops) > 1 else axes
    ax.plot(tlist_ns, data, label=e_ops_labels[i])
    ax.set_ylabel(e_ops_labels[i])
    ax.legend(loc='upper right')
    if i == 0: # Add vertical lines for pulse timings on the first plot
        ax.axvline(t_buffer_end_ns, color='gray', linestyle='--', label=r'End $\tau_2$ / Start Exp.')
        ax.axvline(t_experiment_end_ns, color='black', linestyle='--', label='End Exp. / Start Readout')
        ax.legend(loc='center right')


axes[-1].set_xlabel("Time (ns)")
fig.tight_layout()
plt.suptitle("Heralded Readout Simulation: Heavy Fluxonium System", fontsize=16)
plt.subplots_adjust(top=0.95) # Adjust top to make space for suptitle
plt.show()

# Example: Print final qubit state populations
print("\nFinal State Populations:")
for i in range(len(proj_q_states)):
    print(f"{e_ops_labels[i]}: {output.expect[i][-1]:.4f}")
print(f"{e_ops_labels[len(proj_q_states)]} (avg resonator photons): {output.expect[len(proj_q_states)][-1]:.4f}")
print(f"{e_ops_labels[len(proj_q_states)+1]} (avg JJA photons): {output.expect[len(proj_q_states)+1][-1]:.4f}")
print(f"{e_ops_labels[len(proj_q_states)+2]} (TLS excited state pop): {output.expect[len(proj_q_states)+2][-1]:.4f}")

