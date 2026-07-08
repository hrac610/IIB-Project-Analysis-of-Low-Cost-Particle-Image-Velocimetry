import DigitalTwin
import numpy as np
import time
from pathlib import Path
import pickle

def calculate_non_dimensional_parameters(particle, fluid_flow, camera,double_exposure_properties,
                                         interrogation_properties, n_particles, gravity):
    #These dimensional parameters are subject to the caveat that the FOV centre is always fixed at (d_c/2,d_c/2)and U_max=2.2U

    if gravity != 0 or camera.get_x_center() != fluid_flow.get_characteristic_distance() or camera.get_y_center() != fluid_flow.get_characteristic_distance() or interrogation_properties.get_max_measurable_velocity() != 2.2 * fluid_flow.get_characteristic_velocity():
        raise ValueError("Underlying assumptions not satisfied.")

    R = float(camera.get_x_pixels()**2)
    w_0 = float(camera.get_x_height())
    d_p = float(particle.get_diameter())
    rho_p = float(particle.get_density())
    U = float(fluid_flow.get_characteristic_velocity())
    d_c = float(fluid_flow.get_characteristic_distance()) * 2
    mu = float(fluid_flow.get_dynamic_viscosity())
    rho_f = float(fluid_flow.get_density())
    t_e = float(double_exposure_properties.get_exposure_time())
    t_delta = float(double_exposure_properties.get_interframing_time())
    I = float(double_exposure_properties.get_illumination_intensity())
    d_i = float(double_exposure_properties.get_particle_image_sigma())
    w_i = float(interrogation_properties.get_spot_size())
    N_o = float(interrogation_properties.get_spot_overlap_factor())

    alpha = np.sqrt(R)/w_0
    beta = (rho_p*d_p**2*U)/(18*mu*d_c)
    gamma = rho_p/rho_f
    delta = (rho_f * U * d_c) / mu
    epsilon = d_c / w_0
    zeta = n_particles * w_i**2 / w_0**2
    eta = N_o
    kappa = w_i / w_0
    lambda_ = t_delta * U / d_c
    xi = t_e / t_delta
    phi = (d_i * w_0) / (np.sqrt(R) * w_i)
    psi = (I * t_e) / R

    return (alpha, beta, gamma, delta, epsilon, zeta, eta, kappa, lambda_, xi, phi, psi)

def calculate_dimensional_parameters_from_non_dimensional(alpha, beta, gamma, delta, epsilon, zeta, eta, kappa, lambda_, xi, phi, psi, w_0, t_delta, mu):
    w_0 = w_0
    t_delta = t_delta
    mu = mu
    N_o = eta
    w_i = kappa * w_0
    R = int(np.ceil(np.sqrt(alpha**2 * w_0**2))**2)
    d_c = epsilon * w_0
    n_particles = int(zeta * w_0**2 / w_i**2)
    t_e = xi * t_delta
    d_i = phi * np.sqrt(R) * w_i / w_0
    I = (psi * R) / (t_e)
    rho_f = (delta*kappa**2)/(lambda_*epsilon**2) * t_delta * mu / w_i**2
    rho_p = gamma * rho_f
    U = (delta * mu) / (rho_f * d_c)
    d_p = np.sqrt((18 * beta * mu * d_c) / (rho_p * U))
    return R, w_0, d_p, rho_p, U, d_c, mu, rho_f, t_e, t_delta, I, d_i, w_i, N_o, n_particles

def return_typical_dimensional_parameters():
    (R, w_0, d_p, rho_p, U, d_c,
 mu, rho_f, t_e, t_delta, I, d_i,
 w_i, N_o, n_particles) = (2048*2048, 75e-3, 0.3e-6, 920, 30, 0.072,
                           18.5e-6, 1.225, 1e-8, 4e-5, 1e12, 1,
                           1.171875e-3*2, 1, int(3*4096))
    return (R, w_0, d_p, rho_p, U, d_c, mu, rho_f, t_e, t_delta, I, d_i, w_i, N_o, n_particles)

def save_simulation_data(filename, particle, fluid_flow, camera, n_particles, double_exposure_properties,
                         interrogation_properties, gravity, interrogation_spots, velocity_vectors, errors, rms_error):
    """Saves one simulation entry per file in a folder for fast append performance.

    Args:
        filename: Output folder path. A sequential file (`sim_000000.npz`, ...) is
        created for each simulation entry.
    """

    entry = dict(
        particle_name=particle.get_name(),
        d_p=particle.get_diameter(),
        rho_p=particle.get_density(),
        U=fluid_flow.get_characteristic_velocity(),
        d_c=fluid_flow.get_characteristic_distance() * 2,
        mu=fluid_flow.get_dynamic_viscosity(),
        rho_f=fluid_flow.get_density(),
        camera_name=camera.get_name(),
        x_resolution=camera.get_x_pixels(),
        y_resolution=camera.get_y_pixels(),
        x_min=camera.get_x_min(),
        x_max=camera.get_x_max(),
        y_min=camera.get_y_min(),
        y_max=camera.get_y_max(),
        n_particles=n_particles,
        t_e=double_exposure_properties.get_exposure_time(),
        t_delta=double_exposure_properties.get_time_interframing_time(),
        illumination_intensity=double_exposure_properties.get_illumination_intensity(),
        max_measurable_velocity=interrogation_properties.get_max_measurable_velocity(),
        spot_size=interrogation_properties.get_spot_size(),
        spot_overlap_factor=interrogation_properties.get_spot_overlap_factor(),
        particle_image_sigma=double_exposure_properties.get_particle_image_sigma(),
        gravity=gravity,
        interrogation_spots=interrogation_spots,
        velocity_vectors=velocity_vectors,
        errors=errors,
        rms_error=rms_error,
    )

    out_dir = Path(filename)
    out_dir.mkdir(parents=True, exist_ok=True)

    existing_files = sorted(out_dir.glob("sim_*.npz"))
    sim_index = len(existing_files)
    out_file = out_dir / f"sim_{sim_index:06d}.npz"

    # Store native dtypes to avoid object serialization overhead.
    payload = {}
    for key, value in entry.items():
        if isinstance(value, np.ndarray):
            payload[key] = value
        else:
            payload[key] = np.asarray(value)
    print(f"Saving simulation data to {out_file}...")
    np.savez(out_file, **payload)

def load_simulation_data(filename):
    """Loads all saved simulation data from a folder of per-simulation .npz files.

    Returns a dictionary where each key is a parameter name with '_list' appended,
    and values are lists of the saved values across all entries."""
    out_dir = Path(filename)
    sim_files = sorted(out_dir.glob("sim_*.npz"))

    keys = ['particle_name', 'd_p', 'rho_p', 'U', 'd_c', 'mu', 'rho_f',
            'camera_name', 'x_resolution', 'y_resolution', 'x_min', 'x_max',
            'y_min', 'y_max', 'n_particles', 't_e', 't_delta',
            'illumination_intensity', 'max_measurable_velocity', 'spot_size',
            'spot_overlap_factor', 'particle_image_sigma', 'gravity', 'interrogation_spots', 'velocity_vectors', 'errors', 'rms_error']

    result = {f"{key}_list": [] for key in keys}

    for sim_file in sim_files:
        with np.load(sim_file, allow_pickle=False) as data:
            for key in keys:
                value = data[key]
                result[f"{key}_list"].append(value.item() if value.ndim == 0 else value)

    return result

def simulate(R, w_0, d_p, rho_p, U, d_c, mu, rho_f, t_e, t_delta, I, d_i, w_i, N_o, n_particles):
    particle = DigitalTwin.Particle("Particle 1", d_p, rho_p)
    fluid_flow = DigitalTwin.Potential_flow_around_a_cylinder(U, d_c/2, mu, rho_f)
    camera = DigitalTwin.Camera("Camera 1", int(np.ceil(np.sqrt(R))), int(np.ceil(np.sqrt(R))), d_c / 2, d_c / 2, w_0, w_0)
    double_exposure_properties = DigitalTwin.Double_exposure_properties(t_e, t_delta, I, d_i)
    interrogation_properties = DigitalTwin.Interrogation_properties(fluid_flow.get_characteristic_velocity()*2.2, w_i, N_o)
    gravity = 0
    print("#"*200)
    print(f"Dimensional parameters: {R:.8g}, {w_0:.8g}, {d_p:.8g}, {rho_p:.8g}, {U:.8g}, {d_c:.8g}, {mu:.8g}, {rho_f:.8g}, {t_e:.8g}, {t_delta:.8g},{I:.8g}, {d_i:.8g}, {w_i:.8g}, {N_o:.8g}, {n_particles:.8g}")
    non_dim_params = calculate_non_dimensional_parameters(particle, fluid_flow, camera,double_exposure_properties, interrogation_properties, n_particles, gravity)
    formatted_params = ", ".join(f"{p:.8g}" for p in non_dim_params)
    print(f"Non-dimensional parameters: {formatted_params}")
    start = time.time_ns()
    image1, image2, interrogation_spots, velocity_vectors, errors, rms_error = DigitalTwin.PIV_simulation(particle, fluid_flow, camera,
                                                                                      double_exposure_properties, interrogation_properties, n_particles, gravity)
    end = time.time_ns()
    print(f"RMS error: {rms_error}, RMS error / U: {rms_error/U}, Time taken: {(end-start)/1e9}s")
    return image1, image2, interrogation_spots, velocity_vectors, errors, rms_error

def save_results(results_name, filename=None):
    if filename is None:
        filename = f"{results_name}.pkl"
    
    results_dict = {results_name: globals()[results_name]}
    with open(filename, 'wb') as f:
        pickle.dump(results_dict, f)
    print(f"{results_name} saved to {filename}")

def load_results(results_name, filename=None):
    if filename is None:
        filename = f"{results_name}.pkl"
    
    with open(filename, 'rb') as f:
        results_dict = pickle.load(f)
    
    globals()[results_name] = results_dict[results_name]
    print(f"{results_name} loaded from {filename}")
    return results_dict[results_name]








if __name__ == "__main__":
    #Typical system parameters
    (R, w_0, d_p, rho_p, U, d_c,
    mu, rho_f, t_e, t_delta, I, sigma_i,
    w_i, N_o, n_particles) = return_typical_dimensional_parameters()
    alpha, beta, gamma, delta, epsilon, zeta, eta, kappa, lambda_, xi, phi, psi = calculate_non_dimensional_parameters(DigitalTwin.Particle("Particle 1", d_p, rho_p),
                                                        DigitalTwin.Potential_flow_around_a_cylinder(U, d_c/2, mu, rho_f),
                                                        DigitalTwin.Camera("Camera 1", int(np.ceil(np.sqrt(R))), int(np.ceil(np.sqrt(R))), d_c / 2, d_c / 2, w_0, w_0),
                                                        DigitalTwin.Double_exposure_properties(t_e, t_delta, I, sigma_i),
                                                        DigitalTwin.Interrogation_properties(U*2.2, w_i, N_o),
                                                        int(n_particles),
                                                        gravity=0)

    alpha_range = np.repeat(np.geomspace(alpha*0.1, alpha*2, 40), 100)
    beta_range = np.repeat(np.geomspace(beta*0.1, beta*1000, 80), 100)
    gamma_range = np.repeat(np.geomspace(gamma*0.1, gamma*10, 40), 100)
    delta_range = np.repeat(np.geomspace(delta*0.1, delta*3, 40), 100)
    epsilon_range = np.repeat(np.geomspace(epsilon*0.3, epsilon*10, 40), 100)
    zeta_range = np.repeat(np.geomspace(zeta*0.1, zeta*50, 54), 100)
    eta_range = np.repeat(np.geomspace(eta*0.1, eta*3, 40), 100)
    kappa_range = np.repeat(np.geomspace(kappa*0.4, kappa*10, 40), 100)
    lambda_range = np.repeat(np.geomspace(lambda_*0.01, lambda_*3, 50), 100)
    xi_range = np.repeat(np.geomspace(xi*0.1, xi*2000, 86), 100)
    phi_range = np.repeat(np.geomspace(phi*0.1, phi*10, 40), 100)
    psi_range = np.repeat(np.geomspace(psi*0.001, psi*100, 80), 100)

    
    alpha_results = []
    for alpha_prime in alpha_range:
        (R, w_0, d_p, rho_p, U, d_c, mu, rho_f, t_e, t_delta, I, sigma_i, w_i, N_o, n_particles) = calculate_dimensional_parameters_from_non_dimensional(alpha_prime, beta, gamma, delta, epsilon, zeta, eta, kappa, lambda_, xi, phi, psi, w_0, t_delta, mu)
        image1, image2, interrogation_spots, velocity_vectors, errors, rms_error = simulate(R, w_0, d_p, rho_p, U, d_c, mu, rho_f, t_e, t_delta, I, sigma_i, w_i, N_o, n_particles)
        alpha_results.append((alpha_prime, rms_error/U))
    save_results("alpha_results")
    
    
    beta_results = []
    for beta_prime in beta_range:
        (R, w_0, d_p, rho_p, U, d_c, mu, rho_f, t_e, t_delta, I, sigma_i, w_i, N_o, n_particles) = calculate_dimensional_parameters_from_non_dimensional(alpha, beta_prime, gamma, delta, epsilon, zeta, eta, kappa, lambda_, xi, phi, psi, w_0, t_delta, mu)
        image1, image2, interrogation_spots, velocity_vectors, errors, rms_error = simulate(R, w_0, d_p, rho_p, U, d_c, mu, rho_f, t_e, t_delta, I, sigma_i, w_i, N_o, n_particles)
        beta_results.append((beta_prime, rms_error/U))
    save_results("beta_results")
    
    
    gamma_results = []
    for gamma_prime in gamma_range:
       (R, w_0, d_p, rho_p, U, d_c, mu, rho_f, t_e, t_delta, I, sigma_i, w_i, N_o, n_particles) = calculate_dimensional_parameters_from_non_dimensional(alpha, beta, gamma_prime, delta, epsilon, zeta, eta, kappa, lambda_, xi, phi, psi, w_0, t_delta, mu)
       image1, image2, interrogation_spots, velocity_vectors, errors, rms_error = simulate(R, w_0, d_p, rho_p, U, d_c, mu, rho_f, t_e, t_delta, I, sigma_i, w_i, N_o, n_particles)
       gamma_results.append((gamma_prime, rms_error/U))
    save_results("gamma_results")
    
    delta_results = []
    for delta_prime in delta_range:
        (R, w_0, d_p, rho_p, U, d_c, mu, rho_f, t_e, t_delta, I, sigma_i, w_i, N_o, n_particles) = calculate_dimensional_parameters_from_non_dimensional(alpha, beta, gamma, delta_prime, epsilon, zeta, eta, kappa, lambda_, xi, phi, psi, w_0, t_delta, mu)
        image1, image2, interrogation_spots, velocity_vectors, errors, rms_error = simulate(R, w_0, d_p, rho_p, U, d_c, mu, rho_f, t_e, t_delta, I, sigma_i, w_i, N_o, n_particles)
        delta_results.append((delta_prime, rms_error/U))
    save_results("delta_results")

    epsilon_results = []
    for epsilon_prime in epsilon_range:
        (R, w_0, d_p, rho_p, U, d_c, mu, rho_f, t_e, t_delta, I, sigma_i, w_i, N_o, n_particles) = calculate_dimensional_parameters_from_non_dimensional(alpha, beta, gamma, delta, epsilon_prime, zeta, eta, kappa, lambda_, xi, phi, psi, w_0, t_delta, mu)
        image1, image2, interrogation_spots, velocity_vectors, errors, rms_error = simulate(R, w_0, d_p, rho_p, U, d_c, mu, rho_f, t_e, t_delta, I, sigma_i, w_i, N_o, n_particles)
        epsilon_results.append((epsilon_prime, rms_error/U))
    save_results("epsilon_results")
    
    zeta_results = []
    for zeta_prime in zeta_range:
        (R, w_0, d_p, rho_p, U, d_c, mu, rho_f, t_e, t_delta, I, sigma_i, w_i, N_o, n_particles) = calculate_dimensional_parameters_from_non_dimensional(alpha, beta, gamma, delta, epsilon, zeta_prime, eta, kappa, lambda_, xi, phi, psi, w_0, t_delta, mu)
        image1, image2, interrogation_spots, velocity_vectors, errors, rms_error = simulate(R, w_0, d_p, rho_p, U, d_c, mu, rho_f, t_e, t_delta, I, sigma_i, w_i, N_o, n_particles)
        zeta_results.append((zeta_prime, rms_error/U))
    save_results("zeta_results")
    
    eta_results = []
    for eta_prime in eta_range:
        (R, w_0, d_p, rho_p, U, d_c, mu, rho_f, t_e, t_delta, I, sigma_i, w_i, N_o, n_particles) = calculate_dimensional_parameters_from_non_dimensional(alpha, beta, gamma, delta, epsilon, zeta, eta_prime, kappa, lambda_, xi, phi, psi, w_0, t_delta, mu)
        image1, image2, interrogation_spots, velocity_vectors, errors, rms_error = simulate(R, w_0, d_p, rho_p, U, d_c, mu, rho_f, t_e, t_delta, I, sigma_i, w_i, N_o, n_particles)
        eta_results.append((eta_prime, rms_error/U))
    save_results("eta_results")
    
    kappa_results = []
    for kappa_prime in kappa_range:
        (R, w_0, d_p, rho_p, U, d_c, mu, rho_f, t_e, t_delta, I, sigma_i, w_i, N_o, n_particles) = calculate_dimensional_parameters_from_non_dimensional(alpha, beta, gamma, delta, epsilon, zeta, eta, kappa_prime, lambda_, xi, phi, psi, w_0, t_delta, mu)
        image1, image2, interrogation_spots, velocity_vectors, errors, rms_error = simulate(R, w_0, d_p, rho_p, U, d_c, mu, rho_f, t_e, t_delta, I, sigma_i, w_i, N_o, n_particles)
        kappa_results.append((kappa_prime, rms_error/U))
    save_results("kappa_results")
    
    lambda_results = []
    for lambda_prime in lambda_range:
        (R, w_0, d_p, rho_p, U, d_c, mu, rho_f, t_e, t_delta, I, sigma_i, w_i, N_o, n_particles) = calculate_dimensional_parameters_from_non_dimensional(alpha, beta, gamma, delta, epsilon, zeta, eta, kappa, lambda_prime, xi, phi, psi, w_0, t_delta, mu)
        image1, image2, interrogation_spots, velocity_vectors, errors, rms_error = simulate(R, w_0, d_p, rho_p, U, d_c, mu, rho_f, t_e, t_delta, I, sigma_i, w_i, N_o, n_particles)
        lambda_results.append((lambda_prime, rms_error/U))
    save_results("lambda_results")
    
    xi_results = []
    for xi_prime in xi_range:
        (R, w_0, d_p, rho_p, U, d_c, mu, rho_f, t_e, t_delta, I, sigma_i, w_i, N_o, n_particles) = calculate_dimensional_parameters_from_non_dimensional(alpha, beta, gamma, delta, epsilon, zeta, eta, kappa, lambda_, xi_prime, phi, psi, w_0, t_delta, mu)
        image1, image2, interrogation_spots, velocity_vectors, errors, rms_error = simulate(R, w_0, d_p, rho_p, U, d_c, mu, rho_f, t_e, t_delta, I, sigma_i, w_i, N_o, n_particles)
        xi_results.append((xi_prime, rms_error/U))
    save_results("xi_results")
    
    phi_results = []
    for phi_prime in phi_range:
        (R, w_0, d_p, rho_p, U, d_c, mu, rho_f, t_e, t_delta, I, sigma_i, w_i, N_o, n_particles) = calculate_dimensional_parameters_from_non_dimensional(alpha, beta, gamma, delta, epsilon, zeta, eta, kappa, lambda_, xi, phi_prime, psi, w_0, t_delta, mu)
        image1, image2, interrogation_spots, velocity_vectors, errors, rms_error = simulate(R, w_0, d_p, rho_p, U, d_c, mu, rho_f, t_e, t_delta, I, sigma_i, w_i, N_o, n_particles)
        phi_results.append((phi_prime, rms_error/U))
    save_results("phi_results")
    
    psi_results = []
    for psi_prime in psi_range:
        (R, w_0, d_p, rho_p, U, d_c, mu, rho_f, t_e, t_delta, I, sigma_i, w_i, N_o, n_particles) = calculate_dimensional_parameters_from_non_dimensional(alpha, beta, gamma, delta, epsilon, zeta, eta, kappa, lambda_, xi, phi, psi_prime, w_0, t_delta, mu)
        image1, image2, interrogation_spots, velocity_vectors, errors, rms_error = simulate(R, w_0, d_p, rho_p, U, d_c, mu, rho_f, t_e, t_delta, I, sigma_i, w_i, N_o, n_particles)
        psi_results.append((psi_prime, rms_error/U))
    save_results("psi_results")
    
