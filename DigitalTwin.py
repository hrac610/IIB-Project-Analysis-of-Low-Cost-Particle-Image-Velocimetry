"""
The Digital Twin module simulates particle image velocimetry. An explanation of the architecture is provided in "Analysis of Low Cost Particle Image Velocimetry".
Example code detailing the use of the digital twin for analysis of PIV system errors is provided in "DataGeneration.py" and the results of this analysis are
discussed in "Analysis of Low Cost Particle Image Velocimetry". The Digital Twin is free to use and modify, but please cite it's use in any work resulting from it.
"""



import numpy as np
from numba import njit, prange
import matplotlib.pyplot as plt
import cv2
import time
import os
from concurrent.futures import ThreadPoolExecutor
from itertools import repeat
from scipy import ndimage
from scipy.signal import fftconvolve

#Conventions:
#Functions beginning with a single underscore should not be called in the main body of code. Use their wrapper functions.
#Objects begin with capital letters.

class Particle:
    def __init__(self, name, diameter, density):
        self.__name = name
        self.__diameter = diameter
        self.__density = density

    def get_name(self):
        return self.__name
    def get_diameter(self):
        return self.__diameter
    def get_density(self):
        return self.__density

class Camera:
    def __init__(self, name, x_pixels, y_pixels, x_center, y_center, x_height, y_height):
        self.__name = name
        self.__x_pixels = x_pixels
        self.__y_pixels = y_pixels
        # store camera using centre/height representation
        self.__x_center = x_center
        self.__y_center = y_center
        self.__x_height = x_height
        self.__y_height = y_height
        # compute bounds for validation
        x_min = x_center - x_height / 2
        x_max = x_center + x_height / 2
        y_min = y_center - y_height / 2
        y_max = y_center + y_height / 2
        if x_min >= x_max or y_min >= y_max:
            raise ValueError("Invalid camera bounds: x_min must be less than x_max and y_min must be less than y_max.")
        if self.__x_pixels <= 0 or self.__y_pixels <= 0:
            raise ValueError("Invalid camera resolution: x_pixels and y_pixels must be positive integers.")
        if self.__x_pixels / self.__y_pixels != (x_max - x_min) / (y_max - y_min):
            raise ValueError("Aspect ratio of camera resolution must match aspect ratio of camera bounds.")
    
    def get_name(self):
        return self.__name
    def get_x_pixels(self):
        return self.__x_pixels
    def get_y_pixels(self):
        return self.__y_pixels
    def get_x_center(self):
        return self.__x_center
    def get_y_center(self):
        return self.__y_center
    def get_x_height(self):
        return self.__x_height
    def get_y_height(self):
        return self.__y_height
    # Compatibility getters: compute bounds from centre/height
    def get_x_min(self):
        return self.__x_center - self.__x_height / 2
    def get_x_max(self):
        return self.__x_center + self.__x_height / 2
    def get_y_min(self):
        return self.__y_center - self.__y_height / 2
    def get_y_max(self):
        return self.__y_center + self.__y_height / 2
    def get_pixel_size(self):
        x_min = self.get_x_min()
        x_max = self.get_x_max()
        y_min = self.get_y_min()
        y_max = self.get_y_max()
        if self.__x_pixels / self.__y_pixels != (x_max - x_min) / (y_max - y_min):
            raise ValueError("Aspect ratio of camera resolution must match aspect ratio of camera bounds.")
        return (x_max - x_min) / self.__x_pixels

class Fluid_flow:
    __characteristic_velocity = 0
    __characteristic_distance = 0
    __dynamic_viscosity = 0
    __density = 0

    def get_characteristic_velocity(self):
        raise NotImplementedError("This method should be implemented by subclasses of fluid_flow")
    def get_characteristic_distance(self):
        raise NotImplementedError("This method should be implemented by subclasses of fluid_flow")
    def get_dynamic_viscosity(self):
        raise NotImplementedError("This method should be implemented by subclasses of fluid_flow")
    def get_density(self):
        raise NotImplementedError("This method should be implemented by subclasses of fluid_flow")
    def get_kinematic_viscosity(self):
        raise NotImplementedError("This method should be implemented by subclasses of fluid_flow")

class Potential_flow_around_a_cylinder(Fluid_flow):
    def __init__(self, freestream_velocity, cylinder_radius, dynamic_viscosity, density):
        self.__characteristic_velocity = freestream_velocity
        self.__characteristic_distance = cylinder_radius
        self.__dynamic_viscosity = dynamic_viscosity
        self.__density = density
        self.__kinematic_viscosity = self.__dynamic_viscosity / self.__density
    
    def get_characteristic_velocity(self):
        return self.__characteristic_velocity
    def get_characteristic_distance(self):
        return self.__characteristic_distance
    def get_dynamic_viscosity(self):
        return self.__dynamic_viscosity
    def get_density(self):
        return self.__density
    def get_kinematic_viscosity(self):
        return self.__kinematic_viscosity

class Double_exposure_properties:
    """Class to hold properties related to double exposure imaging.
    Attributes:
        exposure_time: Time duration of each exposure (in seconds)
        time_between_exposures: Time interval between the two exposures (in seconds)
        illumination_intensity: Intensity of the illumination for each exposure (arbitrary units)
        particle_image_sigma: Standard deviation of Gaussian particle image in pixels (truncated at 3 sigma)
    """

    def __init__(self, exposure_time, interframing_time, illumination_intensity, particle_image_sigma):
        self.__exposure_time = exposure_time
        self.__interframing_time = interframing_time
        self.__illumination_intensity = illumination_intensity
        self.__particle_image_sigma = particle_image_sigma

    def get_exposure_time(self):
        return self.__exposure_time
    def get_interframing_time(self):
        return self.__interframing_time
    def get_illumination_intensity(self):
        return self.__illumination_intensity
    def get_particle_image_sigma(self):
        return self.__particle_image_sigma

class Interrogation_properties:
    """Class to hold properties related to interrogation of images.
    Attributes:
        max_measurable_velocity: Maximum velocity that can be measured (in m/s)
        spot_size: Size of the interrogation spot (in meters)
        spot_overlap_factor: Factor determining the overlap between adjacent interrogation spots (dimensionless, typically >1)
    """

    def __init__(self, max_measurable_velocity, spot_size, spot_overlap_factor):
        self.__max_measurable_velocity = max_measurable_velocity
        self.__spot_size = spot_size
        self.__spot_overlap_factor = spot_overlap_factor

    def get_max_measurable_velocity(self):
        return self.__max_measurable_velocity
    def get_spot_size(self):
        return self.__spot_size
    def get_spot_overlap_factor(self):
        return self.__spot_overlap_factor

@njit(fastmath=True)
def _get_velocity_of_potential_flow_around_a_cylinder(position, characteristic_velocity, characteristic_distance):
    x = position[0]
    y = position[1]
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    u_r = characteristic_velocity * (1 - (characteristic_distance/r)**2) * np.cos(theta)
    u_theta = -characteristic_velocity * (1 + (characteristic_distance/r)**2) * np.sin(theta)
    u_x = u_r * np.cos(theta) - u_theta * np.sin(theta)
    u_y = u_r * np.sin(theta) + u_theta * np.cos(theta)
    return np.array([u_x, u_y])

@njit(parallel=True, nogil=True, fastmath=True)
def _compute_derivatives(states, characteristic_velocity, characteristic_distance, tau_p, b):
        """Compute derivatives for all particles at once."""
        #velocities = np.array([get_velocity_of_potential_flow_around_a_cylinder(state, characteristic_velocity, characteristic_distance) for state in states])
        velocities = np.zeros_like(states[:, 0:2])
        for i in prange(states.shape[0]):
            velocities[i] = _get_velocity_of_potential_flow_around_a_cylinder(states[i], characteristic_velocity, characteristic_distance)
        derivatives = np.zeros_like(states)
        derivatives[:, 0:2] = states[:, 2:4]
        derivatives[:, 2:4] = (velocities - states[:, 2:4]) / tau_p + b
        return derivatives

@njit(parallel=True, nogil=True, fastmath=True)
def _particle_trajectories(tau_p, b, characteristic_velocity, characteristic_distance, initial_positions_and_velocities, time_step, total_time):
    """
    Compute trajectories for multiple particles (vectorized) using the RK4 method.
    
    Args:
        particle_diameter: Diameter of the particle
        particle_density: Density of the particle
        characteristic_velocity: Characteristic velocity of the fluid flow
        characteristic_distance: Characteristic distance of the fluid flow
        fluid_density: Density of the fluid
        fluid_kinematic_viscosity: Kinematic viscosity of the fluid
        initial_positions_and_velocities: Array of shape (n_particles, 4) with [x, y, vx, vy] for each particle
        time_step: Time step for integration
        total_time: Total simulation time
    
    Returns:
        Array of shape (n_particles, num_steps, 4) containing trajectories for all particles
    """
    n_particles = initial_positions_and_velocities.shape[0]
    num_steps = int(total_time / time_step)
    trajectories = np.zeros((n_particles, num_steps, 4))
    
    # Initialize with initial conditions
    trajectories[:, 0, :] = initial_positions_and_velocities
    
    # Integrate all particles simultaneously using RK4
    for step in range(1, num_steps):
        current_states = trajectories[:, step-1, :]
        
        k1 = _compute_derivatives(current_states, characteristic_velocity, characteristic_distance, tau_p, b)
        k2 = _compute_derivatives(current_states + k1 * time_step / 2, characteristic_velocity, characteristic_distance, tau_p, b)
        k3 = _compute_derivatives(current_states + k2 * time_step / 2, characteristic_velocity, characteristic_distance, tau_p, b)
        k4 = _compute_derivatives(current_states + k3 * time_step, characteristic_velocity, characteristic_distance, tau_p, b)
        
        trajectories[:, step, :] = current_states + (k1 + 2*k2 + 2*k3 + k4) / 6 * time_step
    
    return trajectories


def particle_trajectories(particle, fluid_flow, initial_positions_and_velocities, time_step, total_time, gravity):
    """
    Compute trajectories for multiple particles (vectorized) using the RK4 method.
    
    Args:
        particle: Instance of the particle class
        fluid_flow: Potential flow around a cylinder, instance of a subclass of fluid_flow
        initial_positions_and_velocities: Array of shape (n_particles, 4) with [x, y, vx, vy] for each particle
        time_step: Time step for integration
        total_time: Total simulation time
        gravity: Gravitational acceleration (in m/s^2)

    Returns:
        Array of shape (n_particles, num_steps, 4) containing trajectories for all particles
    """
    
    particle_diameter = particle.get_diameter()
    particle_density = particle.get_density()
    characteristic_velocity = fluid_flow.get_characteristic_velocity()
    characteristic_distance = fluid_flow.get_characteristic_distance()
    fluid_density = fluid_flow.get_density()
    fluid_kinematic_viscosity = fluid_flow.get_kinematic_viscosity()

    # Pre-compute particle parameters
    tau_p = (((particle_density - fluid_density) * particle_diameter**2)
             / (18 * fluid_density * fluid_kinematic_viscosity * 1))
    b = np.array([0, (particle_density - fluid_density) / particle_density * gravity])

    if 0.5 * tau_p < time_step:
        raise ValueError("Warning: Time step is larger than particle response time (0.5 * tau_p). Consider reducing the time step for better accuracy.")

    return _particle_trajectories(tau_p, b, characteristic_velocity, characteristic_distance,
                                  initial_positions_and_velocities, time_step, total_time)

@njit(parallel=True, nogil=True, fastmath=True)
def _particle_positions_after_time(tau_p, b, characteristic_velocity, characteristic_distance, initial_positions_and_velocities, time_step, total_time):
    """
    Compute trajectories for multiple particles (vectorized) using the RK4 method.
    
    Args:
        particle_diameter: Diameter of the particle
        particle_density: Density of the particle
        characteristic_velocity: Characteristic velocity of the fluid flow
        characteristic_distance: Characteristic distance of the fluid flow
        fluid_density: Density of the fluid
        fluid_kinematic_viscosity: Kinematic viscosity of the fluid
        initial_positions_and_velocities: Array of shape (n_particles, 4) with [x, y, vx, vy] for each particle
        time_step: Time step for integration
        total_time: Total simulation time
    
    Returns:
        Array of shape (n_particles, 4) containing final positions and velocities for all particles
    """
    n_particles = initial_positions_and_velocities.shape[0]
    num_steps = int(total_time / time_step)
    trajectories = np.zeros((n_particles,  4))
    
    # Initialize with initial conditions
    trajectories = initial_positions_and_velocities
    
    # Integrate all particles simultaneously using RK4
    for step in range(1, num_steps):
        current_states = trajectories
        
        k1 = _compute_derivatives(current_states, characteristic_velocity, characteristic_distance, tau_p, b)
        k2 = _compute_derivatives(current_states + k1 * time_step / 2, characteristic_velocity, characteristic_distance, tau_p, b)
        k3 = _compute_derivatives(current_states + k2 * time_step / 2, characteristic_velocity, characteristic_distance, tau_p, b)
        k4 = _compute_derivatives(current_states + k3 * time_step, characteristic_velocity, characteristic_distance, tau_p, b)
        
        trajectories = current_states + (k1 + 2*k2 + 2*k3 + k4) / 6 * time_step
    
    return trajectories

def particle_positions_after_time(particle, fluid_flow, initial_positions_and_velocities, time_step, total_time, gravity):
    """
    Compute trajectories for multiple particles (vectorized) using the RK4 method.
    
    Args:
        particle: Instance of the particle class
        fluid_flow: Potential flow around a cylinder, instance of a subclass of fluid_flow
        initial_positions_and_velocities: Array of shape (n_particles, 4) with [x, y, vx, vy] for each particle
        time_step: Time step for integration
        total_time: Total simulation time
        gravity: Gravitational acceleration (in m/s^2)

    Returns:
        Array of shape (n_particles, 4) containing final positions and velocities for all particles
    """
    
    particle_diameter = particle.get_diameter()
    particle_density = particle.get_density()
    characteristic_velocity = fluid_flow.get_characteristic_velocity()
    characteristic_distance = fluid_flow.get_characteristic_distance()
    fluid_density = fluid_flow.get_density()
    fluid_kinematic_viscosity = fluid_flow.get_kinematic_viscosity()

    # Pre-compute particle parameters
    tau_p = (((particle_density - fluid_density) * particle_diameter**2)
             / (18 * fluid_density * fluid_kinematic_viscosity * 1))
    b = np.array([0, (particle_density - fluid_density) / particle_density * gravity])

    if 0.5 * tau_p < time_step:
        raise ValueError("Warning: Time step is larger than particle response time (0.5 * tau_p). Consider reducing the time step for better accuracy.")

    return _particle_positions_after_time(tau_p, b, characteristic_velocity, characteristic_distance,
                                  initial_positions_and_velocities, time_step, total_time)

@njit(parallel=True, nogil=True, fastmath=True)
def _generate_initial_particles(n, x_bounds, y_bounds, cylinder_radius, freestream_velocity):
    """
    Generate n initial particle positions and velocities.
    Positions are uniformly distributed within the specified bounds and outside the cylinder.
    
    Args:
        n: Number of particles to generate
        x_bounds: Tuple (x_min, x_max)
        y_bounds: Tuple (y_min, y_max)
        cylinder_radius: Radius of the exclusion cylinder
        freestream_velocity: Freestream velocity of the fluid flow
    
    Returns:
        Array of shape (n, 4) with [x, y, vx, vy] for each particle
    """
    particles = np.zeros((n, 4))
    x_min, x_max = x_bounds
    y_min, y_max = y_bounds

    i = 0
    while i < n:
        # Generate random position uniformly within bounds
        x = np.random.uniform(x_min, x_max)
        y = np.random.uniform(y_min, y_max)
        
        # Check if position is outside the cylinder
        r = np.sqrt(x**2 + y**2)
        if r > cylinder_radius:
            particles[i, 0] = x
            particles[i, 1] = y
            # Calculate initial velocity from fluid flow
            particles[i, 2:4] = _get_velocity_of_potential_flow_around_a_cylinder(particles[i], freestream_velocity,
                                                                                 cylinder_radius)
            i += 1
    
    return particles


def generate_initial_particles(n, camera, fluid_flow):  #Should do something to make sure some particles are outside of the camera view.
    """Generate n initial particle positions and velocities using the camera bounds and fluid flow characteristics.

    Args:
        n: Number of particles to generate
        camera: Instance of the camera class to determine bounds
        fluid_flow: Instance of a subclass of fluid_flow to determine cylinder radius and freestream velocity
    
    Returns:
        Array of shape (n, 4) with [x, y, vx, vy] for each particle
    """

    x_bounds = (camera.get_x_min(), camera.get_x_max())
    y_bounds = (camera.get_y_min(), camera.get_y_max())
    return _generate_initial_particles(n, x_bounds, y_bounds, fluid_flow.get_characteristic_distance(), fluid_flow.get_characteristic_velocity())

def _create_image(x_pixels, y_pixels, x_min, x_max, y_min, y_max, particle_positions, illumination_intensity, time_step, particle_image_sigma=1.0):
    """
    Create an image from particle positions with Gaussian illumination using scipy vectorization.

    Args:
        x_pixels: Number of pixels in x direction
        y_pixels: Number of pixels in y direction
        x_min, x_max: Bounds of the camera in x direction
        y_min, y_max: Bounds of the camera in y direction
        particle_positions: Array of shape (n_particles, num_steps, 4) with [x, y, vx, vy] for each particle at each time step
        illumination_intensity: Intensity to add for each particle
        time_step: Time step for integration
        particle_image_sigma: Standard deviation of Gaussian illumination kernel in pixels
    
    Returns:
        2D numpy array representing the image
    """
    # Flatten and extract x, y positions
    position_data = np.ndarray.flatten(particle_positions)
    xs = position_data[0::4]
    ys = position_data[1::4]

    # Vectorized conversion to pixel indices
    x_pixels_arr = np.floor((xs - x_min) * x_pixels / (x_max - x_min)).astype(np.int32)
    y_pixels_arr = np.floor((ys - y_min) * y_pixels / (y_max - y_min)).astype(np.int32)
    # Flip y axis for image coordinates
    y_pixels_arr = y_pixels - y_pixels_arr - 1

    # Mask for particles within reasonable bounds
    kernel_extent = 4 * particle_image_sigma
    valid = (x_pixels_arr >= -kernel_extent) & (x_pixels_arr < x_pixels + kernel_extent) & \
            (y_pixels_arr >= -kernel_extent) & (y_pixels_arr < y_pixels + kernel_extent)
    x_pixels_arr = x_pixels_arr[valid]
    y_pixels_arr = y_pixels_arr[valid]

    # Create binary image with particle positions
    particle_image = np.zeros((y_pixels, x_pixels), dtype=np.float64)
    
    # Add 1.0 at each valid particle position
    valid_mask = (x_pixels_arr >= 0) & (x_pixels_arr < x_pixels) & \
                 (y_pixels_arr >= 0) & (y_pixels_arr < y_pixels)
    particle_image[y_pixels_arr[valid_mask], x_pixels_arr[valid_mask]] += 1.0
    
    # Apply Gaussian blur to the entire image at once (vectorized)
    image = ndimage.gaussian_filter(particle_image, sigma=particle_image_sigma)
    
    image_f = image * illumination_intensity * time_step

    # Clip to 255 for uint8 image output
    image = np.clip(image_f, 0, 255).astype(np.uint8)
    return image

def create_image(camera, particle_positions, illumination_intensity, time_step, particle_image_sigma=1.0):
    return _create_image(camera.get_x_pixels(), camera.get_y_pixels(), camera.get_x_min(), camera.get_x_max(),
                         camera.get_y_min(), camera.get_y_max(), particle_positions, illumination_intensity, time_step, particle_image_sigma)

def display_image(image):
    cv2.imshow('Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def double_exposure_simulation(n, particle, camera, fluid_flow, double_exposure_properties, gravity):
    """Simulate a double exposure image of particles in a fluid flow.

    Args:
        particle: Instance of the particle class
        camera: Instance of the camera class
        fluid_flow: Potential flow around a cylinder, instance of a subclass of fluid_flow
        double_exposure_properties: Instance of double_exposure_properties class
        gravity: Gravitational acceleration (in m/s^2)

    Returns:
        Tuple of two images (numpy arrays) representing the two exposures
    """
    # Generate initial particle positions and velocities
    initial_positions_and_velocities = generate_initial_particles(n, camera, fluid_flow)

    #Set time step to the smallest of half the particle response time, the time to cross half a pixel at freestream velocity, and exposure time / 10.
    time_step = np.min([0.5 * (((particle.get_density() - fluid_flow.get_density()) * particle.get_diameter()**2)
             / (18 * fluid_flow.get_density() * fluid_flow.get_kinematic_viscosity() * 1)),
             float(2 * camera.get_pixel_size() / fluid_flow.get_characteristic_velocity()), double_exposure_properties.get_exposure_time() / 10])   

    # Simulate particle trajectories for first exposure
    trajectories_exposure_1 = particle_trajectories(particle, fluid_flow, initial_positions_and_velocities,
                                                time_step, double_exposure_properties.get_exposure_time(), gravity)
    
    time_step_not_exposed = 0.5 * (((particle.get_density() - fluid_flow.get_density()) * particle.get_diameter()**2)
             / (18 * fluid_flow.get_density() * fluid_flow.get_kinematic_viscosity() * 1))
    #Simulate particle motion between exposures
    trajectories_not_exposed = particle_positions_after_time(particle, fluid_flow, initial_positions_and_velocities,
                                                time_step_not_exposed, double_exposure_properties.get_interframing_time(), gravity)
    
    # Simulate particle trajectories for second exposure (starting from the end of the first exposure)
    trajectories_exposure_2 = particle_trajectories(particle, fluid_flow, trajectories_not_exposed,
                                                time_step, double_exposure_properties.get_exposure_time(), gravity)
    # Create image from combined trajectories
    image1 = create_image(camera, trajectories_exposure_1, double_exposure_properties.get_illumination_intensity(),
                        time_step, double_exposure_properties.get_particle_image_sigma())
    image2 = create_image(camera, trajectories_exposure_2, double_exposure_properties.get_illumination_intensity(),
                        time_step, double_exposure_properties.get_particle_image_sigma())
    
    return image1, image2

@njit(parallel=True, nogil=True, fastmath=True)
def _correlation(window1, image2, max_s, centre_pixel_x, centre_pixel_y, spot_half_length):
    correlations = np.zeros((max_s * 2 + 1,max_s * 2 + 1))
    for i in prange(max_s * 2 + 1):
        for j in prange(max_s * 2 + 1):
            x = centre_pixel_x + i - max_s
            y = centre_pixel_y + j - max_s
            window2 = image2[y - spot_half_length:y + spot_half_length + 1][:,x - spot_half_length:x + spot_half_length + 1]
            correlations[i][j] = np.sum(np.multiply(window1, window2))
    return correlations

def _FFT_correlation(window1, image2, max_s, centre_pixel_x, centre_pixel_y, spot_half_length):
    """FFT-based summed-product correlation. Same signature and return shape as _correlation.

    Returns correlations[i][j] where i indexes x-shift and j indexes y-shift.
    """
    S = 2 * max_s + 2 * spot_half_length + 1

    # Extract the search area once and use SciPy's FFT convolution directly.
    y0 = centre_pixel_y - max_s - spot_half_length
    x0 = centre_pixel_x - max_s - spot_half_length
    search = image2[y0:y0 + S, x0:x0 + S]

    # Flip the template to compute cross-correlation via convolution.
    kernel = np.flipud(np.fliplr(window1))
    return fftconvolve(search, kernel, mode="valid").T.copy()

def _interrogate_spot(image1, image2, centre_pixel_x, centre_pixel_y, spot_half_length, max_s):
    if (centre_pixel_x - spot_half_length - max_s < 0 or centre_pixel_x + spot_half_length + max_s + 1 > len(image1) or
    centre_pixel_y - spot_half_length - max_s < 0 or centre_pixel_y + spot_half_length + max_s + 1 > len(image1[0])):
        print(f"centre_pixel_x: {centre_pixel_x}, centre_pixel_y: {centre_pixel_y}, spot_half_length: {spot_half_length}, max_s: {max_s}")
        raise ValueError("Spot and search area must be fully contained within the image.")
    
    window1 = image1[centre_pixel_y - spot_half_length:centre_pixel_y + spot_half_length + 1][:,centre_pixel_x - spot_half_length:centre_pixel_x + spot_half_length + 1]
    correlations = _FFT_correlation(window1, image2, max_s, centre_pixel_x, centre_pixel_y, spot_half_length)
    max_positions = np.argmax(correlations)
    shift_vector = np.zeros(2)
    shift_vector[0] = int(np.floor(max_positions / (max_s * 2 + 1))) - max_s
    shift_vector[1] = -(int(max_positions % (max_s * 2 + 1)) - max_s)
    if np.max(correlations) == np.min(correlations):
        shift_vector[0] = 0
        shift_vector[1] = 0
    return shift_vector     #Returns the shift vector in pixels

def _interrogate_spot_for_centre(image1, image2, spot_half_length, max_s, centre_pixel_x, centre_pixel_y):
    return _interrogate_spot(image1, image2, int(centre_pixel_x), int(centre_pixel_y), spot_half_length, max_s)

def _interrogate_images(image1, image2, max_measurable_velocity, spot_size, spot_overlap_factor, x_min, x_max, y_min, y_max, time_between_exposures):
    #Some variables required for interrogation are computed
    image1 = np.asarray(image1, dtype=np.float32)
    image2 = np.asarray(image2, dtype=np.float32)
    image_size = len(image1)
    interrogation_spots_xcount = int((x_max - x_min) / spot_size * spot_overlap_factor)
    spot_half_length = int(spot_size / 2 / (x_max - x_min) * image_size)

    #More variables required for interrogation are computed
    x_scale = (x_max - x_min) / image_size
    z_scale = (y_max - y_min) / image_size
    if np.abs((x_max - x_min) - (y_max - y_min)) > 1e-6:
        raise RuntimeError("X and Z scales not equal in interrogation. See interrogate_images function.")
    max_s = int(np.ceil(max_measurable_velocity * time_between_exposures / x_scale))

    min_interrogation_pixel = spot_half_length + max_s
    max_interrogation_pixel = image_size - 1 - spot_half_length - max_s

    # The interrogation spot centre locations are defined on a regular grid.
    grid = np.arange(1, interrogation_spots_xcount + 1, dtype=np.float64)
    interrogation_spots = np.stack(np.meshgrid(grid, grid, indexing="ij"), axis=-1).reshape(-1, 2)
    interrogation_spots = ((interrogation_spots * (max_interrogation_pixel - min_interrogation_pixel)
                            / (interrogation_spots_xcount + 1)) + min_interrogation_pixel)

    # The interrogation is performed in parallel at the spot level.
    centre_pixels = interrogation_spots.astype(np.int32, copy=False)
    if len(centre_pixels) == 0:
        shift_vectors = np.zeros((0, 2), dtype=np.float64)
    else:
        max_workers = min(len(centre_pixels), os.cpu_count() or 1)
        if max_workers > 1:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                shift_vectors = np.array(
                    list(executor.map(
                        _interrogate_spot_for_centre,
                        repeat(image1),
                        repeat(image2),
                        repeat(spot_half_length),
                        repeat(max_s),
                        centre_pixels[:, 0],
                        centre_pixels[:, 1],
                    )),
                    dtype=np.float64,
                )
        else:
            shift_vectors = np.array([
                _interrogate_spot_for_centre(image1, image2, spot_half_length, max_s, x, y)
                for x, y in centre_pixels
            ], dtype=np.float64)

    velocity_vectors = (x_scale / time_between_exposures) * shift_vectors
    interrogation_spots[:, 0] = interrogation_spots[:, 0] * x_scale + x_min   #Project pixel x to displacement x
    interrogation_spots[:, 1] = y_max - (interrogation_spots[:, 1] * z_scale) #Project pixel z to displacement z

    return interrogation_spots, velocity_vectors

def interrogate_images(image1, image2, interrogation_properties, camera, double_exposure_properties):
    print("Spot half length: ", int(interrogation_properties.get_spot_size() / 2 / (camera.get_x_max() - camera.get_x_min()) * len(image1)))
    print("Max s: ", int(np.ceil(interrogation_properties.get_max_measurable_velocity() * double_exposure_properties.get_interframing_time() / ((camera.get_x_max() - camera.get_x_min()) / len(image1)))))
    return _interrogate_images(image1, image2, interrogation_properties.get_max_measurable_velocity(),
                                interrogation_properties.get_spot_size(), interrogation_properties.get_spot_overlap_factor(),
                                camera.get_x_min(), camera.get_x_max(), camera.get_y_min(), camera.get_y_max(),
                                double_exposure_properties.get_interframing_time())

def plot_vectors(coordinates, vectors, scale, x_bounds, y_bounds, x_label = "x position", y_label = "y position", title = "Vector field"):
    """Plot a vector field using matplotlib.
    Args:
        coordinates: Array of shape (n, 2) with x and y coordinates of the vector origins
        vectors: Array of shape (n, 2) with x and y components of the vectors
        scale: Scaling factor for the vectors in the plot
        x_bounds: Tuple (x_min, x_max) for the plot limits in x direction
        y_bounds: Tuple (y_min, y_max) for the plot limits in y direction
        x_label: Label for the x-axis
        y_label: Label for the y-axis
        title: Title of the plot
    """
    plt.quiver(coordinates[:,0],coordinates[:,1],vectors[:,0],vectors[:,1], scale=scale)
    ax = plt.gca()
    ax.set_aspect('equal')
    ax.set_xlim(x_bounds)
    ax.set_ylim(y_bounds)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.show()

def plot_vector_field(coordinates, vectors, cylinder_diameter, characteristic_velocity, w_0,
                      title = "Velocity field", x_label = "x position (m)", y_label = "y position (m)",
                      arrow_fraction=0.12, scale_override=None, velocity_multiple_max=2.0):
    """Create a publication-quality vector field plot and display it.

    Inputs:
        coordinates: (N,2) array of vector base positions (x,y) at their actual locations
        vectors: (N,2) array of vector components (u,v)
        cylinder_diameter: physical diameter of the cylinder (same units as coordinates)
        characteristic_velocity: characteristic velocity for normalisation
        w_0: physical width and height of the plot domain (data units)
        arrow_fraction: fraction of cylinder diameter that the largest arrow should occupy (default 0.12)
        scale_override: if provided, use this value for matplotlib.quiver 'scale' (bypasses auto-scaling)
        velocity_multiple_max: colormap reaches maximum at velocity_multiple_max * characteristic_velocity (default 2.0)

    Behaviour:
        - The cylinder is drawn centred at the origin (0,0) with diameter cylinder_diameter.
        - The plot window is centred at (d_c/2, d_c/2) with width and height w_0.
        - Axis limits: [d_c/2 - w_0/2, d_c/2 + w_0/2] in both x and y.
        - The plot is displayed with plt.show() and the function also returns (fig, ax).
    """
    import matplotlib.tri as mtri

    coords = np.asarray(coordinates)
    vecs = np.asarray(vectors)
    if coords.shape[0] != vecs.shape[0]:
        raise ValueError('coordinates and vectors must have the same length')

    d_c = float(cylinder_diameter)
    w_0 = float(w_0)
    velocity_multiple_max = float(velocity_multiple_max)

    # Compute non-dimensional speed: normalized by characteristic_velocity and velocity_multiple_max, capped at [0,1]
    speeds = np.linalg.norm(vecs, axis=1)
    nondim = speeds / (float(characteristic_velocity) * velocity_multiple_max)
    nondim = np.clip(nondim, 0.0, 1.0)

    # Create triangulation for smooth background using actual coordinates
    triang = mtri.Triangulation(coords[:,0], coords[:,1])

    # Figure aesthetics (square)
    fig, ax = plt.subplots(figsize=(6, 6))
    # Background field
    tpc = ax.tripcolor(triang, nondim, cmap='turbo', shading='gouraud', vmin=0, vmax=1)
    tpc.set_zorder(0)
    cbar = fig.colorbar(tpc, ax=ax, pad=0.02, shrink=0.7)
    cbar.set_label('$|\\mathbf{u}|/U_c$', fontsize=10)
    # Set colorbar ticks to reflect velocity_multiple_max scale
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    cbar.set_ticklabels([f'{v*velocity_multiple_max:.2f}' for v in [0, 0.25, 0.5, 0.75, 1.0]])

    # Determine quiver scaling
    # The reference vector (characteristic_velocity) should fit in 0.4*w_0 visual units
    max_speed = np.max(speeds) if np.max(speeds) > 0 else float(characteristic_velocity)
    max_ref_visual = 0.4 * w_0
    
    # Scale based on capping reference vector
    scale_from_ref = float(characteristic_velocity) / max_ref_visual
    
    # Visual size of largest arrow with reference scale
    max_arrow_visual = max_speed / scale_from_ref
    
    # Arrow fraction constraint
    max_arrow_from_fraction = arrow_fraction * d_c
    
    if scale_override is not None:
        scale = float(scale_override)
    else:
        # Choose the more restrictive constraint
        if max_arrow_visual > max_arrow_from_fraction:
            scale = max_speed / max_arrow_from_fraction
        else:
            scale = scale_from_ref

    # Plot vectors at their actual locations
    q = ax.quiver(coords[:,0], coords[:,1], vecs[:,0], vecs[:,1], angles='xy', scale_units='xy', scale=scale,
                  width=0.0030, headwidth=3.0, headlength=4.5, headaxislength=3.5, color='k', alpha=0.9)
    q.set_zorder(2)

    # Add reference quiver key: show characteristic_velocity magnitude
    # Cap arrow visual size to 0.4*w_0, but always label with true characteristic_velocity
    ref_len = float(characteristic_velocity)
    max_visual_ref_len = 0.4 * w_0
    actual_visual_ref_len = ref_len / scale  # size in data units with current scale
    if actual_visual_ref_len > max_visual_ref_len:
        # Reduce the data value used for quiverkey to cap visual size
        ref_len_display = max_visual_ref_len * scale
    else:
        ref_len_display = ref_len
    try:
        ax.quiverkey(q, 0.88, 0.1, ref_len_display, f'{ref_len:.2g} m/s', labelpos='E', coordinates='figure')
    except Exception:
        pass

    # Draw cylinder at origin with radius = d_c/2 (in front of colour, behind vectors)
    cylinder_radius = 0.5 * d_c
    circ_white = plt.Circle((0, 0), cylinder_radius, color='white', zorder=1)
    ax.add_patch(circ_white)
    
    # Draw black outline on top
    circ_black = plt.Circle((0, 0), cylinder_radius, fill=False, edgecolor='black', linewidth=2.5, zorder=1.5)
    ax.add_patch(circ_black)
    # Axis formatting
    ax.set_aspect('equal')
    ax.set_xlabel(x_label, fontsize=11)
    ax.set_ylabel(y_label, fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.grid(alpha=0.25)


    # Set plot limits: centre at (d_c/2, d_c/2) with width and height w_0
    plot_centre = 0.5 * d_c
    half_width = 0.5 * w_0
    ax.set_xlim(plot_centre - half_width, plot_centre + half_width)
    ax.set_ylim(plot_centre - half_width, plot_centre + half_width)

    fig.tight_layout()
    plt.show()
    return fig, ax

@njit(parallel=True, nogil=True, fastmath=True)
def _calculate_errors(coordinates, measured_velocities, characteristic_velocity, characteristic_distance):
    true_vectors = np.zeros_like(measured_velocities)
    for i in prange(coordinates.shape[0]):
        true_vectors[i] = _get_velocity_of_potential_flow_around_a_cylinder(coordinates[i], characteristic_velocity, characteristic_distance)
        #mask for particles inside the cylinder
        if np.sqrt(coordinates[i][0]*coordinates[i][0] + coordinates[i][1]*coordinates[i][1]) < characteristic_distance:
            true_vectors[i] = measured_velocities[i]
    return np.subtract(true_vectors, measured_velocities)

def calculate_errors(coordinates, measured_velocities, fluid_flow):
    return _calculate_errors(coordinates, measured_velocities, fluid_flow.get_characteristic_velocity(), fluid_flow.get_characteristic_distance())

@njit(fastmath=True)
def compute_rms(errors):
    return np.sqrt(np.mean(np.sum(np.power(errors, 2), axis=1)))

def PIV_simulation(particle, fluid_flow, camera, double_exposure_properties, interrogation_properties, n_particles, gravity):
    """Simulate a PIV experiment and compute errors compared to the true velocity field.
    Args:
        particle: Instance of the Particle class
        fluid_flow: Potential flow around a cylinder, instance of a subclass of Fluid_flow
        camera: Instance of the Camera class
        double_exposure_properties: Instance of Double_exposure_properties class
        interrogation_properties: Instance of Interrogation_properties class
        n_particles: Number of particles to simulate
        gravity: Gravitational acceleration (in m/s^2)
        
    Returns:
        Tuple containing:
        - image1: First exposure image (numpy array)
        - image2: Second exposure image (numpy array)
        - interrogation_spots: Array of shape (n_spots, 2) with x and y coordinates of interrogation spots
        - velocity_vectors: Array of shape (n_spots, 2) with x and y components of measured velocity vectors
        - errors: Array of shape (n_spots, 2) with x and y components of errors between measured and true velocity vectors
        - rms_error: Root mean square error of the velocity measurements"""
    t = time.time_ns()
    image1, image2 = double_exposure_simulation(n_particles, particle, camera, fluid_flow, double_exposure_properties, gravity)
    t = time.time_ns() - t
    print(f"Double exposure simulation took {t / 1e9:.2f} seconds.")
    t = time.time_ns()
    interrogation_spots, velocity_vectors = interrogate_images(image1, image2, interrogation_properties, camera, double_exposure_properties)
    t = time.time_ns() - t
    print(f"Interrogation took {t / 1e9:.2f} seconds.")
    errors = calculate_errors(interrogation_spots, velocity_vectors, fluid_flow)
    rms_error = compute_rms(errors)
    return image1, image2, interrogation_spots, velocity_vectors, errors, rms_error
