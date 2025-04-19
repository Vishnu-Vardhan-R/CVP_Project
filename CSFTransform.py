"""
For reference:
https://smin95.github.io/dataviz/understanding-the-contrast-sensitivity-function.html
"""
import cv2
import numpy as np
from scipy.fftpack import fft2, ifft2, fftshift, ifftshift



class CSF_transform():
  
  def __init__(self, age):
    """
    Args:
        age (int): Age in months. Must be one of [1, 3, 8, 48].
    """
    # Parameters for each age (extracted from literature [2](Fig. 1.A))
    self.age = int(age)
    self.parameters = {1: {"gamma_max": 9, "f_max": 0.5, "delta": 0.4, "beta": 0.6},
                      3: {"gamma_max": 15, "f_max": 0.8, "delta": 0.4, "beta": 1.8},
                      8: {"gamma_max": 31, "f_max": 0.9, "delta": 0.4, "beta": 1.5},
                      48: {"gamma_max": 100, "f_max": 3.0, "delta": 0.4, "beta": 4.0}}


  def compute_S_prime(self, log_f, gamma_max, log_f_max, beta) -> np.array:
    """CSF subpart 1:
              
        Computes the S' function for a given set of frequencies.
        
        Args: 
          log_f (np.array): Frequencies in log scale.
          gamma_max (𝛄max): peak sensitivity of the CSF.
          log_f_max: frequency where the peak sensitivity occurs. It also indicates the center of the log-contrast sensitivity function.
          beta (𝛽): width of the sensitivity function, defined at half the maximum sensitivity (𝛄max).
            
        Returns: 
          returns S'(f) which is used when f>fmax
    """
    # Constants
    kappa = np.log10(2)
    beta_prime = np.log10(2 * beta)
    # Calculate S'(f) in log scale
    S_prime = np.log10(gamma_max) - kappa * ((log_f - log_f_max) / (beta_prime / 2))**2
    return S_prime


  def compute_CSF(self, frequencies, gamma_max, log_f_max, delta, beta) -> np.array:
    """CSF subpart 2:
              
        Computes the CSF function for a given set of frequencies.
        
        Args: 
          frequencies (np.array): Frequencies in cycles per degree.
          gamma_max (𝛄max): peak sensitivity of the CSF.
          log_f_max: frequency where the peak sensitivity occurs. It also indicates the center of the log-contrast sensitivity function.
          delta (𝛿): threshold for the CSF function.
          beta (𝛽): width of the sensitivity function, defined at half the maximum sensitivity (𝛄max).
            
        Returns: 
          returns the CSF curve which is used when f<fmax
    """
    # Convert linear frequency to log scale
    log_f = np.log10(frequencies + 1e-5)
    # Calculate S'(log_f)
    S_prime = self.compute_S_prime(log_f, gamma_max, log_f_max, beta)
    S = np.where((log_f < log_f_max) & (S_prime < np.log10(gamma_max) - delta), np.log10(gamma_max) - delta, S_prime)
    return 10**S  


  def __call__(self, image) -> np.array:
    """Apply the CSF transform to an image.
    Args:
        image (np.array): Input image in RGB format.
    Returns:
        np.array: Transformed image with adjusted color channels.
    """
    gamma_max, f_max, delta, beta = self.parameters[self.age].values()

    # PIL image to cv2 image
    # image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
    
    # Convert image to float32 and normalize to [0, 1]
    image = np.array(image) / 255.0

    # Initialize an empty list to store the processed color channels
    adjusted_channels = []

    # Create a frequency grid
    rows, cols = image.shape[:2]
    u = np.arange(-rows // 2, rows // 2)
    v = np.arange(-cols // 2, cols // 2)
    U, V = np.meshgrid(v, u)
    frequencies = np.sqrt(U**2 + V**2)  # Calculate radial frequency

    # Process each color channel separately
    for i in range(3):  # Loop over the R, G, and B channels
      # Apply FFT to the current channel
      fft_channel = fft2(image[:, :, i])
      fft_channel_shifted = fftshift(fft_channel)  # Shift zero frequency to the center
      
      csf = self.compute_CSF(frequencies, gamma_max, f_max, delta, beta)

      # Apply the CSF to the FFT coefficients
      csf = csf / np.max(csf) #normalise the CSF map
      adjusted_fft_channel = fft_channel_shifted * csf

      # Inverse FFT to get the adjusted image channel
      adjusted_fft_channel = ifftshift(adjusted_fft_channel)  # Shift back
      adjusted_channel = np.real(ifft2(adjusted_fft_channel))  # Take the real part
      adjusted_channel = np.clip(adjusted_channel, 0, 1)  # Clip to valid range

      # Add the adjusted channel to the list
      adjusted_channels.append(adjusted_channel)

    # Stack the adjusted channels along the third axis to form an RGB image
    adjusted_image = np.stack(adjusted_channels, axis=2)
    
    return adjusted_image