import os
import numpy as np
import librosa
from scipy.signal import medfilt


def remove_noise_from_audio(y, sr, noise_factor=1.5, filter_kernel_size=(1, 7)):
    s_full, phase = librosa.magphase(librosa.stft(y))

    num_frames = s_full.shape[1]
    noise_duration_frames = min(int(sr * 0.1 // (sr / num_frames)), num_frames - 1)

    noise_power = np.mean(s_full[:, :noise_duration_frames], axis=1)

    noise_power *= noise_factor

    # Create a mask
    mask = s_full > noise_power[:, None]
    mask = mask.astype(float)

    mask = medfilt(mask, kernel_size=filter_kernel_size)

    s_clean = s_full * mask

    # Reconstruct the cleaned audio
    y_clean = librosa.istft(s_clean * phase)

    return y_clean


# Detect the start of actual sound by finding where amplitude exceeds a threshold
def detect_sound_start(y, sr, threshold=0.01):
  # Calculate amplitude envelope
  frame_length = int(sr * 0.025)  # 25ms frames
  hop_length = int(sr * 0.010)    # 10ms hop
  
  # Get RMS energy for each frame
  rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
  
  # Find first frame that exceeds threshold
  start_frame = 0
  for i, energy in enumerate(rms):
    if energy > threshold:
      start_frame = i
      break
      
  # Convert frame index to sample index
  start_sample = start_frame * hop_length
  
  # Trim audio to start at detected point
  y_trimmed = y[start_sample:]
  
  return y_trimmed


def vocal_compressor(y, sr, threshold=-20.0, ratio=4.0, attack=0.005, release=0.05, makeup_gain=0.0):
    """
    Apply a compressor effect optimized for vocals
    Parameters:
    - y: Input audio signal (numpy array loaded with librosa)
    - sr: Sample rate
    - threshold: Threshold level in dB above which compression starts
    - ratio: Compression ratio (e.g., 4.0 means 4:1)
    - attack: Attack time in seconds
    - release: Release time in seconds
    - makeup_gain: Output gain compensation in dB
    """
    
    # Convert signal to dB scale (take absolute value and apply log)
    y_abs = np.abs(y)
    y_db = 20 * np.log10(y_abs + 1e-10)  # Add small value to avoid log(0)
    
    # Initialize array for gain reduction
    gain_reduction = np.zeros_like(y_db)
    
    # Variables for envelope follower
    env = np.zeros_like(y_db)
    attack_coef = np.exp(-1.0 / (attack * sr))  # Attack smoothing coefficient
    release_coef = np.exp(-1.0 / (release * sr))  # Release smoothing coefficient
    
    # Calculate envelope and apply gain reduction
    for i in range(1, len(y)):
        # Envelope follower: track signal level with attack/release
        env[i] = max(y_db[i], release_coef * env[i-1] + (1 - attack_coef) * y_db[i])
        
        # Apply compression to levels exceeding threshold
        if env[i] > threshold:
            excess = env[i] - threshold
            gain_reduction[i] = excess * (1 - 1/ratio)
    
    # Convert gain reduction from dB to linear scale and apply makeup gain
    gain = np.power(10, -gain_reduction / 20.0)  # Convert dB reduction to amplitude
    gain = gain * np.power(10, makeup_gain / 20.0)  # Apply makeup gain
    
    # Apply gain to original signal
    y_compressed = y * gain
    
    # Prevent clipping by normalizing if needed
    max_val = np.max(np.abs(y_compressed))
    if max_val > 1.0:
        y_compressed = y_compressed / max_val
        
    return y_compressed
