#!/usr/bin/env python
# wiz_patterns.py - Pattern generators for WizLighting

import time
import logging
import random
import math
from fractions import Fraction

# Import core functionality
from wiz_core import hsv_to_rgb, blend_colors

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("WizPatterns")

# ----- Basic Pattern Functions -----

def solid_color(controller, ips, color, brightness=None):
    """Set all lights to a single color
    
    Args:
        controller: WizController instance
        ips: List of IP addresses
        color: Color dictionary with r,g,b keys
        brightness: Optional brightness (1-100)
    
    Returns:
        list: Results from each light
    """
    return controller.set_group_color(ips, color, brightness)

def alternating_colors(controller, ips, colors, brightness=None):
    """Set alternating colors across a group of lights
    
    Args:
        controller: WizController instance
        ips: List of IP addresses
        colors: List of color dictionaries with r,g,b keys
        brightness: Optional brightness (1-100)
    
    Returns:
        list: Results from each light
    """
    results = []
    for i, ip in enumerate(ips):
        color_idx = i % len(colors)
        result = controller.set_color(ip, colors[color_idx], brightness)
        results.append({"ip": ip, "result": result})
    return results

def gradient(controller, ips, start_color, end_color, brightness=None):
    """Create a gradient of colors across a group of lights
    
    Args:
        controller: WizController instance
        ips: List of IP addresses
        start_color: Starting color dictionary with r,g,b keys
        end_color: Ending color dictionary with r,g,b keys
        brightness: Optional brightness (1-100)
    
    Returns:
        list: Results from each light
    """
    results = []
    num_lights = len(ips)
    
    for i, ip in enumerate(ips):
        ratio = i / (num_lights - 1) if num_lights > 1 else 0
        color = blend_colors(start_color, end_color, ratio)
        result = controller.set_color(ip, color, brightness)
        results.append({"ip": ip, "result": result})
    
    return results

def rainbow(controller, ips, brightness=None, offset=0):
    """Create a rainbow pattern across a group of lights
    
    Args:
        controller: WizController instance
        ips: List of IP addresses
        brightness: Optional brightness (1-100)
        offset: Hue offset (0-360)
    
    Returns:
        list: Results from each light
    """
    results = []
    num_lights = len(ips)
    
    for i, ip in enumerate(ips):
        # Distribute hues evenly across the spectrum
        hue = (i * (360 / num_lights) + offset) % 360
        color = hsv_to_rgb(hue, 1.0, 1.0)
        result = controller.set_color(ip, color, brightness)
        results.append({"ip": ip, "result": result})
    
    return results

def apply_scene(controller, ips, scene_colors, bulb_shift=0, brightness=None):
    """Apply a scene of colors to a group of lights with optional bulb shift
    
    Args:
        controller: WizController instance
        ips: List of IP addresses
        scene_colors: List of color dictionaries with r,g,b keys
        bulb_shift: Number of positions to shift colors (default: 0)
        brightness: Optional brightness (1-100)
    
    Returns:
        list: Results from each light
    """
    results = []
    
    for i, ip in enumerate(ips):
        # Calculate which color to use for this bulb with shift
        color_idx = (i + bulb_shift) % len(scene_colors)
        color = scene_colors[color_idx]
        result = controller.set_color(ip, color, brightness)
        results.append({"ip": ip, "result": result})
    
    return results

# ----- Animated Pattern Functions -----

def pulse(controller, ips, color, min_brightness=5, max_brightness=100, period=2.0, steps=20):
    """Pulse lights between min and max brightness
    
    Args:
        controller: WizController instance
        ips: List of IP addresses
        color: Color dictionary with r,g,b keys
        min_brightness: Minimum brightness (1-100)
        max_brightness: Maximum brightness (1-100)
        period: Time for one complete pulse cycle in seconds
        steps: Number of brightness steps in one direction
    
    Returns:
        None - runs until interrupted
    """
    try:
        step_time = period / (2 * steps)
        while True:
            # Pulse up
            for i in range(steps + 1):
                brightness = min_brightness + (max_brightness - min_brightness) * (i / steps)
                controller.set_group_color(ips, color, int(brightness))
                time.sleep(step_time)
            
            # Pulse down
            for i in range(steps, -1, -1):
                brightness = min_brightness + (max_brightness - min_brightness) * (i / steps)
                controller.set_group_color(ips, color, int(brightness))
                time.sleep(step_time)
    
    except KeyboardInterrupt:
        logger.info("Pulse pattern interrupted")
    except Exception as e:
        logger.error(f"Error in pulse pattern: {e}")

def color_cycle(controller, ips, colors, time_per_color=1.0, fade_time=0.5):
    """Cycle through a list of colors with fading transitions
    
    Args:
        controller: WizController instance
        ips: List of IP addresses
        colors: List of color dictionaries with r,g,b keys
        time_per_color: Time to display each color in seconds
        fade_time: Transition time between colors in seconds
    
    Returns:
        None - runs until interrupted
    """
    try:
        while True:
            for i, color in enumerate(colors):
                next_color = colors[(i + 1) % len(colors)]
                
                # Set the current color
                controller.set_group_color(ips, color)
                
                # Hold for the specified time
                time.sleep(max(0, time_per_color - fade_time))
                
                # Fade to the next color if fade_time > 0
                if fade_time > 0:
                    fade_steps = int(fade_time / 0.05)  # 50ms per step
                    fade_steps = max(2, fade_steps)  # At least 2 steps
                    
                    for step in range(1, fade_steps + 1):
                        ratio = step / fade_steps
                        blend = blend_colors(color, next_color, ratio)
                        controller.set_group_color(ips, blend)
                        time.sleep(fade_time / fade_steps)
    
    except KeyboardInterrupt:
        logger.info("Color cycle pattern interrupted")
    except Exception as e:
        logger.error(f"Error in color cycle pattern: {e}")

def chase(controller, ips, color, background_color={"r":0, "g":0, "b":0}, speed=0.2):
    """Create a chase effect where a single color moves across the lights
    
    Args:
        controller: WizController instance
        ips: List of IP addresses
        color: Moving color dictionary with r,g,b keys
        background_color: Background color dictionary with r,g,b keys
        speed: Time between steps in seconds
    
    Returns:
        None - runs until interrupted
    """
    try:
        num_lights = len(ips)
        
        # First set all lights to background color
        controller.set_group_color(ips, background_color)
        
        while True:
            for pos in range(num_lights):
                # Set all lights to background except the current position
                for i, ip in enumerate(ips):
                    if i == pos:
                        controller.set_color(ip, color)
                    else:
                        controller.set_color(ip, background_color)
                
                time.sleep(speed)
    
    except KeyboardInterrupt:
        logger.info("Chase pattern interrupted")
    except Exception as e:
        logger.error(f"Error in chase pattern: {e}")

def theater_chase(controller, ips, color, background_color={"r":0, "g":0, "b":0}, positions=3, speed=0.2):
    """Create a theater chase effect where every nth light is lit
    
    Args:
        controller: WizController instance
        ips: List of IP addresses
        color: Active color dictionary with r,g,b keys
        background_color: Background color dictionary with r,g,b keys
        positions: Number of positions in the pattern (usually 3)
        speed: Time between steps in seconds
    
    Returns:
        None - runs until interrupted
    """
    try:
        while True:
            for offset in range(positions):
                # Set colors based on position
                for i, ip in enumerate(ips):
                    if i % positions == offset:
                        controller.set_color(ip, color)
                    else:
                        controller.set_color(ip, background_color)
                
                time.sleep(speed)
    
    except KeyboardInterrupt:
        logger.info("Theater chase pattern interrupted")
    except Exception as e:
        logger.error(f"Error in theater chase pattern: {e}")

def rainbow_cycle(controller, ips, speed=0.1, cycles=1):
    """Create a continuously rotating rainbow pattern
    
    Args:
        controller: WizController instance
        ips: List of IP addresses
        speed: Time between steps in seconds
        cycles: Number of complete hue cycles (e.g., 2 = 720 degrees)
    
    Returns:
        None - runs until interrupted
    """
    try:
        steps = 360
        while True:
            for step in range(steps):
                offset = (step * cycles * 360 / steps) % 360
                rainbow(controller, ips, offset=offset)
                time.sleep(speed)
    
    except KeyboardInterrupt:
        logger.info("Rainbow cycle pattern interrupted")
    except Exception as e:
        logger.error(f"Error in rainbow cycle pattern: {e}")

# ----- Advanced Pattern Functions -----

def apply_scene_with_timing(controller, ips, scene_colors, bulb_shift=0, 
                           hold_time=0.5, fade_time=0.5, off_time=0.0):
    """Apply a scene with specific timing parameters
    
    Args:
        controller: WizController instance
        ips: List of IP addresses
        scene_colors: List of color dictionaries with r,g,b keys
        bulb_shift: Number of positions to shift colors
        hold_time: Time to hold each color in seconds
        fade_time: Time for fade transition in seconds
        off_time: Time to remain off between colors in seconds
    
    Returns:
        None - applies the scene once and returns
    """
    for i, ip in enumerate(ips):
        # Calculate which color to use for this bulb with shift
        color_idx = (i + bulb_shift) % len(scene_colors)
        color = scene_colors[color_idx]
        
        if fade_time > 0:
            controller.set_color(ip, color, transition_time=fade_time)
        else:
            controller.set_color(ip, color)
    
    # Hold for the specified time
    time.sleep(hold_time)
    
    # Turn off if off_time is specified
    if off_time > 0:
        for ip in ips:
            controller.turn_off(ip)
        time.sleep(off_time)

def insert_offs(colors, offs_ratio=1/3, multiply_colors=True):
    """Insert off states between colors at a specified ratio
    
    Args:
        colors: List of color dictionaries with r,g,b keys
        offs_ratio: Ratio of offs to colors (e.g., 1/3 means every third color is off)
        multiply_colors: Whether to multiply the color pool to achieve exact ratio
    
    Returns:
        list: New color list with offs inserted
    """
    if offs_ratio <= 0:
        return colors.copy()
    
    # Create a copy to avoid modifying the original
    colors_copy = colors.copy()
    
    # Handle fractional value
    if isinstance(offs_ratio, float) and offs_ratio < 1:
        frac = Fraction(offs_ratio).limit_denominator()  # e.g., Fraction(1,3)
        period = frac.denominator  # For 1/3, period will be 3
        
        # If requested, multiply color pool to achieve an exact ratio
        if multiply_colors:
            colors_copy = colors * period  # e.g., 7 colors * 3 = 21 colors
        
        original_len = len(colors_copy)
        # Calculate insertion indices
        indices = [i for i in range(period, original_len + 1, period)]
        
        # Insert off steps at the indices in reverse order
        for idx in reversed(indices):
            colors_copy.insert(idx, {"name": "Off", "r": 0, "g": 0, "b": 0})
    else:
        # For int or >=1 values, insert off steps after each color
        for i in range(len(colors_copy) - 1, -1, -1):
            for _ in range(int(offs_ratio)):
                colors_copy.insert(i + 1, {"name": "Off", "r": 0, "g": 0, "b": 0})
    
    return colors_copy

def extend_with_reverse(colors, add_reverse=True):
    """Extend colors with a reversed copy of the list
    
    Args:
        colors: List of color dictionaries with r,g,b keys
        add_reverse: Whether to add the reversed copy
    
    Returns:
        list: Extended color list
    """
    if not add_reverse:
        return colors.copy()
    
    # Create a copy to avoid modifying the original
    colors_copy = colors.copy()
    
    # Create a reversed copy excluding first and last items to avoid duplicates
    reversed_colors = colors_copy[1:-1][::-1] if len(colors_copy) > 2 else []
    
    # Extend the original list with the reversed colors
    colors_copy.extend(reversed_colors)
    
    return colors_copy

def extend_with_duplication(colors, duplication_factor=2):
    """Duplicate each color in the list multiple times
    
    Args:
        colors: List of color dictionaries with r,g,b keys
        duplication_factor: Number of times to duplicate each color
    
    Returns:
        list: Extended color list
    """
    if duplication_factor <= 1:
        return colors.copy()
    
    # Create a new list with duplicated colors
    new_colors = []
    for color in colors:
        new_colors.extend([color.copy() for _ in range(duplication_factor)])
    
    return new_colors

def add_timing_to_colors(colors, hold_time=None, fade_time=None, off_time=None, update_existing=True):
    """Add timing information to each color in a list
    
    Args:
        colors: List of color dictionaries with r,g,b keys
        hold_time: Hold time in seconds (or callable that returns time)
        fade_time: Fade time in seconds
        off_time: Off time in seconds
        update_existing: Whether to update existing values or keep them
    
    Returns:
        list: Color list with timing information added
    """
    # Create a copy to avoid modifying the original
    colors_copy = []
    for color in colors:
        color_copy = color.copy()
        
        if hold_time is not None and (update_existing or "hold_time" not in color_copy):
            color_copy["hold_time"] = hold_time() if callable(hold_time) else hold_time
            
        if fade_time is not None and (update_existing or "fade_time" not in color_copy):
            color_copy["fade_time"] = fade_time
            
        if off_time is not None and (update_existing or "off_time" not in color_copy):
            color_copy["off_time"] = off_time
        
        colors_copy.append(color_copy)
    
    return colors_copy

def make_gradient_between_colors(colors, steps=10, max_step=255):
    """Create a smooth gradient between consecutive colors in a list
    
    Args:
        colors: List of color dictionaries with r,g,b keys
        steps: Number of interpolation steps between colors
        max_step: Maximum step size for each RGB component
    
    Returns:
        list: New color list with gradient steps
    """
    if len(colors) < 2 or steps < 1:
        return colors.copy()
    
    new_colors = []
    
    for i in range(len(colors) - 1):
        color1 = colors[i]
        color2 = colors[i + 1]
        
        # Add the first color
        new_colors.append(color1.copy())
        
        # Calculate RGB differences
        r_diff = color2["r"] - color1["r"]
        g_diff = color2["g"] - color1["g"]
        b_diff = color2["b"] - color1["b"]
        
        # Create gradient steps
        for step in range(1, steps + 1):
            # Calculate interpolated values
            r = int(color1["r"] + (step * r_diff / steps))
            g = int(color1["g"] + (step * g_diff / steps))
            b = int(color1["b"] + (step * b_diff / steps))
            
            # Create a new color dictionary
            new_color = {"r": r, "g": g, "b": b}
            
            # Copy any additional properties from color1
            for key, value in color1.items():
                if key not in ("r", "g", "b", "name") and key not in new_color:
                    new_color[key] = value
            
            # Add to the list
            new_colors.append(new_color)
    
    # Add the last color
    new_colors.append(colors[-1].copy())
    
    return new_colors


# ----- Music-driven Pattern Functions -----

def beat_pulse(controller, ips, base_color, pulse_color, brightness=100, duration=0.1):
    """Create a quick pulse effect for beat synchronization
    
    Args:
        controller: WizController instance
        ips: List of IP addresses
        base_color: Base color dictionary with r,g,b keys
        pulse_color: Pulse color dictionary with r,g,b keys
        brightness: Brightness for pulse (1-100)
        duration: Duration of pulse in seconds
    
    Returns:
        None - performs one pulse and returns
    """
    # Pulse to the beat color
    controller.set_group_color(ips, pulse_color, brightness)
    time.sleep(duration)
    
    # Return to base color
    controller.set_group_color(ips, base_color)

def frequency_color_map(controller, ips, bass_level, mid_level, high_level, brightness=None):
    """Map frequency band levels to RGB colors
    
    Args:
        controller: WizController instance
        ips: List of IP addresses
        bass_level: Bass level (0-1)
        mid_level: Mid level (0-1)
        high_level: High level (0-1)
        brightness: Optional brightness override (1-100)
    
    Returns:
        None - sets the colors once and returns
    """
    # Map frequency levels to RGB color
    color = {
        "r": int(bass_level * 255),
        "g": int(mid_level * 255),
        "b": int(high_level * 255)
    }
    
    # If no significant audio, use a minimum value to avoid complete darkness
    if bass_level + mid_level + high_level < 0.1:
        color = {"r": 10, "g": 10, "b": 10}
    
    # Calculate brightness from overall energy if not explicitly provided
    if brightness is None:
        brightness = int(((bass_level + mid_level + high_level) / 3) * 100)
        brightness = max(10, min(100, brightness))  # Clamp between 10-100
    
    # Apply the color to all lights
    controller.set_group_color(ips, color, brightness)

def spectral_color_map(controller, ips, spectral_centroid, brightness=None):
    """Map spectral centroid to a color hue
    
    Args:
        controller: WizController instance
        ips: List of IP addresses
        spectral_centroid: Normalized spectral centroid (0-1)
        brightness: Optional brightness override (1-100)
    
    Returns:
        None - sets the colors once and returns
    """
    # Map spectral centroid to hue (0-360)
    hue = spectral_centroid * 360
    
    # Convert HSV to RGB
    color = hsv_to_rgb(hue, 1.0, 1.0)
    
    # Set brightness if provided
    if brightness is not None:
        controller.set_group_color(ips, color, brightness)
    else:
        controller.set_group_color(ips, color)
def on_off_beat(controller, ips, on_color={"r": 255, "g": 255, "b": 255}, brightness=100, duration=0.1):
    """Create a simple on/off effect for beat synchronization
    
    Args:
        controller: WizController instance
        ips: List of IP addresses
        on_color: Color to use when light is on
        brightness: Brightness for on state (1-100)
        duration: Duration of on state in seconds
    
    Returns:
        None - turns lights on and then off after duration
    """
    # Turn on to the specified color
    controller.set_group_color(ips, on_color, brightness)
    time.sleep(duration)
    
    # Turn off using turn_off method instead of setting black color
    for ip in ips:
        controller.turn_off(ip)
