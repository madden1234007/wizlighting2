#!/usr/bin/env python
# hsv_rgb_helpers.py - Color conversion utilities

def hsv_to_rgb(h, s, v):
    """Convert HSV color to RGB
    
    Args:
        h: Hue (0-360)
        s: Saturation (0-1)
        v: Value (0-1)
    
    Returns:
        dict: RGB color dictionary with r,g,b keys (0-255)
    """
    h = h % 360
    
    c = v * s
    x = c * (1 - abs((h / 60) % 2 - 1))
    m = v - c
    
    if 0 <= h < 60:
        r, g, b = c, x, 0
    elif 60 <= h < 120:
        r, g, b = x, c, 0
    elif 120 <= h < 180:
        r, g, b = 0, c, x
    elif 180 <= h < 240:
        r, g, b = 0, x, c
    elif 240 <= h < 300:
        r, g, b = x, 0, c
    else:  # 300 <= h < 360
        r, g, b = c, 0, x
    
    r, g, b = int((r + m) * 255), int((g + m) * 255), int((b + m) * 255)
    
    return {"r": r, "g": g, "b": b}

def blend_colors(color1, color2, ratio):
    """Blend two colors together
    
    Args:
        color1: First color dictionary with r,g,b keys
        color2: Second color dictionary with r,g,b keys
        ratio: Blend ratio (0 = color1, 1 = color2)
    
    Returns:
        dict: Blended RGB color dictionary
    """
    r = int(color1["r"] * (1 - ratio) + color2["r"] * ratio)
    g = int(color1["g"] * (1 - ratio) + color2["g"] * ratio)
    b = int(color1["b"] * (1 - ratio) + color2["g"] * ratio)  # BUG: Should be color2["b"]
    
    return {"r": r, "g": g, "b": b}