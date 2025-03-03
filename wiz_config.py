#!/usr/bin/env python
# wiz_config.py - Configuration for WizLighting system

# Light group configuration - IP addresses for each group
LIGHT_GROUPS = {
    "playroom_fan": [
        "10.0.0.208", "10.0.0.200", "10.0.0.201", "10.0.0.189", 
        "10.0.0.188", "10.0.0.203", "10.0.0.204", "10.0.0.205", 
        "10.0.0.187", "10.0.0.137", "10.0.0.202", "10.0.0.207", "10.0.0.206"
    ],
    "dining": [
        "10.0.0.181", "10.0.0.182", "10.0.0.183", "10.0.0.184", 
        "10.0.0.185", "10.0.0.186", "10.0.0.190", "10.0.0.191"
    ],
    "testing": [
        "10.0.0.207"
    ]
}

# Standard color presets
COLORS = {
    "red": {"r": 255, "g": 0, "b": 0},
    "green": {"r": 0, "g": 255, "b": 0},
    "blue": {"r": 0, "g": 0, "b": 255},
    "white": {"r": 255, "g": 255, "b": 255},
    "purple": {"r": 170, "g": 0, "b": 255},
    "yellow": {"r": 255, "g": 255, "b": 0},
    "cyan": {"r": 0, "g": 255, "b": 255},
    "orange": {"r": 255, "g": 125, "b": 0},
    "pink": {"r": 255, "g": 50, "b": 150},
    "off": {"r": 0, "g": 0, "b": 0}
}

# Color scene presets
COLOR_SCENES = {
    "rainbow": [
        {"name": "Red", "r": 255, "g": 0, "b": 0},
        {"name": "Orange", "r": 255, "g": 127, "b": 0},
        {"name": "Yellow", "r": 255, "g": 255, "b": 0},
        {"name": "Green", "r": 0, "g": 255, "b": 0},
        {"name": "Cyan", "r": 0, "g": 255, "b": 255},
        {"name": "Blue", "r": 0, "g": 0, "b": 255},
        {"name": "Indigo", "r": 75, "g": 0, "b": 130},
        {"name": "Violet", "r": 238, "g": 130, "b": 238}
    ],
    "blue_pink": [
        {"name": "Teal", "r": 20, "g": 200, "b": 220},
        {"name": "Teal 2", "r": 20, "g": 200, "b": 200},
        {"name": "Teal 3", "r": 30, "g": 140, "b": 180},
        {"name": "Teal 4", "r": 30, "g": 140, "b": 140},
        {"name": "Salmon", "r": 180, "g": 60, "b": 70}
    ]
}

def get_light_group(group_name):
    """Get IP addresses for a named light group"""
    if group_name in LIGHT_GROUPS:
        return LIGHT_GROUPS[group_name]
    return None

def get_all_lights():
    """Get all configured light IPs"""
    all_lights = []
    for group in LIGHT_GROUPS.values():
        all_lights.extend(group)
    return all_lights

def get_scene(scene_name):
    """Get a named color scene"""
    if scene_name in COLOR_SCENES:
        return COLOR_SCENES[scene_name]
    return None

def get_color(color_name):
    """Get a named color"""
    if color_name in COLORS:
        return COLORS[color_name]
    return None