#!/usr/bin/env python
# wiz_core.py - Core functionality for WizLighting control

import socket
import json
import time
import logging
# Add these lines after the existing imports
from hsv_rgb_helpers import hsv_to_rgb, blend_colors

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("WizCore")

class WizController:
    """Core controller for WiZ lights using UDP communication"""
    
    def __init__(self, port=38899, timeout=1.0):
        """Initialize the controller with default port and timeout"""
        self.port = port
        self.timeout = timeout
    
    def send_command(self, ip, command):
        """Send a UDP command to a WiZ light and return the response"""
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(self.timeout)
        
        try:
            sock.sendto(json.dumps(command).encode(), (ip, self.port))
            try:
                data, addr = sock.recvfrom(1024)
                return json.loads(data.decode())
            except socket.timeout:
                logger.warning(f"No response from {ip}")
                return {"status": "timeout", "message": f"No response from {ip}"}
        except Exception as e:
            logger.error(f"Error sending to {ip}: {e}")
            return {"status": "error", "message": str(e)}
        finally:
            sock.close()
    
    def set_color(self, ip, color, brightness=None, transition_time=0):
        """Set a specific RGB color for a single light"""
        command = {"method": "setPilot", "params": color.copy()}
        
        if brightness is not None:
            command["params"]["dimming"] = int(brightness)
        
        if transition_time > 0:
            command["params"]["transitionPeriod"] = int(transition_time * 1000)  # Convert to ms
        
        return self.send_command(ip, command)
    
    def set_group_color(self, ips, color, brightness=None, transition_time=0):
        """Set the same color for all lights in a list of IPs"""
        results = []
        for ip in ips:
            result = self.set_color(ip, color, brightness, transition_time)
            results.append({"ip": ip, "result": result})
        return results
    
    def turn_on(self, ip, brightness=None):
        """Turn on a light with optional brightness"""
        command = {"method": "setPilot", "params": {"state": True}}
        
        if brightness is not None:
            command["params"]["dimming"] = int(brightness)
        
        return self.send_command(ip, command)
    
    def turn_off(self, ip):
        """Turn off a light"""
        command = {"method": "setPilot", "params": {"state": False}}
        return self.send_command(ip, command)
    
    def get_state(self, ip):
        """Get the current state of a light"""
        command = {"method": "getPilot"}
        return self.send_command(ip, command)

# Default controller instance for convenience
default_controller = WizController()