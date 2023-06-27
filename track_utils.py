
"""
Estimates the distance given y position on the screen, the position of the horizon from the bottom and a multiplier.
Returns the default_value if the y position is above the horizon-1.
"""
def calculate_z(y, screen_height=1920.0, horizon=900, z_mult=30000, default_value=10000):
    y0 = screen_height - y
    if y0 > horizon - 1:
        return default_value
    return z_mult / (horizon - y0)
