def adjust_frequency_to_unit(value: float, unit: str):
    """
    Adjusts a frequency to a given unit.
    """
    if unit == "Hz":
        return value
    elif unit == "kHz":
        return value * 1000
    elif unit == "MHz":
        return value * 1000000
    else:
        raise ValueError(f"Invalid unit: {unit}")