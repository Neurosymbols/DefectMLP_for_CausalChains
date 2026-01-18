def get_defect_description(defect_label: int) -> str:
    """Get human-readable defect description"""
    descriptions = {
        0: "No defect detected - board passes quality inspection",
        1: "Open Circuit - incomplete or missing solder joints causing electrical disconnection",
        2: "Solder Bridging - unwanted solder connections between adjacent pads causing electrical shorts"
    }
    return descriptions.get(defect_label, "Unknown defect")

def get_mechanism_description(mech_label:int) -> str:
    """Get human-readable mechanism description"""
    descriptions = {
        0: "Aperture overfill observed, indicating excess solder paste deposited during printing.",
        1: "No mechanism-related issues detected; the board meets process and quality requirements.",
        2: "Poor paste transfer identified, suggesting incomplete or inconsistent solder paste release from the stencil."
    }
    return descriptions.get(mech_label, "Unknown mechanism")

def get_violation_description(violation_list: list, high_risk_list: list) -> str:
    """Get human-readable violation description"""
    if not violation_list and not high_risk_list:
        return "All parameters within specification - process operating normally"
    elif not violation_list:
        return (f"Warning: {len(high_risk_list)} parameter(s) approaching limits - "
                f"no violations yet but monitor closely")
    else:
        return (f"Critical: {len(violation_list)} parameter violation(s) detected - "
                f"immediate corrective action required")