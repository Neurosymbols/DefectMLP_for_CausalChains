def get_defect_description(defect_label: int) -> str:
    """Get human-readable defect description"""
    descriptions = {
        0: "No defect detected - board passes quality inspection",
        1: "Open Circuit - incomplete or missing solder joints causing electrical disconnection",
        2: "Solder Bridging - unwanted solder connections between adjacent pads causing electrical shorts"
    }
    return descriptions.get(defect_label, "Unknown defect")

def get_mechanism_description(present: bool, probability: float) -> str:
    """Get human-readable mechanism description"""
    if present:
        return (f"Poor paste transfer detected ({probability*100:.1f}% probability) - "
                f"insufficient paste transfer from stencil to board pads")
    else:
        return (f"No mechanism detected ({(1-probability)*100:.1f}% confidence) - "
                f"paste transfer appears normal")

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