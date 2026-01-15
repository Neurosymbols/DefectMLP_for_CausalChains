# in mlp directory
import pandas as pd

def label_mechanisms(df):
    """
    Label mechanism failure causes based on physics rules
    
    Mechanisms are MUTUALLY EXCLUSIVE:
    - Aperture overfill: TOO MUCH paste
    - Poor paste transfer: TOO LITTLE paste
    
    Priority: If both conditions true, aperture overfill takes precedence
    
    Rules:
    1. Aperture overfill: 
       - StencilThickness > USL OR PasteViscosity < LSL
       - PasteViscosity < LSL AND AmbientRh > USL
    
    2. Poor paste transfer:
       - StencilThickness < LSL OR PasteViscosity > USL
       - AmbientRh < LSL AND PasteViscosity > USL
    
    Args:
        df: DataFrame with process parameters
    
    Returns:
        DataFrame with added mechanism columns (binary 0/1)
    """
    # Define specification limits
    STENCIL_THICKNESS_USL = 105     # µm
    STENCIL_THICKNESS_LSL = 95      # µm
    PASTE_VISCOSITY_USL = 250       # Pa·s
    PASTE_VISCOSITY_LSL = 150       # Pa·s
    AMBIENT_RH_USL = 50             # %
    AMBIENT_RH_LSL = 30             # %
    
    df = df.copy()
    
    # ========================================================================
    # RULE 1: APERTURE OVERFILL (too much paste)
    # ========================================================================
    aperture_overfill_conditions = (
        # Condition 1: High stencil OR low viscosity
        (df['stencil_thickness'] > STENCIL_THICKNESS_USL) |
        (df['paste_viscosity'] < PASTE_VISCOSITY_LSL) |
        # Condition 2: Low viscosity AND high RH
        ((df['paste_viscosity'] < PASTE_VISCOSITY_LSL) & 
         (df['ambient_rh'] > AMBIENT_RH_USL))
    )
    
    # ========================================================================
    # RULE 2: POOR PASTE TRANSFER (too little paste)
    # ========================================================================
    poor_paste_transfer_conditions = (
        # Condition 1: Low stencil OR high viscosity
        (df['stencil_thickness'] < STENCIL_THICKNESS_LSL) |
        (df['paste_viscosity'] > PASTE_VISCOSITY_USL) |
        # Condition 2: Low RH AND high viscosity
        ((df['ambient_rh'] < AMBIENT_RH_LSL) & 
         (df['paste_viscosity'] > PASTE_VISCOSITY_USL))
    )
    
    # ========================================================================
    # MUTUAL EXCLUSIVITY: Priority to aperture overfill
    # ========================================================================
    
    # Initialize both as 0
    df['apertureoverfill'] = 0
    df['poorpastetransfer'] = 0
    
    # Assign aperture overfill first (higher priority)
    df.loc[aperture_overfill_conditions, 'apertureoverfill'] = 1
    
    # Assign poor paste transfer ONLY if aperture overfill is NOT present
    df.loc[poor_paste_transfer_conditions & ~aperture_overfill_conditions, 'poorpastetransfer'] = 1
    
    # Verify mutual exclusivity
    assert ((df['apertureoverfill'] == 1) & (df['poorpastetransfer'] == 1)).sum() == 0, \
        "ERROR: Both mechanisms present on same board!"
    
    return df


def validate_mechanism_labels(df):
    """
    Validate mechanism labels and print statistics
    
    Args:
        df: DataFrame with mechanism labels
    """
    mechanism_cols = ['apertureoverfill', 'poorpastetransfer']
    
    print("\n" + "="*60)
    print("MECHANISM LABEL STATISTICS")
    print("="*60)
    
    for col in mechanism_cols:
        if col in df.columns:
            count = df[col].sum()
            pct = df[col].mean() * 100
            print(f"{col:20s}: {count:6d} ({pct:5.2f}%)")
    
    # Mutual exclusivity check
    print("\n" + "="*60)
    print("MUTUAL EXCLUSIVITY CHECK")
    print("="*60)
    
    both = ((df['apertureoverfill'] == 1) & (df['poorpastetransfer'] == 1)).sum()
    overfill_only = ((df['apertureoverfill'] == 1) & (df['poorpastetransfer'] == 0)).sum()
    transfer_only = ((df['apertureoverfill'] == 0) & (df['poorpastetransfer'] == 1)).sum()
    neither = ((df['apertureoverfill'] == 0) & (df['poorpastetransfer'] == 0)).sum()
    
    print(f"Both mechanisms:        {both:6d} ({both/len(df)*100:5.2f}%) [Should be 0!]")
    print(f"Aperture overfill only: {overfill_only:6d} ({overfill_only/len(df)*100:5.2f}%)")
    print(f"Poor transfer only:     {transfer_only:6d} ({transfer_only/len(df)*100:5.2f}%)")
    print(f"No mechanisms:          {neither:6d} ({neither/len(df)*100:5.2f}%)")
    
    if both > 0:
        print("\n⚠️  WARNING: Mutual exclusivity violated!")
    else:
        print("\n✓ Mutual exclusivity verified")
    
    print("="*60)


# df = pd.read_csv("./training_data_200k_v3.csv")
# df = label_mechanisms(df)
# validate_mechanism_labels(df)