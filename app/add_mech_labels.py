def label_mechanisms(df):
    """
    Label mechanism failure causes based on physics rules
    
    Rules:
    1. Aperture overfill: StencilThickness > USL
    2. PostPrintSpread: PasteViscosity < LSL AND AmbientRh > USL
    3. Poorpastetransfer: StencilThickness < LSL OR PasteViscosity > USL
    
    Args:
        df: DataFrame with process parameters
    
    Returns:
        DataFrame with added mechanism columns (binary 0/1)
    """
    # Define specification limits (from your specs)
    STENCIL_THICKNESS_USL = 110     # um
    STENCIL_THICKNESS_LSL = 95      # um
    PASTE_VISCOSITY_USL = 250       # Pa·s
    PASTE_VISCOSITY_LSL = 150       # Pa·s
    AMBIENT_RH_USL = 50             # %
    
    df = df.copy()
    
    # Rule 1: Aperture overfill
    df['apertureoverfill'] = (
        df['stencil_thickness'] > STENCIL_THICKNESS_USL
    ).astype(int)
    
    # Rule 2: PostPrintSpread
    df['postprintspread'] = (
        (df['paste_viscosity'] < PASTE_VISCOSITY_LSL) & 
        (df['ambient_rh'] > AMBIENT_RH_USL)
    ).astype(int)
    
    # Rule 3: Poorpastetransfer
    df['poorpastetransfer'] = (
        (df['stencil_thickness'] < STENCIL_THICKNESS_LSL) |
        (df['paste_viscosity'] > PASTE_VISCOSITY_USL)
    ).astype(int)
    
    return df

def validate_mechanism_labels(df):
    """
    Validate mechanism labels and print statistics
    
    Args:
        df: DataFrame with mechanism labels
    """
    mechanism_cols = ['apertureoverfill', 'postprintspread', 'poorpastetransfer']
    
    print("\n" + "="*60)
    print("MECHANISM LABEL STATISTICS")
    print("="*60)
    
    for col in mechanism_cols:
        if col in df.columns:
            count = df[col].sum()
            pct = df[col].mean() * 100
            print(f"{col:20s}: {count:6d} ({pct:5.2f}%)")
    
    # Co-occurrence analysis
    print("\n" + "="*60)
    print("MECHANISM CO-OCCURRENCE")
    print("="*60)
    
    # Check if we have the mechanisms
    if all(col in df.columns for col in ['apertureoverfill', 'postprintspread', 'poorpastetransfer']):
        all_three = (
            (df['apertureoverfill'] == 1) & 
            (df['postprintspread'] == 1) & 
            (df['poorpastetransfer'] == 1)
        ).sum()
        
        any_two = (
            ((df['apertureoverfill'] == 1) & (df['postprintspread'] == 1)) |
            ((df['apertureoverfill'] == 1) & (df['poorpastetransfer'] == 1)) |
            ((df['postprintspread'] == 1) & (df['poorpastetransfer'] == 1))
        ).sum()
        
        exactly_one = (
            (df['apertureoverfill'] + df['postprintspread'] + df['poorpastetransfer']) == 1
        ).sum()
        
        none = (
            (df['apertureoverfill'] == 0) & 
            (df['postprintspread'] == 0) & 
            (df['poorpastetransfer'] == 0)
        ).sum()
        
        print(f"All 3 mechanisms:  {all_three:6d} ({all_three/len(df)*100:5.2f}%)")
        print(f"Any 2 mechanisms:  {any_two:6d} ({any_two/len(df)*100:5.2f}%)")
        print(f"Exactly 1:         {exactly_one:6d} ({exactly_one/len(df)*100:5.2f}%)")
        print(f"No mechanisms:     {none:6d} ({none/len(df)*100:5.2f}%)")
    
    print("="*60)