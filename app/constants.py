PARAMETER_SPECS = {
    'paste_volume': {
        'nominal': 0.040,       # mm³
        'tolerance': 0.004,     # ±0.004
        'usl': 0.044,
        'lsl': 0.036,
        'unit': 'mm³'
    },
    'stencil_thickness': {
        'nominal': 100,       # mm (100 μm)
        'tolerance': 5,     # ±5 μm
        'usl': 105,
        'lsl': 95,
        'unit': 'µm'
    },
    'paste_viscosity': {
        'nominal': 200.0,       # Pa·s
        'tolerance': 50.0,      # ±50 Pa·s
        'usl': 250.0,
        'lsl': 150.0,
        'unit': 'Pa·s'
    },
    'ambient_rh': {
        'nominal': 40.0,        # %
        'tolerance': 10.0,      # ±10%
        'usl': 50.0,
        'lsl': 30.0,
        'unit': '%'
    },
    'ambient_temperature': {
        'nominal': 23.0,        # °C
        'tolerance': 3.0,       # ±3°C
        'usl': 26.0,
        'lsl': 20.0,
        'unit': '°C'
    }
}

# model constants
HIDDEN_DIMS = [256, 128, 64]
DROPOUT_RATE = 0.3
USE_BATCHNORM = True

# data prep constants
MONGO_URI = 'mongodb://localhost:27017/'
DB = 'pcb_manufacturing'
COL = 'synthetic_boards'
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
GROUP_COL = 'batch_id'
RANDOM_SEED = 42