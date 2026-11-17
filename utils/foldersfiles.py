from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# folder paths
RAW_DATA_DIR               = PROJECT_ROOT / 'data'    / '0-raw'
CLEANED_ORIGINALS_DATA_DIR = PROJECT_ROOT / 'data'    / '1-cleaned_originals'
CLEANED_PROCESSED_DATA_DIR = PROJECT_ROOT / 'data'    / '2-cleaned_processed'
FIGURES_DIR                = PROJECT_ROOT / 'reports' / 'figures'
UTILS                      = PROJECT_ROOT / 'utils'

# folders names
PARQUETS       = 'parquets'
PICKLES        = 'pickles'
EXCELS         = 'excels'
PSEUDOPATIENTS = 'pps'
JSONS2         = 'jsons-pp-2'
JSONS4         = 'jsons-pp-4'
JSONST         = 'jsons-pp-t'

# dataset names
ESCRIBE     = '1-escribe'
ICONS_RICOV = '2_1-estr_icons-ricoveri'
ICONS_CASSA = '2_2-estr_icons-cassa'
ICONS_FARMA = '2_3-estr_icons-farma'
PAZIENTI    = '3-pazienti'
EUMACO      = '4-eumaco'
EUMACO_BP   = '4-eumaco-bp'
EUMACO_SPO2 = '4-eumaco-spo2'
EUMACO_WT   = '4-eumaco-wt'

# eumaco transmission fields abbreviations
BP_SHORT   = 'bp'
SPO2_SHORT = 'spo2'
WT_SHORT   = 'wt'
