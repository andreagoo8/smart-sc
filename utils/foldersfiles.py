from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATA_DIR       = PROJECT_ROOT / 'data'    / '0-raw'
PROCESSED_DATA_DIR = PROJECT_ROOT / 'data'    / '1-cleaned'
FIGURES_DIR        = PROJECT_ROOT / 'reports' / 'figures'

PARQUETS       = 'parquets'
PICKLES        = 'pickles'
EXCELS         = 'excels'
PSEUDOPATIENTS = 'pps'
JSONS2         = 'jsons-pp-2'
JSONS4         = 'jsons-pp-4'

ESCRIBE       = '1-escribe'
ICONS_RICOV   = '2_1-estr_icons-ricoveri'
ICONS_CASSA   = '2_2-estr_icons-cassa'
ICONS_FARMA   = '2_3-estr_icons-farma'
PAZIENTI      = '3-pazienti'
EUMACO        = '4-eumaco'
EUMACO_BPT    = '4-eumaco-bpt'
EUMACO_SPO2   = '4-eumaco-spo2'
EUMACO_WT     = '4-eumaco-wt'
EUMACO_BPT_s  = 'bp'
EUMACO_SPO2_s = 'spo2'
EUMACO_WT_s   = 'w'
