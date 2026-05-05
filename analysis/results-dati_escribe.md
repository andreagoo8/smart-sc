# Cleaning done:
* `patient_date_of_birth` nearly all dates formatted in `DD.MM.YYYY` format except for 33 entries, that have the date formatted as `DD/MM/YYYY` (for example row 790); because of this, `dayfirst=True, format='mixed'` are required. Since `errors='raise'` is on and no exceptions have been raised, all the other days are consistent.

# Open questions:
## Personality disorder
* `patient_age`, `patient_date_of_birth`, `patient_race`, `patient_sex` are all inconsistent among pseudopaz. Some pseudopaz have three different sex. Moreover, `patient_age` and `patient_date_of_birth` are inconsistent among them. 

## Final cleanings
* `note2` some of them are the same written in different modes (like `Neuologia`, `Neuro`, `NEURO`, `NEUROLOGIA`)
* `patient_location` some 143 string-int values are written in inconsistent modes, like `OTO` and `Oto`

* `patient_sex` has 994 `0` values, 557 `2` values and 236 `1` values. Resolve what does `2` mean.
* `overreading_phy` has 886 doctor names (32 different ones) and 901 NaNs. Maybe NaN rows are the name of the last inserted doctor?
* `sequence_number` has 885 different int values. Some of them are negatives: it's ok?


## Consistencies across columns
* `patient_height_units`, `patient_race`, `patient_race_str`, `patient_weight_units`
	* 1759 `I` - `0` - `NaN` - `L` values
	* 28 `C` - `U` - `Unknown` - `K` values

* `p_onset`, `p_offset`
	* 506 x `0`
