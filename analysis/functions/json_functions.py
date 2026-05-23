import os
import json
# from pathlib import Path

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd


# json
## pretty print jsons
def pretty_print_folder(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        with open(os.path.join(input_dir, filename), "r", encoding="utf-8") as file:
            data = json.load(file)

        name = filename[:-5]  # strip ".json"
        out_path = os.path.join(output_dir, f"{name}-pp.json")

        with open(out_path, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=2, ensure_ascii=False, sort_keys=False)



## returns the structure of a field in a json/dictionary
### simple variant
def show_structure_simple(data):
    '''Takes in input a dictionary and returns a version of it where:
    - keys are kept unchanged
    - values are substituted with the type name (iterable are NOT analyzed recursively)'''

    result = {}

    for key, val in data.items():
        if isinstance(val, dict):
            result[key] = show_structure_simple(val)  # Recursive call
        else:
            result[key] = type(val).__name__
    return result

### digger variant
def show_structure_digger(data):
    '''Takes in input a dictionary and returns a version of it where:
    - keys are kept unchanged
    - values are substituted with the type name (iterable are analyzed recursively)'''
    
    if isinstance(data, Mapping):
        result = {}
        for key, val in data.items():
            result[key] = show_structure_digger(val)
        return result

    if isinstance(data, Sequence) and not isinstance(data, (str, bytes)):
        result = []
        for item in data:
            result.append(show_structure_digger(item))
        return result

    return type(data).__name__

## returns a sample of a field (optional: specify the json file and the instance position)
def get_sample(json_files_dict, json_filenames, field_to_check, *, as_structure=True, json_filename=None, position=0):
    '''
    Takes in input:
    mandatory:
    - a list of json filenames
    - a dictionary {json_filename: json_file_content}
    - the transmission subfield to check
    optionals:
    - bool as_structure option
    - json filename to read (default=first of json_filenames)
    - transmission field instance's position(default=first one)

    Returns:
    - the structure of the instance, if as_structure=True (default)
    - the actual instance, if as_structure=False
    '''

    if json_filename is None:
        json_filename = json_filenames[0]

    # Gets the required instance
    instance = json_files_dict[json_filename]['transmissions'][field_to_check][0][position]

    if as_structure == False:
        return instance

    # Compute the structure of the required instance
    return show_structure_digger(instance)

## returns overview of the consistency of a field among json files
def overview_of_field(json_files_dict, json_filenames, field_to_check, sample_instance, *, position=True):
    '''
    Takes in input:
    - a dictionary {json_filename: json_file_content}
    - a list of json filenames
    - the transmission subfield to check
    - a sample instance to check against
    - the position option

    Returns an overview dictionary {'different': {json_file: [position, ...]}, 'missing': [json_files]} that tells:
    - given a json file which have the field different from the sample one:
        - the instance's position within the file, if position=True (default behaviour)
        - the actual instance structure, if position=False
    - which json files misses the transmission subfield to check
    '''

    # The returned object will have this structure:
    # {'different': {json_file: [position, ...]}, 'missing': [json_files]}
    overview = {'different': {}, 'missing': []}

    # Iterates over the jsons
    for json_name in json_filenames:

        # transmissions will be a dict that contains the actual "transmissions" field of the json
        # {field_to_check: content}
        transmissions = json_files_dict[json_name].get('transmissions', {})

        # If the transmission field to check is NOT present:
        # Append the json name to the missing list ...
        # .. and go directly to the next json file 
        if field_to_check not in transmissions:
            overview['missing'].append(json_name)
            continue
        
        # If the transmission field to check is present in the json:

        # Place in instances the complete list of instances of the field_to_check
        instances = transmissions[field_to_check][0]

        # Iterate over the long list of the instances and set:
        # - t_position as the transmission field instance's position
        # - t_instance as the the transmission field's instance itself
        for t_position, t_instance in enumerate(instances):

            # Get the structure of the transmission field's instance
            instance_structure = show_structure_digger(t_instance)

            # If the sample instance is different from the current instance...
            if sample_instance != instance_structure:

                # Set value as:
                # - t_position if show_positions is True (default)
                # - instance_structure if show_positions is False
                value = t_position if position else instance_structure

                # Add the desired value to the different dictionary
                overview['different'].setdefault(json_name, []).append(value)

    return overview

# checks if the "external" fields except the `transmissions` have the same structure
def compare_general_jsons(json_files_dict, json_filenames, sample_json_name, *, names=True):
    '''
    Takes in input:
    - a list of json filenames
    - a dictionary {json_filename: json_file_content}
    - a sample instance to check against
    - the json_name option

    Returns an overview list that tells:
    - if names == True (default), the name of the json files that have a structure different from the sample instance
    - if names == False, the actual structure of the different json    '''

    different_general_jsons = []
    sample_json = show_structure_simple(json_files_dict[sample_json_name])
    sample_json.pop('transmissions')

    # Iterates over the jsons
    for json_name in json_filenames:
        current_json = show_structure_simple(json_files_dict[json_name])
        current_json.pop('transmissions')

        if sample_json != current_json:

            if names == True:
                different_general_jsons.append(json_name)
            else:
                different_general_jsons.append(current_json)

    return different_general_jsons

# panads df
## helper for json_to_df
def get_deep(dic, path):
    '''
    Helper function that takes in input:
    - a dictionary
    - a path to a value of the dictionary, colon separated
    and returns the corresponding value, even if the key:value pair is nested inside the dictionary

    For example, running the function on `d = {'ext': {'int': 'val'}}`:
    - get_deep(d, 'ext') -> {'int': 'val'}
    - get_deep(d, 'ext:int') -> 'val'
    '''
    
    current = dic
    for key in path.split(':'):
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return None
    return current

## json to dict converter
def json_to_df(json_files_dict, json_filenames, *, just_external=False, transm_field=None, external_json_fields=[], transm_inst_fields=[]):
    '''
    Takes in input:
    - a dictionary {json_filename: json_file_content}
    - a list of json filenames
    - the just_external bool option
    - the transmission subfield to check
    - the external fields common to every json file
    - the fields internal to the transmission subfield to check

    Iterates over the json files and the transmission field instances

    Returns a list containing dictionaries, where each one carries the informations
    of every instance of a specific transmission field,
    merged with the informations of the json itself

    If just_external is True, returns just the external informations of the json
    (number of rows will be equal to the number of json files)
    '''

    data = []

    # Iterates over the jsons
    for json_name in json_filenames:
        
        # current_json is 
        current_json = json_files_dict[json_name]

        # Dictionary containing all the "external" information of a json,
        # common to every instance of a transmission
        helper_dict = {}

        # Iterates over a list of external fields
        for field in external_json_fields:

            # Gets the value of the required field ...
            value = get_deep(current_json, field)
            # and puts it into the helper_dict
            helper_dict[field] = value

        # Return only the fields external to the transmission field
        if just_external == True:
            # Appends the data to the initial list
            data.append(helper_dict)
            continue

        # TODO: Maybe is better to place NAs?
        if transm_field not in current_json['transmissions']:
            continue

        # list containing all the instances of a transmission field
        instances = current_json['transmissions'][transm_field][0]

        for instance in instances:
            # Makes a copy of the common dictionary,
            # to allow to replace it at every loop
            instance_helper_dict = helper_dict.copy()
            
            # Iterates over a list of internal transmission fields
            for field in transm_inst_fields:

               # Gets the value of the required field ... 
                value = get_deep(instance, field)
                # and puts it into the instance_helper_dict
                instance_helper_dict[field] = value

            # Appends the data to the initial list
            data.append(instance_helper_dict)

    return data

## panads df analysis
def columns_consistency(df, column1, column2, *, bidirectional=True, return_number=False):
    '''
    Finds inconsistency between two columns.
    Options:
    - if bidirectional is True (default), checks inconsistencies both between column1-column2 and column2-column1
    - if return_number is True, return only the number of inconsistent rows
    '''

    # Returns a Series where:
    # - index is the list of all the uniqued values of column1
    # - values are the most common column2 value for each column1 value
    mode_mapping_12 = df.groupby(column1)[column2].agg(lambda x: x.mode()[0])

    # Returns a Series where:
    # - index is the original index of df
    # - values are the most common column2 value for the column1 value
    expected_values_12 = df[column1].map(mode_mapping_12)

    # Makes a mask where every row that in column2 has a different expected value is marked as True
    mask = df[column2] != expected_values_12

    if bidirectional == True:
        # Makes a copy of the original mask, since later it will be overrided
        mask_12 = mask

        # Same operations than before
        mode_mapping_21 = df.groupby(column2)[column1].agg(lambda x: x.mode()[0])
        expected_values_21 = df[column2].map(mode_mapping_21)
        mask_21 = df[column1] != expected_values_21

        # Element-wise logical OR between the two masks
        mask = mask_12 | mask_21
    
    # Returns only the two involved columns and the rows where inconsistencies are found
    outliers = df[mask][[column1, column2]]

    if return_number == True:
        return f'Number of inconsistent rows: {len(outliers)}'

    return outliers
