# python version >= 3.8 

from rdkit import Chem
import numpy as np
import json
import requests
from collections import defaultdict

def open_file(path):
    with open(path, 'r') as fp:
        return fp.read()
    
def write_file(path, file):
    with open(path, 'w') as fp:
        fp.write(file)

def query_pdb_normal(query):
    # HTTP request headers
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json"
    }

    url = "https://search.rcsb.org/rcsbsearch/v2/query"
    req = json.dumps(query)
    response = requests.post(url, req, headers=headers)
    
    # check response
    if response.status_code == 200:
        return response.json()
    else:
        return f"Response Error: {response.status_code}, {response.text}"

def query_pdb_graphql(query, pdb_id):
    # GraphQL endpoint URL
    url = "https://data.rcsb.org/graphql"

    # Set variables
    variables = {
        "id": pdb_id
    }

    # HTTP request headers
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json"
    }

    # send POST request
    response = requests.post(url, json={"query": query, "variables": variables}, headers=headers)

    # check response
    if response.status_code == 200:
        return response.json()
    else:
        return f"Error: {response.status_code}, {response.text}"
    
def get_parsed_chains(pdb_file_path:str) -> dict:
    """
    Parse a PDB file and extract chain information.

    This function reads a PDB file, processes its content, and organizes the data
    into chains and HETATM groups.

    Args:
        pdb_file_path: The file path of the PDB file to be parsed.

    Returns:
        A dictionary where keys are chain IDs (for standard chains)
        or chain_ID_HETATM_name (for HETATM groups), and values are the corresponding
        PDB entries as strings, including a 'TER' line at the end of each group.
    """

    def process_group(group, chain_id):
        """
        Process a group of PDB lines and add them to the result dictionary.

        This helper function handles both standard chains and HETATM groups.
        """
        if not group:
            return

        # Check if the group consists entirely of HETATM lines
        if all(line.startswith('HETATM') for line in group):
            hetatm_groups = defaultdict(list)
            # Group HETATM lines by their molecule name
            for line in group:
                hetatm_name = line[17:20].strip()
                hetatm_groups[hetatm_name].append(line)
            
            # Process each HETATM group
            for hetatm_name, contents in hetatm_groups.items():
                key = f"{chain_id}_{hetatm_name}"
                parsed_chains[key] = '\n'.join(contents + ['TER'])
        else:
            # For standard chains, use the chain ID as the key
            parsed_chains[chain_id] = '\n'.join(group + ['TER'])

    parsed_chains = {}
    current_group = []
    current_chain_id = None

    with open(pdb_file_path, 'r') as file:
        for line in file:
            # Process only ATOM, HETATM, and TER lines
            if line.startswith(('ATOM', 'HETATM', 'TER')):
                line = line.rstrip()
                # Check for new chain or TER line
                if line.startswith('TER') or (current_chain_id and line[21] != current_chain_id):
                    # Process the current group before starting a new one
                    process_group(current_group, current_chain_id)
                    current_group = []
                    if not line.startswith('TER'):
                        # Start a new chain
                        current_chain_id = line[21]
                        current_group.append(line)
                else:
                    # Continue building the current group
                    if not current_chain_id:
                        current_chain_id = line[21]
                    current_group.append(line)

    # Process the last group
    process_group(current_group, current_chain_id)

    return parsed_chains

def get_filtered_chains(parsed_chains: dict) -> dict:
    """
    Filter the parsed PDB chains to retain only one standard chain and one HETATM chain which represents ligand.

    This function takes the output of func 'parse_pdb_chains' and returns a new dictionary
    with at most two entries: one standard chain (the first one encountered) and
    one HETATM chain (the longest one excepts HOH).

    Args:
        parsed_chains: A dictionary as returned by parse_pdb_chains.

    Returns:
        A new dictionary with at most two entries: one standard chain and one HETATM chain.
    """
    filtered_chains = {}
    standard_chain = None
    hetatm_chain = None
    max_hetatm_length = 0

    for key, value in parsed_chains.items():
        if '_' not in key and standard_chain is None:
            # This is the first standard chain encountered
            standard_chain = (key, value)
        elif ('_' in key) and ('HOH' not in key):
            # This is a HETATM chain
            if len(value) > max_hetatm_length:
                hetatm_chain = (key, value)
                max_hetatm_length = len(value)

    if standard_chain:
        filtered_chains[standard_chain[0]] = standard_chain[1]
    if hetatm_chain:
        filtered_chains[hetatm_chain[0]] = hetatm_chain[1]

    return filtered_chains

def extract_pocket(protein_pdb:str, ligand_pdb:str, distance_cutoff=5) -> str:
    '''
    Extract pocket atoms from protein_pdb.
    
    The pocket is composed of protein atom pdb lines whose distance from the ligand atom is within a cutoff.
    '''
    p_total_coords = []
    l_total_coords = []

    for p_line in protein_pdb.splitlines()[:-1]:
        p_coords = p_line[30:38], p_line[38:46], p_line[46:54]
        p_coords = [float(v) for v in p_coords]
        p_total_coords.append(p_coords)
        
    for l_line in ligand_pdb.splitlines()[:-1]:
        l_coords = l_line[30:38], l_line[38:46], l_line[46:54]
        l_coords = [float(v) for v in l_coords]
        l_total_coords.append(l_coords)

    p_total_coords, l_total_coords = np.array(p_total_coords), np.array(l_total_coords)
    distance_mat = np.sqrt(np.square(p_total_coords[:, np.newaxis, :] - l_total_coords[np.newaxis, :, :]).sum(axis=2))
    p_index = np.nonzero(distance_mat < distance_cutoff)[0]
    p_index = sorted(list(set(p_index)))

    pocket_lines = [protein_pdb.splitlines()[i] for i in p_index] + ['TER']
    
    return '\n'.join(pocket_lines)

# # use example
# if __name__ == '__main__':
#     pdb_file_path = './8V2F.pdb'
#     parsed_chains = get_parsed_chains(pdb_file_path=pdb_file_path)
#     filtered_chains = get_filter_chains(parsed_chains=parsed_chains)
#     p, l = list(filtered_chains.items())[0], list(filtered_chains.items())[1]
#     extract_pocket(p, l, distance_cutoff=5)