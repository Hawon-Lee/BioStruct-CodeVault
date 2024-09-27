from rdkit import Chem

def create_mol_from_smiles(smiles: str, retry_with_sanitize: bool = False) -> Chem.Mol:
    '''
    Create RDKit mol object from input SMILES string.
    
    Args:
        smiles (str): Input SMILES string
        retry_with_sanitize (bool): Whether to retry with sanitize=False when initial creation returns none
    
    Returns: 
        Chem.Mol: Created RDKit molecule object
    '''
    m = Chem.MolFromSmiles(smiles)
    # Retry with sanitize=False if option is set and initial attempt failed
    if retry_with_sanitize and m is None:
        m = Chem.MolFromSmiles(smiles, sanitize=False)
    # Raise error if molecule creation ultimately fails
    if m is None:
        raise ValueError(f'Failed to create molecule from smiles: {smiles}')
    return m

def create_mol_from_file(file_path: str, retry_with_sanizite: bool = False) -> Chem.Mol:
    '''
    Create RDKit mol object from input structure file.
    Automatically detect 'sdf, mol2, pdb' file extension.
    
    Args:
        file_path (str): Input file path which ends with file extension of sdf, mol2 or pdb.
        retry_with_sanitize (bool): Whether to retry with sanitize=False when initial creation returns none
    
    Returns:
        Chem.Mol: Created RDKit molecule object
    '''
    
    # Mapping of file extensions to reader functions
    file_readers = {
        'sdf':Chem.MolFromMolFile,
        'mol2':Chem.MolFromMol2File,
        'pdb':Chem.MolFromPDBFile,
    }
    
    # Extract file extension and map to appropriate reader function
    file_extension = file_path.split('.')[-1].lower()
    reader_func = file_readers.get(file_extension)
    
    # Raise error when input file extension is not supported.
    if reader_func is None:
        raise ValueError(f"Unsupported file format : {file_extension}")
    
    # Create mol
    m = reader_func(file_path)
    # Retry with sanitize=False if option is set and initial attempt failed
    if retry_with_sanizite and m is None:
        m = reader_func(file_path, sanitize=False)
    
    # Raise error when attempts with sanitize=False also failed.
    if m is None:
        raise ValueError(f'Failed to create molecule from file: {file_path}')
        
    return m