from rdkit import Chem
import sys

def add_indices(input_file, output_file):
    mol = Chem.MolFromMolFile(input_file, sanitize=True)
    if mol is None:
        raise ValueError("Could not read molecule")

    mol = Chem.RemoveHs(mol)
    Chem.AssignStereochemistry(mol, force=True, cleanIt=True)

    for atom in mol.GetAtoms():
        atom.SetProp("atomLabel", str(atom.GetIdx()))

    writer = Chem.SDWriter(output_file)
    writer.write(mol)
    writer.close()

if __name__ == "__main__":
    add_indices(sys.argv[1], sys.argv[2])