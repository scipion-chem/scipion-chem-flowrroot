'''Script to rebuild the chemistry of docked poses using the rdkit Chem module in the rdkit-env.
A PDBQT pose stores no bond orders and no formal charges, so the molecule perceived from it is often not even
valid. When a template SMILES is known, the template is the result and only the coordinates are taken from the
pose. Poses without a template are kept only if RDKit can sanitize them'''

import sys, os, argparse

from rdkit import Chem, RDLogger

RDLogger.DisableLog('rdApp.*')

def genericCopy(mol):
    '''Copy keeping only elements and connectivity, which is all a PDBQT pose defines'''
    mol = Chem.RWMol(mol)
    for bond in mol.GetBonds():
        bond.SetBondType(Chem.BondType.SINGLE)
        bond.SetIsAromatic(False)
    for atom in mol.GetAtoms():
        atom.SetIsAromatic(False)
        atom.SetFormalCharge(0)
        atom.SetNoImplicit(True)
        atom.SetNumExplicitHs(0)

    mol = mol.GetMol()
    Chem.FastFindRings(mol)
    return mol

def poseOnTemplate(smiles, poseMol):
    '''The template molecule carrying the coordinates of the pose atoms it matches'''
    template = Chem.MolFromSmiles(smiles)
    if template is None:
        return None

    match = genericCopy(poseMol).GetSubstructMatch(genericCopy(template))
    if len(match) != template.GetNumAtoms():
        return None

    poseConf, conf = poseMol.GetConformer(), Chem.Conformer(template.GetNumAtoms())
    for tIdx, pIdx in enumerate(match):
        conf.SetAtomPosition(tIdx, poseConf.GetAtomPosition(pIdx))

    mol = Chem.Mol(template)
    mol.RemoveAllConformers()
    mol.AddConformer(conf, assignId=True)
    Chem.SanitizeMol(mol)
    Chem.AssignStereochemistryFrom3D(mol)
    return mol

def readPose(inFile):
    mols = [mol for mol in Chem.SDMolSupplier(inFile, sanitize=False, removeHs=False)]
    return mols[0] if mols and mols[0] is not None else None

def fixPose(inFile, smiles):
    '''The molecule to score, None if it cannot be built'''
    poseMol = readPose(inFile)
    if poseMol is None or not poseMol.GetNumConformers():
        return None

    if smiles:
        mol = poseOnTemplate(smiles, poseMol)
    else:
        # No template: the pose has to define a valid molecule on its own
        mol = Chem.Mol(poseMol)
        mol = None if Chem.SanitizeMol(mol, catchErrors=True) else Chem.RemoveHs(mol)

    return Chem.AddHs(mol, addCoords=True) if mol is not None else None

if __name__ == "__main__":
    '''Use: python <scriptName> -iD/--inputDir <dirWith_index.sdf_files> -od/--outputDir <outputDirectory>
    [-t/--templates <templatesFile>]
    The templates file holds one "<index>\t<SMILES>" per line. Only the molecules that could be built are
    written to the output directory, keeping their <index>.sdf name
    '''
    parser = argparse.ArgumentParser(description='Rebuilds the chemistry of docked poses using RDKit')
    parser.add_argument('-iD', '--inputDir', type=str, required=True, help='Input directory with <index>.sdf files')
    parser.add_argument('-od', '--outputDir', type=str, required=True, help='Output directory')
    parser.add_argument('-t', '--templates', type=str, required=False, default='',
                        help='File with the template SMILES of the molecules that have one')

    args = parser.parse_args()

    templates = {}
    if args.templates and os.path.exists(args.templates):
        with open(args.templates) as f:
            for line in f:
                idx, smiles = line.rstrip('\n').split('\t')
                templates[idx] = smiles

    for inFile in sorted(os.listdir(args.inputDir)):
        idx = os.path.splitext(inFile)[0]
        mol = fixPose(os.path.join(args.inputDir, inFile), templates.get(idx))
        if mol is None:
            print('Could not build a valid molecule for the pose {}, it will not be scored. A PDBQT pose needs '
                  'the REMARK SMILES header written by Meeko to be rebuilt'.format(inFile))
            continue

        mol.SetProp('_Name', idx)
        writer = Chem.SDWriter(os.path.join(args.outputDir, '{}.sdf'.format(idx)))
        writer.write(mol)
        writer.close()
