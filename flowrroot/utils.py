
import json
import string
import re

import os, glob
import pyworkflow.protocol.params as params
from pwem.protocols import EMProtocol
from pyworkflow.object import String
import shutil

from pwchem import Plugin
from pwchem.constants import RDKIT_DIC
from pwem.convert import cifToPdb
from flowrroot.constants import FLOWR_DIC
from pwchem.utils import pdbqt2other

def _convertFiles(self):
    struct = self.inputAtomStruct.get()
    fileName = struct.getFileName()
    base = os.path.splitext(os.path.basename(fileName))[0]
    outFile = self._getExtraPath(base + '.pdb')

    if fileName.lower().endswith('.cif'):
        cifToPdb(fileName, outFile)
    elif fileName.lower().endswith('.pdbqt'):
        pdbqt2other(self, fileName, outFile)

    return outFile

def _createLigandFile(self):
    molFile = self._getExtraPath('ligands.txt')
    outPath = self._getExtraPath('ligands.sdf')
    with open(molFile, 'w') as f:
        for mol in self.inputSetOfMols.get():
            f.write(os.path.abspath(mol.getFileName()) + '\n')

    args = ['-i', os.path.abspath(molFile), '-o', os.path.abspath(outPath), '-of', 'sdf']

    Plugin.runScript(self, 'rdkit_IO.py', args, env=RDKIT_DIC, cwd=self._getExtraPath())

def _predictAffinity(self):
    scriptPath = os.path.join(Plugin.getVar(FLOWR_DIC['home']), 'flowr_root/flowr/predict/predict_from_pdb.py')
    modelPath = os.path.join(Plugin.getVar(FLOWR_DIC['home']), 'checkpoints/flowr_root_v2.1.ckpt')
    outPath = self._getExtraPath('denovo_affinity')
    struct = self.inputAtomStruct.get()
    fileName = struct.getFileName()
    base = os.path.splitext(os.path.basename(fileName))[0]
    outFile = self._getExtraPath(base + '.pdb')
    if not os.path.exists(outFile):
        outFile = os.path.abspath(self.inputAtomStruct.get().getFileName())

    if self.optimizeLigands.get():
        sdfFiles = glob.glob(os.path.join(self._getExtraPath(), '*optimized*.sdf'))
        sdfFiles = [f for f in sdfFiles if os.path.getsize(f) > 0]
        if not sdfFiles:
            sdfFiles = glob.glob(os.path.join(self._getExtraPath(), '*.sdf'))
            sdfFiles = [f for f in sdfFiles if os.path.getsize(f) > 0]
    else:
        sdfFiles = glob.glob(os.path.join(self._getExtraPath(), '*.sdf'))
        sdfFiles = [f for f in sdfFiles if os.path.getsize(f) > 0]

    sdfFile = sdfFiles[0]
    args = [
        '--pdb_file', outFile,
        '--ligand_file', sdfFile,
        '--multiple_ligands',
        '--arch', 'pocket',  # NEEDS to be this value bc of the model
        '--pocket_type', 'holo',  # NEEDS to be this value bc of the model
        '--pocket_cutoff', self.pocketCutoff.get(),
        '--coord_noise_scale', self.noiseScale.get(),
        '--num_workers', (self.numberOfThreads.get()),
        '--ckpt_path', modelPath,
        '--save_dir', os.path.abspath(outPath),
        '--max_pocket_size', self.maxPocketSize.get()
    ]
    if self.cutPocket.get(): args.append('--cut_pocket')
    if self.sampleMolSizes.get(): args.append('--sample_mol_sizes')

    args.extend(['--seed', self.seed.get()])

    if self.useGpu.get():
        args.append('--gpus')
        args.append('1')

    fullProgram = (
        f"export PYTHONPATH={os.path.join(Plugin.getVar(FLOWR_DIC['home']), 'flowr_root')}:$PYTHONPATH && "
        f"python"
    )

    args_str = " ".join(map(str, args))

    Plugin.runCondaCommand(
        self,
        program=fullProgram,
        args=f"{scriptPath} {args_str}",
        condaDic=FLOWR_DIC,
        cwd=Plugin.getVar(self._getExtraPath())
    )

def getLigandIndex(self):
    for i, mol in enumerate(self.inputSetOfMols.get()):
        if str(mol) == self.referenceMol.get():
            return i

def _createArgs(self, outFile, outPath):
    ligIdx = getLigandIndex(self)
    modelPath = os.path.join(Plugin.getVar(FLOWR_DIC['home']), 'checkpoints/flowr_root_v2.1.ckpt')

    args = [
        '--pdb_file', outFile,
        '--ligand_file', (os.path.abspath(self._getExtraPath('ligands.sdf'))),
        '--ligand_id', ligIdx,
        '--arch', 'pocket',  # NEEDS to be this value bc of the model
        '--pocket_type', 'holo',  # NEEDS to be this value bc of the model
        '--pocket_cutoff', self.pocketCutoff.get(),
        '--sample_n_molecules_per_target', self.nMolecules.get(),
        '--max_sample_iter', self.sampleIters.get(),
        '--coord_noise_scale', self.noiseScale.get(),
        '--batch_cost', self.batchCost.get(),
        '--num_workers', (self.numberOfThreads.get()),
        '--ckpt_path', modelPath,
        '--save_dir', os.path.abspath(outPath),
        '--filter_valid_unique',
        '--filter_pb_valid',
        '--min_pocket_size', self.minPocketSize.get(),
        '--max_pocket_size', self.maxPocketSize.get()
    ]
    if self.cutPocket.get(): args.append('--cut_pocket')
    if self.sampleMolSizes.get(): args.append('--sample_mol_sizes')
    args.extend(['--seed', self.seed.get()])
    if self.optimizeLigands.get():
        args.append('--optimize_gen_ligs')
    if self.kekulize.get():
        args.append('--kekulize')

    if self.useGpu.get():
        args.append('--gpus')
        args.append('1')
    return args

