
import json
import math
import string
import re

import os, glob
from pwem.protocols import EMProtocol
from pyworkflow.object import String
import shutil

from pwchem import Plugin
from pwchem.constants import RDKIT_DIC
from pwem.convert import cifToPdb
from flowrroot.constants import FLOWR_DIC
from pwchem.utils import pdbqt2other

ROI_INPUT = 1

def usesROI(protocol):
    '''Whether the protocol defines the pocket with a structural ROI instead of a reference ligand'''
    return hasattr(protocol, 'inputType') and protocol.inputType.get() == ROI_INPUT

def getReceptorFile(protocol):
    '''Receptor the pocket is defined on, taken from the input object itself (a ROI and a docked molecule
    both carry the structure they were obtained on)'''
    inputObj = protocol.inputStructROIs.get() if usesROI(protocol) else protocol.inputSetOfMols.get()
    return inputObj.getProteinFile()

def getSelectedROI(protocol):
    for roi in protocol.inputStructROIs.get():
        if str(roi) == protocol.referenceROI.get():
            return roi.clone()

def _convertFiles(self):
    fileName = getReceptorFile(self)
    base = os.path.splitext(os.path.basename(fileName))[0]
    outFile = self._getExtraPath(base + '.pdb')

    if fileName.lower().endswith('.cif'):
        cifToPdb(fileName, outFile)
    elif fileName.lower().endswith('.pdbqt'):
        pdbqt2other(self, fileName, outFile)
    else:
        outFile = fileName

    return outFile

def _createROILigandFile(self):
    '''FLOWR.root always needs a ligand file: it cuts the pocket around it and takes the number of atoms to
    generate from it. A ROI defines no molecule, so a dummy one is written out of the ROI points, spread over
    the cavity with as many atoms as the molecules to generate'''
    points = _spreadPoints(getSelectedROI(self).getPointsCoords(), self.molSize.get())
    outFile = self._getExtraPath('ligands.sdf')

    lines = ['ROI', '     flowrroot          3D', '',
             '%3d%3d  0  0  0  0  0  0  0  0999 V2000' % (len(points), len(points) - 1)]
    for x, y, z in points:
        lines.append('%10.4f%10.4f%10.4f C   0  0  0  0  0  0  0  0  0  0  0  0' % (x, y, z))
    for i in range(1, len(points)):
        lines.append('%3d%3d  1  0' % (i, i + 1))
    lines += ['M  END', '$$$$', '']

    with open(outFile, 'w') as f:
        f.write('\n'.join(lines))
    return outFile

def _spreadPoints(points, n):
    '''n of the points, as far apart as possible (farthest point sampling), so that the whole cavity is
    covered however few points are asked for'''
    points = [tuple(p) for p in points]
    selected = [points[0]]
    while len(selected) < n:
        selected.append(max(points, key=lambda p: min(sum((a - b) ** 2 for a, b in zip(p, q))
                                                      for q in selected)))
    return selected

def _createLigandFile(self):
    molFile = self._getExtraPath('ligands.txt')
    outPath = self._getExtraPath('ligands.sdf')
    with open(molFile, 'w') as f:
        for mol in self.inputSetOfMols.get():
            f.write(os.path.abspath(mol.getFileName()) + '\n')

    args = ['-i', os.path.abspath(molFile), '-o', os.path.abspath(outPath), '-of', 'sdf']

    Plugin.runScript(self, 'rdkit_IO.py', args, env=RDKIT_DIC, cwd=self._getExtraPath())

def _individualMols(self, folder):
    outPath = self._getExtraPath(folder)

    # FLOWR always writes the generated molecules and, beside them, their protonated and optimized version,
    # which it also filters by PoseBusters validity and can therefore leave empty
    sdfFiles = [f for f in glob.glob(os.path.join(outPath, 'samples_*.sdf')) if os.path.getsize(f) > 0]
    optimized = [f for f in sdfFiles if 'optimized' in os.path.basename(f)]
    generated = [f for f in sdfFiles if 'optimized' not in os.path.basename(f)]

    sdfFile = optimized[0] if self.optimizeLigands.get() and optimized else generated[0]
    # create indiv files
    splitDir = self._getPath()
    os.makedirs(splitDir, exist_ok=True)
    args = [
        "-i", os.path.abspath(sdfFile),
        "-of", "sdf",
        "-od", os.path.abspath(splitDir),
        "--overWrite"
    ]
    Plugin.runScript(
        self,
        'rdkit_IO.py',
        args,
        env=RDKIT_DIC,
        cwd=self._getExtraPath()
    )

def _extractSdfProperties(sdf_path):
    with open(sdf_path, "r") as f:
        return _blockProperties(f.readlines())

def _blockProperties(lines):
    props = {}
    current_key = None
    for line in lines:
        line = line.strip()
        # Detect property key
        if line.startswith(">") and "<" in line and ">" in line:
            match = re.search(r"<(.+?)>", line)
            if match:
                current_key = match.group(1).strip().lower()
            continue
        # Value line
        if current_key and line and not line.startswith(">") and line != "$$$$":
            try:
                props[current_key] = float(line)
            except:
                props[current_key] = line
            current_key = None
    return props

def _sdfMolecules(sdfFile):
    '''Properties, heavy atoms and centre of each molecule of an SDF file, in order'''
    molecules = []
    with open(sdfFile) as f:
        blocks = f.read().split('$$$$')

    for block in blocks:
        lines = block.lstrip('\n').split('\n')
        if len(lines) < 5:
            continue

        atoms, coords = [], []
        for line in lines[4:4 + int(lines[3][:3])]:
            element = line[31:34].strip()
            if element != 'H':
                atoms.append(element)
                coords.append((float(line[0:10]), float(line[10:20]), float(line[20:30])))

        molecules.append({'props': _blockProperties(lines), 'atoms': tuple(sorted(atoms)),
                          'centre': tuple(sum(c[i] for c in coords) / len(coords) for i in range(3))})
    return molecules

def getAffinities(self, molFile):
    '''Affinity properties of a generated molecule. FLOWR only writes them in the unoptimized samples file:
    when the ligands are optimized the molecules kept come from the optimized one, which it writes without any
    property and filters by PoseBusters validity, so each of them has to be matched back to the molecule it was
    optimized from (the one with its same atoms and, since optimization only relaxes the pose, the closest centre)'''
    props = _extractSdfProperties(molFile)
    if props:
        return props

    samples = [f for f in glob.glob(os.path.join(self._getExtraPath(), '*', 'samples_*.sdf'))
               if 'optimized' not in os.path.basename(f)]
    if not samples:
        return {}

    molecule = _sdfMolecules(molFile)[0]
    candidates = [mol for mol in _sdfMolecules(samples[0]) if mol['atoms'] == molecule['atoms']]
    if not candidates:
        return {}

    return min(candidates, key=lambda mol: math.dist(mol['centre'], molecule['centre']))['props']

def getLigandIndex(self):
    if usesROI(self):
        # The dummy ligand written from the ROI is the only molecule of the file
        return 0

    for i, mol in enumerate(self.inputSetOfMols.get()):
        if str(mol) == self.referenceMol.get():
            return i

def _createArgs(self, outFile, outPath):
    ligIdx = getLigandIndex(self)
    if self.model.get() == 0: model = 'flowr_root_v2.1.ckpt'
    else: model = 'flowr_root_v2.2.ckpt'
    modelPath = os.path.join(Plugin.getVar(FLOWR_DIC['home']), f'checkpoints/{model}')

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

def _createFlowrArgs(protocol, outputDir):
    fileName = getReceptorFile(protocol)
    base = os.path.splitext(os.path.basename(fileName))[0]

    outFile = protocol._getExtraPath(base + '.pdb')
    if not os.path.exists(outFile):
        outFile = os.path.abspath(fileName)

    return _createArgs(
        protocol,
        os.path.abspath(outFile),
        outputDir
    )