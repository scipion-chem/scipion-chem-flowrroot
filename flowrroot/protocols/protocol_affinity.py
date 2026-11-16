# **************************************************************************
# *
# * Authors:   Joaquin Algorta (joaquin.algorta@cnb.csis.es)
# *
# * Unidad de  Bioinformatica of Centro Nacional de Biotecnologia , CSIC
# *
# * This program is free software; you can redistribute it and/or modify
# * it under the terms of the GNU General Public License as published by
# * the Free Software Foundation; either version 2 of the License, or
# * (at your option) any later version.
# *
# * This program is distributed in the hope that it will be useful,
# * but WITHOUT ANY WARRANTY; without even the implied warranty of
# * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# * GNU General Public License for more details.
# *
# * You should have received a copy of the GNU General Public License
# * along with this program; if not, write to the Free Software
# * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
# * 02111-1307  USA
# *
# *  All comments concerning this program package may be sent to the
# *  e-mail address 'scipion@cnb.csic.es'
# *
# **************************************************************************
import os
import re
import shutil
import pyworkflow.protocol.params as params
from pwem.protocols import EMProtocol
from pwem.convert import cifToPdb
from pyworkflow.object import Float

from pwchem import Plugin
from pwchem.constants import OPENBABEL_DIC, RDKIT_DIC
from pwchem.objects import SetOfSmallMolecules
from pwchem.utils import pdbqt2other

from flowrroot import Plugin as flowrPlugin
from flowrroot.constants import FLOWR_DIC

PREDICT_SCRIPT = 'flowr/predict/predict_from_pdb.py'
# How the input poses are read, using pwchem's own converters
CONVERTERS = [('obabel_IO.py', OPENBABEL_DIC), ('rdkit_IO.py', RDKIT_DIC)]
# What pwchem's rdkit_IO.py can read (obabel_IO.py reads any of the formats poses come in)
RDKIT_FORMATS = {'.sdf', '.sd', '.mol2', '.pdb', '.mae', '.maegz', '.smi', '.smiles'}
# SDF property (lower case, as written by FLOWR.root) -> attribute name in the output molecules
AFF_PROPS = {'pic50': 'FLOWR_pIC50', 'pki': 'FLOWR_pKi', 'pkd': 'FLOWR_pKd', 'pec50': 'FLOWR_pEC50'}


class ProtFlowrAffinity(EMProtocol):
    """
    AI Generated:

Binding Affinity Prediction Protocol

This protocol predicts protein-ligand binding affinities for a set of posed
small molecules using the FLOWR.root affinity model. Nothing new is generated:
the input poses are kept as they are and only scored.

For every molecule, four affinity values are predicted at once (pIC50, pKi,
pKd and pEC50), expressed as negative logarithms of the corresponding
concentration, so higher values mean stronger predicted binding.

Core Concepts
-------------
Structure-Based Affinity Prediction:
    The model reads the three-dimensional arrangement of the ligand inside
    its binding site and predicts how strongly the complex binds. The pose
    itself is the input, so the quality of the prediction depends on the
    quality of the pose.

Binding Pocket Representation:
    The region of the receptor surrounding the ligand is extracted and given
    to the model as structural context, the same representation used by the
    generative protocols of this plugin.

Per-Molecule Receptor:
    Each molecule can carry its own receptor. Molecules coming from different
    docking runs, or from different targets, can therefore be scored in a
    single execution of the protocol.

Multi-Task Prediction:
    A single forward pass produces the four affinity types, which come from
    different prediction heads of the same model.

Workflow
--------
1. Receptor grouping:
    - The receptor of every molecule is resolved (see Input below).
    - Molecules sharing a receptor are collected into the same group.

2. Receptor preparation:
    - The receptor of each group is converted to PDB when needed
      (CIF and PDBQT inputs are converted, PDB is used as it is).

3. Ligand preparation:
    - The poses of a group are merged into a single SDF file.
    - Hydrogens are added, since docked poses often lack them.
    - Molecules that are not placed inside their receptor are discarded
      at this point (see Notes).

4. Affinity prediction:
    - FLOWR.root is run once per group over all its ligands.
    - The pocket is cut around each ligand and the affinities are predicted.

5. Output creation:
    - Predicted values are read back and assigned to a copy of the input set.

Input
-----
- inputSmallMolecules:
    SetOfSmallMolecules to score. The docked pose of each molecule is used
    when it exists, otherwise its original file. In both cases the molecule
    must already be placed in its binding site.

- inputStructure:
    Optional receptor, used only for the molecules that have no protein file
    of their own and whose set does not define one either.

The receptor of a molecule is resolved in this order:
    1. The protein file stored in the molecule itself.
    2. The protein file of the input set.
    3. The optional input structure.

Parameters
----------
Model Selection
~~~~~~~~~~~~~~~
- model:
    FLOWR.root checkpoint used for the prediction.

Input Conversion
~~~~~~~~~~~~~~~~
- convertEngine:
    Program used to read the input poses, OpenBabel or RDKit. OpenBabel
    reads every format poses come in, RDKit does not read PDBQT but handles
    the formats it does read better.

Pocket Definition
~~~~~~~~~~~~~~~~~
- cutPocket:
    If enabled, only the binding pocket is given to the model.
    Otherwise the complete receptor is used.

- pocketCutoff:
    Distance cutoff (A) used to define the pocket around each ligand.

Prediction Control
~~~~~~~~~~~~~~~~~~
- batchCost:
    How many ligand-pocket complexes are processed per forward pass, which
    is what determines the GPU memory the prediction needs.

- noiseScale:
    Noise added to the ligand coordinates before predicting. Small values
    can be used to probe how sensitive the prediction is to the exact pose.

- seed:
    Random seed, relevant when coordinate noise is applied.

- minPocketSize / maxPocketSize:
    Minimum and maximum number of atoms accepted in the pocket. A molecule
    with fewer receptor atoms around it than minPocketSize is taken not to
    be placed in the receptor, and is not scored.

GPU Support
~~~~~~~~~~~
- useGpu:
    Execute FLOWR.root using GPU acceleration.

- gpuList:
    GPU devices available for execution.

Output
------
- outputSmallMolecules:
    A copy of the input set in which every molecule carries the attributes:

    - FLOWR_pIC50
    - FLOWR_pKi
    - FLOWR_pKd
    - FLOWR_pEC50

    Higher values indicate stronger predicted binding. Molecules that could
    not be scored keep these attributes empty.

Use Cases
---------
- Rescoring of docking poses with a structure-aware deep learning model
- Ranking of virtual screening hits by predicted affinity
- Comparison of alternative poses of the same compound
- Affinity estimation for molecules generated by the other FLOWR protocols
- Scoring of several targets, or of several docking runs, in one step

Notes
-----
- A PDBQT pose stores no bond orders or formal charges, so the chemistry of
  the molecule is rebuilt from the SMILES header that Meeko writes in the
  PDBQT files, keeping the coordinates of the pose. Poses in a format that
  does define the chemistry (SDF, MOL2, CIF) are used directly. A PDBQT pose
  whose molecule has no such header anywhere cannot be scored, and is
  reported in the log.
- Batch cost controls how much GPU memory the prediction needs. Lower it if
  the run fails with an out-of-memory error.
- The pocket is cut around the pose, so molecules that are not placed in
  their receptor cannot be scored. These are detected before running the
  model, reported in the log, and returned with empty affinity values
  instead of aborting the prediction of the rest.
- Predicted affinities are model estimates and are not calibrated
  experimental values; they are most useful for ranking.
"""
    _label = 'Affinity prediction'
    _defaultName = 'flowrAffinity'

    # -------------------------- DEFINE param functions ----------------------
    def _defineParams(self, form):
        form.addHidden(params.USE_GPU, params.BooleanParam, default=True,
                       label="Use GPU for execution: ",
                       help="This protocol has both CPU and GPU implementation. Select the one you want to use.")
        form.addHidden(params.GPU_LIST, params.StringParam, default='0', label="Choose GPU IDs",
                       help="Add a list of GPU devices that can be used (comma-separated)")

        form.addSection(label='Input')
        form.addParam('inputSmallMolecules', params.PointerParam, pointerClass='SetOfSmallMolecules',
                      label='Input ligands: ',
                      help='Set of small molecules (posed in their binding site) to predict the affinity of. '
                           'The protein file of each molecule is used as receptor when available.')
        form.addParam('inputStructure', params.PointerParam, pointerClass='AtomStruct', allowsNull=True,
                      label='Receptor (if not in the molecules): ',
                      help='Receptor used for the molecules that do not have an associated protein file.')
        form.addParam('model', params.EnumParam, choices=['flowr_root_v2.1.ckpt', 'flowr_root_v2.2.ckpt'],
                      default=1, label="Model to use: ", help='Select which model to use.')
        form.addParam('convertEngine', params.EnumParam, choices=['OpenBabel', 'RDKit'], default=0,
                      label='Read the poses with: ',
                      help='Program used to read the input poses. OpenBabel reads every format used for poses, '
                           'RDKit does not read PDBQT but handles the formats it does read (SDF, MOL2, PDB, MAE) '
                           'better.')

        group = form.addGroup('Parameters')
        group.addParam('cutPocket', params.BooleanParam, default=True, label='Cut pocket: ',
                       help='Choose whether the program sees the whole protein or only the pocket.')
        group.addParam('pocketCutoff', params.FloatParam, default=6.0, label='Pocket cutoff: ',
                       condition='cutPocket',
                       help='Distance (A) around the ligand defining the pocket. It is also the distance used to '
                            'tell whether a molecule is placed in the receptor at all.')
        group.addParam('batchCost', params.IntParam, default=4, label='Batch cost: ',
                       help='How many ligand-pocket complexes FLOWR.root processes per forward pass. It is what '
                            'determines the GPU memory used: lower it if the prediction runs out of memory, '
                            'raise it on a large GPU to go faster.')
        group.addParam('noiseScale', params.FloatParam, default=0.0, expertLevel=params.LEVEL_ADVANCED,
                       label='Noise: ',
                       help='Noise added to the ligand coordinates before predicting. A small value can be used '
                            'to probe how sensitive the prediction is to the exact pose.')
        group.addParam('seed', params.IntParam, default=42, label='Random seed:',
                       expertLevel=params.LEVEL_ADVANCED,
                       help='Seed for reproducible prediction, relevant when coordinate noise is applied.')
        group.addParam('minPocketSize', params.IntParam, default=10, label='Minimum pocket size:',
                       expertLevel=params.LEVEL_ADVANCED,
                       help='Minimum number of atoms allowed in the pocket. A molecule with fewer receptor atoms '
                            'around it is not placed in the receptor and is not scored.')
        group.addParam('maxPocketSize', params.IntParam, default=1000, label='Maximum pocket size:',
                       expertLevel=params.LEVEL_ADVANCED,
                       help='Maximum number of atoms allowed in the pocket.')

        form.addParallelSection(threads=4, mpi=1)

    # --------------------------- STEPS functions ------------------------------
    def _insertAllSteps(self):
        oStep = self._insertFunctionStep(self.organizeInputStep, needsGPU=False)
        pSteps = []
        for gId in range(len(self.getGroups())):
            pSteps.append(self._insertFunctionStep(self.predictStep, gId, prerequisites=[oStep]))
        self._insertFunctionStep(self.createOutputStep, prerequisites=pSteps, needsGPU=False)

    def organizeInputStep(self):
        """Group the molecules by receptor and, per group, prepare the receptor and the poses FLOWR.root reads"""
        for gId, (protFile, molIdxs) in enumerate(self.getGroups().items()):
            os.makedirs(self.getGroupDir(gId), exist_ok=True)
            self.preparePdb(protFile, gId)
            self.linkPoses(gId, molIdxs)
            self.convertPoses(gId)
            self.fixPoses(gId, molIdxs)
            self.writeLigandsFile(gId, molIdxs)

    def predictStep(self, gId):
        gDir = os.path.abspath(self.getGroupDir(gId))
        if not self.getKept(gId):
            self.warning(f'No molecule of group {gId} is placed in its receptor, nothing to predict')
            return
        flowrPlugin.runFLOWRroot(self, self.getPredictArgs(gDir), cwd=gDir, script=PREDICT_SCRIPT)

    def createOutputStep(self):
        inMols = self.inputSmallMolecules.get()
        scores = {}  # index in the input set -> {attribute: value}
        for gId in range(len(self.getGroups())):
            scores.update(self.readGroupScores(gId))

        outMols = SetOfSmallMolecules.createCopy(inMols, self._getPath(), copyInfo=True)
        for i, mol in enumerate(inMols):
            newMol = mol.clone()
            for attr in AFF_PROPS.values():
                setattr(newMol, attr, Float())
                newMol.setAttributeValue(attr, scores.get(i, {}).get(attr))
            outMols.append(newMol)
        self._defineOutputs(outputSmallMolecules=outMols)

    # --------------------------- INFO functions -----------------------------------
    def _summary(self):
        return ['Affinity values written as FLOWR_pIC50, FLOWR_pKi, FLOWR_pKd and FLOWR_pEC50.']

    def _validate(self):
        validations = []
        molSet = self.inputSmallMolecules.get()
        if molSet is not None and self.convertEngine.get() == 1:
            badExts = {os.path.splitext(self.getPoseOf(mol))[1].lower() for mol in molSet} - RDKIT_FORMATS
            if badExts:
                validations.append(f'RDKit does not read {", ".join(sorted(badExts))} poses: '
                                   f'read them with OpenBabel instead.')

        if molSet is not None and not molSet.getProteinFile() and not self.inputStructure.get():
            if any(not m.getProteinFile() for m in molSet):
                validations.append('Some molecules have no associated protein file: '
                                   'define a receptor in "Receptor (if not in the molecules)".')
        return validations

    # --------------------------- UTILS functions -----------------------------------
    def getMolsList(self):
        if not hasattr(self, '_molsList'):
            self._molsList = [m.clone() for m in self.inputSmallMolecules.get()]
        return self._molsList

    @staticmethod
    def getPoseOf(mol):
        return mol.getPoseFile() or mol.getFileName()

    def getProteinOf(self, mol):
        """Receptor priority: molecule's own protein file, the set's one, the optional input structure"""
        protFile = mol.getProteinFile() or self.inputSmallMolecules.get().getProteinFile()
        if not protFile and self.inputStructure.get():
            protFile = self.inputStructure.get().getFileName()
        return os.path.abspath(protFile)

    def getGroups(self):
        """{receptor file: [indices (in the input set) of the molecules using it]}"""
        groups = {}
        for i, mol in enumerate(self.getMolsList()):
            groups.setdefault(self.getProteinOf(mol), []).append(i)
        return groups

    def linkPoses(self, gId, molIdxs):
        """Every pose of the group staged as <its index in the input set>.<ext>, so the index survives the
        conversion (pwchem's converters name each output after its input file)"""
        inDir, mols = self.getGroupFile(gId, 'inPoses'), self.getMolsList()
        os.makedirs(inDir, exist_ok=True)
        for i in molIdxs:
            poseFile = os.path.abspath(self.getPoseOf(mols[i]))
            destFile = os.path.join(inDir, f'{i}{os.path.splitext(poseFile)[1]}')
            if not os.path.exists(destFile):
                try:
                    os.link(poseFile, destFile)
                except OSError:
                    shutil.copy(poseFile, destFile)

    def convertPoses(self, gId):
        """Poses to SDF with pwchem's converter, a single call for the whole group"""
        outDir = self.getGroupFile(gId, 'sdfPoses')
        os.makedirs(outDir, exist_ok=True)
        scriptName, envDic = CONVERTERS[self.convertEngine.get()]
        args = f'--multiFiles -iD "{self.getGroupFile(gId, "inPoses")}" --pattern "*" ' \
               f'-of sdf --outputDir "{outDir}" --overWrite'
        Plugin.runScript(self, scriptName, args, env=envDic, cwd=self.getGroupDir(gId))

    def fixPoses(self, gId, molIdxs):
        """Rebuild the chemistry the pose files do not define (see scripts/fixPoseChemistry.py)"""
        outDir, mols, smiDic = self.getGroupFile(gId, 'fixedPoses'), self.getMolsList(), self.getTemplateSmiles()
        os.makedirs(outDir, exist_ok=True)

        templatesFile = self.getGroupFile(gId, 'templates.tsv')
        with open(templatesFile, 'w') as f:
            for i in molIdxs:
                if mols[i].getMolName() in smiDic:
                    f.write(f'{i}\t{smiDic[mols[i].getMolName()]}\n')

        args = f'-iD "{self.getGroupFile(gId, "sdfPoses")}" -od "{outDir}" -t "{templatesFile}"'
        Plugin.runScript(self, 'fixPoseChemistry.py', args, env=RDKIT_DIC, cwd=self.getGroupDir(gId),
                         scriptDir=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'scripts'))

    def writeLigandsFile(self, gId, molIdxs):
        """The single SDF FLOWR.root scores, leaving out the molecules that are not placed in the receptor
        (it cannot build their pocket and dies on the first one it meets)"""
        protCoords = self.readProteinCoords(self.getGroupFile(gId, 'protein.pdb'))
        cutoff, minSize = self.pocketCutoff.get(), self.minPocketSize.get()

        kept = []
        with open(self.getGroupFile(gId, 'ligands.sdf'), 'w') as fOut:
            for i in molIdxs:
                molFile = os.path.join(self.getGroupFile(gId, 'fixedPoses'), f'{i}.sdf')
                if not os.path.exists(molFile):
                    continue

                with open(molFile) as f:
                    molBlock = f.read()
                if not self.inPocket(molBlock, protCoords, cutoff, minSize):
                    self.info(f'Molecule {i}: fewer than {minSize} receptor atoms within {cutoff} A, it is not '
                              f'placed in the receptor and will not be scored')
                    continue

                fOut.write(molBlock)
                kept.append(str(i))

        with open(self.getGroupFile(gId, 'kept.txt'), 'w') as f:
            f.write('\n'.join(kept))

    @staticmethod
    def readProteinCoords(pdbFile):
        coords = []
        with open(pdbFile) as f:
            for line in f:
                if line.startswith(('ATOM', 'HETATM')):
                    coords.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
        return coords

    @staticmethod
    def inPocket(molBlock, protCoords, cutoff, minAtoms):
        """Whether at least minAtoms receptor atoms lie within cutoff of the molecule (SDF V2000 atom block)"""
        lines = molBlock.splitlines()
        nAtoms = int(lines[3][:3])
        molCoords = [(float(li[:10]), float(li[10:20]), float(li[20:30])) for li in lines[4:4 + nAtoms]]

        cut2, n = cutoff ** 2, 0
        for px, py, pz in protCoords:
            for lx, ly, lz in molCoords:
                if (px - lx) ** 2 + (py - ly) ** 2 + (pz - lz) ** 2 <= cut2:
                    n += 1
                    break
            if n >= minAtoms:
                return True
        return False

    def getTemplateSmiles(self):
        """{molName: SMILES} taken from the REMARK SMILES header Meeko writes in the PDBQT files.

        A PDBQT keeps no bond orders, so the pose alone is not enough to rebuild the molecule. The header is
        looked up in the files of every molecule of the same name (its conformers and poses), since Meeko only
        writes it in the first conformer of each molecule."""
        templates = {}
        for mol in self.getMolsList():
            molName = mol.getMolName()
            if molName in templates:
                continue
            for molFile in [self.getPoseOf(mol), mol.getFileName()]:
                smiles = self.readRemarkSmiles(molFile)
                if smiles:
                    templates[molName] = smiles
                    break
        return templates

    @staticmethod
    def readRemarkSmiles(molFile):
        """The SMILES of a Meeko PDBQT header, None for any other file"""
        if not molFile or not molFile.lower().endswith('.pdbqt') or not os.path.exists(molFile):
            return None
        with open(molFile) as f:
            for line in f:
                if line.startswith('REMARK SMILES') and not line.startswith('REMARK SMILES IDX'):
                    return line.split()[2]
                if line.startswith(('ATOM', 'HETATM')):
                    break
        return None

    def getGroupFile(self, gId, fileName):
        return os.path.abspath(os.path.join(self.getGroupDir(gId), fileName))

    def getKept(self, gId):
        """Indices (in the input set) of the molecules of the group that were actually staged for FLOWR.root"""
        with open(os.path.join(self.getGroupDir(gId), 'kept.txt')) as f:
            return [int(i) for i in f.read().split()]

    def getGroupDir(self, gId):
        return self._getExtraPath(f'group_{gId}')

    def preparePdb(self, protFile, gId):
        outFile = os.path.abspath(os.path.join(self.getGroupDir(gId), 'protein.pdb'))
        ext = os.path.splitext(protFile)[1].lower()
        if ext in ['.cif', '.mmcif']:
            cifToPdb(protFile, outFile)
        elif ext == '.pdbqt':
            pdbqt2other(self, protFile, outFile)
        else:
            os.symlink(protFile, outFile) if not os.path.exists(outFile) else None
        return outFile

    def getPredictArgs(self, gDir):
        model = ['flowr_root_v2.1.ckpt', 'flowr_root_v2.2.ckpt'][self.model.get()]
        args = [
            '--pdb_file', os.path.join(gDir, 'protein.pdb'),
            '--ligand_file', os.path.join(gDir, 'ligands.sdf'),
            '--multiple_ligands',
            '--arch', 'pocket', '--pocket_type', 'holo', '--pocket_noise', 'fix',
            '--pocket_cutoff', self.pocketCutoff.get(),
            '--coord_noise_scale', self.noiseScale.get(),
            '--batch_cost', self.batchCost.get(),
            '--seed', self.seed.get(),
            '--min_pocket_size', self.minPocketSize.get(),
            '--max_pocket_size', self.maxPocketSize.get(),
            '--num_workers', self.numberOfThreads.get(),
            '--ckpt_path', os.path.join(Plugin.getVar(FLOWR_DIC['home']), 'checkpoints', model),
            '--save_dir', gDir,
        ]
        if self.cutPocket.get():
            args.append('--cut_pocket')
        if getattr(self, params.USE_GPU).get():
            args += ['--gpus', 1]
        return args

    def readGroupScores(self, gId):
        """{index in the input set: {attribute: value}} read from the FLOWR.root output of the group.
        Output molecules come in the order of ligands.sdf, so they are matched through kept.txt."""
        gDir = self.getGroupDir(gId)
        kept = self.getKept(gId)
        if not kept:
            return {}

        outSdf = os.path.join(gDir, 'gen_lig_with_aff.sdf')
        with open(outSdf) as f:
            blocks = [b for b in f.read().split('$$$$') if b.strip()]
        if len(blocks) != len(kept):
            raise ValueError(f'FLOWR.root scored {len(blocks)} of the {len(kept)} molecules of group {gId}: '
                             f'cannot safely map the scores (see {outSdf})')

        scores = {}
        for i, block in zip(kept, blocks):
            props = self.parseSdfProps(block)
            scores[i] = {attr: props.get(key) for key, attr in AFF_PROPS.items()}
        return scores

    @staticmethod
    def parseSdfProps(block):
        props, key = {}, None
        for line in block.splitlines():
            line = line.strip()
            if line.startswith('>'):
                # Property header, e.g. ">  <pic50>  (1)"
                match = re.search(r'<(.+?)>', line)
                key = match.group(1).strip().lower() if match else None
            elif key and line:
                try:
                    props[key] = float(line)
                except ValueError:
                    pass
                key = None
        return props
