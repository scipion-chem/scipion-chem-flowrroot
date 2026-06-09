# **************************************************************************
# *
# * Authors:   Blanca Pueche (blanca.pueche@cnb.csis.es)
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

from pwem.objects import  AtomStruct, SetOfAtomStructs
from pwem.objects import Sequence, SetOfSequences
from pwchem.objects import SmallMolecule, SetOfSmallMolecules
from pwchem.protocols.Sequences.protocol_define_sequences import ProtDefineSetOfSequences
from pwchem.utils.utilsFasta import parseFasta



class ProtInpainting(EMProtocol):
    """
    """
    _label = 'Molecular inpainting'

    # -------------------------- DEFINE param functions ----------------------
    def _defineParams(self, form):
        """ Define the input parameters that will be used.
        Params:
            form: this is the form to be populated with sections and params.
        """
        form.addHidden('useGpu', params.BooleanParam, default=True,
                       label="Use GPU for execution",
                       help="This protocol has both CPU and GPU implementation. Choose one.")

        form.addHidden('gpuList', params.StringParam, default='0',
                       label="Choose GPU IDs",
                       help="Comma-separated GPU devices that can be used.")


        form.addSection(label='Input')
        form.addParam('inputAtomStruct', params.PointerParam,
                      pointerClass='AtomStruct',
                      label="Input structure: ",
                      help='Select the AtomStruct object')
        form.addParam('inputSetOfMols', params.PointerParam,
                      pointerClass='SetOfSmallMolecules',
                      label="Input reference ligands set: ",
                      help='Select the AtomStruct object')
        form.addParam('referenceMol', params.StringParam,
                        label='Reference ligand: ',
                        help='Reference ligand')

        form.addParam('affinity', params.BooleanParam, default=False,
                       label="Predict affinity: ",
                       help='Choose whether to predict affinity of the new molecules with input protein')

        group = form.addGroup('Parameters')
        group.addParam('atoms', params.StringParam, default='', #todo create wizard
                       label='Atoms to modify: ',
                       help="Specify the atoms to modify.")

        group.addParam('pocketCutoff', params.FloatParam, default=6.0,
                       label='Pocket cutoff: ',
                       help="Number of step size. Its related to the temperature at which the diffusion process samples the distribution.")
        group.addParam('cutPocket', params.BooleanParam, default=True,
                      label="Cut pocket: ",
                      help='Choose whether the program sees whole protein or only the pocket.')
        group.addParam('nMolecules', params.IntParam, default=1,
                       label='Number of generated molecules: ', help="Number of generated molecules.")

        group.addParam('seed', params.IntParam, default=42, label='Random seed:',expertLevel=params.LEVEL_ADVANCED,
                        help='Seed for reproducible generation.')
        group.addParam('maxPocketSize',params.IntParam,default=1000,label='Maximum pocket size:',expertLevel=params.LEVEL_ADVANCED,
                       help='Maximum number of atoms allowed in the pocket.')
        group.addParam('optimizeLigands', params.BooleanParam, default=True, expertLevel=params.LEVEL_ADVANCED,
                        label='Optimize ligands:')
        group.addParam('sampleIters', params.IntParam, default=20, expertLevel=params.LEVEL_ADVANCED,
                       label='Max. iterations: ', help="Maximum number of sample iterations.")
        group.addParam('noiseScale', params.FloatParam, default=0.0,
                       label='Noise: ', expertLevel=params.LEVEL_ADVANCED,
                       help="How much noise added to generation to increase diversity.")
        group.addParam('sampleMolSizes', params.BooleanParam, default=True,
                        label="Sample molecule sizes: ", expertLevel=params.LEVEL_ADVANCED,
                        help="Enables stochastic sampling of molecular sizes, allowing the model to generate ligands with varying number of atoms based on learned size distribution.")
        group.addParam('batchCost', params.IntParam, default=20,
                       label='Batch cost: ', expertLevel=params.LEVEL_ADVANCED,
                       help="How much noise added to generation to increase diversity.")
        group.addParam('filterCondSubstructure', params.BooleanParam,
                       default=False,
                       label="Strict substructure filtering: ",
                       help="If enabled, generated molecules that do not contain the specified substructure will be discarded. This may fail if invalid molecules are produced during generation.")

        form.addParallelSection(threads=4, mpi=1)

    # --------------------------- STEPS functions ------------------------------
    def _insertAllSteps(self):
        self._insertFunctionStep(self.convertFilesStep)
        self._insertFunctionStep(self.createLigandFile)
        self._insertFunctionStep(self.runFlowrStep)
        if self.affinity.get():
            self._insertFunctionStep(self.predictAffinityStep)

        self._insertFunctionStep(self.createOutputStep)

    def convertFilesStep(self):
        struct = self.inputAtomStruct.get()
        fileName = struct.getFileName()
        base = os.path.splitext(os.path.basename(fileName))[0]
        outFile = self._getExtraPath(base + '.pdb')
        if fileName.lower().endswith('.cif'):
            cifToPdb(fileName, outFile)

    def createLigandFile(self):
        molFile = self._getExtraPath('ligands.txt')
        outPath = self._getExtraPath('ligands.sdf')
        with open(molFile, 'w') as f:
            for mol in self.inputSetOfMols.get():
                f.write(os.path.abspath(mol.getFileName()) + '\n')

        args = ['-i', os.path.abspath(molFile), '-o', os.path.abspath(outPath), '-of', 'sdf']

        Plugin.runScript(self, 'rdkit_IO.py', args, env=RDKIT_DIC, cwd=self._getExtraPath())


    def runFlowrStep(self):
        scriptPath = os.path.join(Plugin.getVar(FLOWR_DIC['home']),'flowr_root/flowr/gen/generate_from_pdb.py')
        outPath = self._getExtraPath('inpainting')
        modelPath = os.path.join(Plugin.getVar(FLOWR_DIC['home']),'checkpoints/flowr_root_v2.1.ckpt')

        ligIdx = self.getLigandIndex()

        struct = self.inputAtomStruct.get()
        fileName = struct.getFileName()
        base = os.path.splitext(os.path.basename(fileName))[0]
        outFile = self._getExtraPath(base + '.pdb')
        if not os.path.exists(outFile):
            outFile = os.path.abspath(self.inputAtomStruct.get().getFileName())

        args = [
            '--pdb_file', outFile,
            '--ligand_file', (os.path.abspath(self._getExtraPath('ligands.sdf'))),
            '--ligand_id', ligIdx,
            '--arch', 'pocket', # NEEDS to be this value bc of the model
            '--pocket_type', 'holo', # NEEDS to be this value bc of the model
            '--pocket_cutoff', self.pocketCutoff.get(),
            '--sample_n_molecules_per_target', self.nMolecules.get(),
            '--max_sample_iter', self.sampleIters.get(),
            '--coord_noise_scale', self.noiseScale.get(),
            '--batch_cost', self.batchCost.get(),
            '--num_workers', (self.numberOfThreads.get()),
            '--ckpt_path', modelPath,
            '--save_dir', os.path.abspath(outPath),
            '--filter_valid_unique',
            '--max_pocket_size', self.maxPocketSize.get(),
            '--substructure_inpainting',
            '--substructure'

        ] + [idx for idx in re.split(r'[\s,]+', self.atoms.get().strip()) if idx]

        if self.filterCondSubstructure.get():
            args.append('--filter_cond_substructure')

        args.extend(['--seed', self.seed.get()])
        if self.optimizeLigands.get():
            args.append('--add_hs_and_optimize')

        if self.cutPocket.get(): args.append('--cut_pocket')
        if self.sampleMolSizes.get(): args.append('--sample_mol_sizes')

        if self.useGpu.get():
            args.append('--gpus')
            args.append('1')

        fullProgram = (
            f"export PYTHONPATH={os.path.join(Plugin.getVar(FLOWR_DIC['home']),'flowr_root')}:$PYTHONPATH && "
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

    def predictAffinityStep(self):
        scriptPath = os.path.join(Plugin.getVar(FLOWR_DIC['home']), 'flowr_root/flowr/predict/predict_from_pdb.py')
        modelPath = os.path.join(Plugin.getVar(FLOWR_DIC['home']), 'checkpoints/flowr_root_v2.1.ckpt')
        outPath = self._getExtraPath('inpainting_affinity')
        struct = self.inputAtomStruct.get()
        fileName = struct.getFileName()
        base = os.path.splitext(os.path.basename(fileName))[0]
        outFile = self._getExtraPath(base + '.pdb')
        if not os.path.exists(outFile):
            outFile = os.path.abspath(self.inputAtomStruct.get().getFileName())

        args = [
            '--pdb_file', outFile,
            '--ligand_file', (glob.glob(os.path.join(self._getExtraPath('inpainting'), '*optimized-hs.sdf'))[0]),
            '--multiple_ligands',
            '--add_hs_and_optimize_gen_ligs',
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
        if self.optimizeLigands.get():
            args.append('--add_hs_and_optimize_gen_ligs')

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

    def createOutputStep(self):
        outPath = self._getExtraPath('inpainting')
        sdfFiles = glob.glob(os.path.join(outPath, '*optimized-hs.sdf'))

        if not sdfFiles:
            self.warning("No SDF files found")
            return
        splitDir = self._getPath()
        for sdf in sdfFiles:
            args = [
                '-i', os.path.abspath(sdf),
                '-od', os.path.abspath(splitDir),
                '-of', 'sdf',
                '-ob', 'flowr_mol'
            ]

            Plugin.runScript(
                self,
                'rdkit_IO.py',
                args,
                env=RDKIT_DIC,
                cwd=self._getExtraPath()
            )

        sdfs = glob.glob(os.path.join(self._getPath(), '*.sdf'))
        outMols = SetOfSmallMolecules().create(outputPath=self._getPath())
        outMols.setProteinFile(self.inputAtomStruct.get().getFileName())
        for sdf in sdfs:
            molName = os.path.splitext(os.path.basename(sdf))[0]
            mol = SmallMolecule(smallMolFilename=sdf, molName=molName)
            outMols.append(mol)

        self._defineOutputs(outputSmallMolecules=outMols)

    # --------------------------- INFO functions -----------------------------------
    def _summary(self):
        summary = []
        return summary

    def _methods(self):
        methods = []
        return methods

    def _validate(self):
        validations = []
        return validations

    def _warnings(self):
        warnings = []
        return warnings

    # --------------------------- UTILS functions -----------------------------------
    def getLigandIndex(self):
        for i, mol in enumerate(self.inputSetOfMols.get()):
            if str(mol) == self.referenceMol.get():
                return i