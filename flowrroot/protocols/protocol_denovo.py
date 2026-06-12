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
from pwchem.utils import pdbqt2other

from pwem.objects import  AtomStruct, SetOfAtomStructs
from pwem.objects import Sequence, SetOfSequences
from pwchem.objects import SmallMolecule, SetOfSmallMolecules
from pwchem.protocols.Sequences.protocol_define_sequences import ProtDefineSetOfSequences
from pwchem.utils.utilsFasta import parseFasta
from .. import utils

class ProtDenovoGeneration(EMProtocol):
    """
    De novo generation creates new ligands from scratch,
     using only the protein pocket structure as input. The model learns to generate molecules that are complementary to the binding site
    """
    _label = 'De Novo ligand generation'

    @classmethod
    def mainParams(self, form):
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

    @classmethod
    def parameters(self, group):
        group.addParam('pocketCutoff', params.FloatParam, default=6.0,
                       label='Pocket cutoff: ',
                       help="Number of step size. Its related to the temperature at which the diffusion process samples the distribution.")
        group.addParam('cutPocket', params.BooleanParam, default=True,
                       label="Cut pocket: ",
                       help='Choose whether the program sees whole protein or only the pocket.')
        group.addParam('nMolecules', params.IntParam, default=5,
                       label='Number of generated molecules: ', help="Number of generated molecules.")

        group.addParam('seed', params.IntParam, default=42, label='Random seed:', expertLevel=params.LEVEL_ADVANCED,
                       help='Seed for reproducible generation.')
        group.addParam('minPocketSize', params.IntParam, default=10, label='Minimum pocket size:',
                       expertLevel=params.LEVEL_ADVANCED,
                       help='Minimum number of atoms allowed in the pocket.')
        group.addParam('maxPocketSize', params.IntParam, default=1000, label='Maximum pocket size:',
                       expertLevel=params.LEVEL_ADVANCED,
                       help='Maximum number of atoms allowed in the pocket.')
        group.addParam('optimizeLigands', params.BooleanParam, default=True,
                       label='Optimize ligands:')
        group.addParam('kekulize', params.BooleanParam, default=False, expertLevel=params.LEVEL_ADVANCED,
                       label='kekulize ligands:')
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
        self.mainParams(form)

        group = form.addGroup('Parameters')
        self.parameters(group)


        form.addParallelSection(threads=4, mpi=1)

    # --------------------------- STEPS functions ------------------------------
    def _insertAllSteps(self):
        self._insertFunctionStep(self.convertFilesStep)
        self._insertFunctionStep(self.createLigandFileStep)
        self._insertFunctionStep(self.runFlowrStep)
        if self.affinity.get():
            self._insertFunctionStep(self.predictAffinityStep)

        self._insertFunctionStep(self.createOutputStep)

    def convertFilesStep(self):
        return utils._convertFiles(self)

    def createLigandFileStep(self):
        utils._createLigandFile(self)

    def runFlowrStep(self):
        scriptPath = os.path.join(Plugin.getVar(FLOWR_DIC['home']),'flowr_root/flowr/gen/generate_from_pdb.py')
        outPath = self._getExtraPath('denovo')
        struct = self.inputAtomStruct.get()
        fileName = struct.getFileName()
        base = os.path.splitext(os.path.basename(fileName))[0]
        outFile = self._getExtraPath(base + '.pdb')
        if not os.path.exists(outFile):
            outFile = os.path.abspath(self.inputAtomStruct.get().getFileName())

        args = utils._createArgs(self, outFile, outPath)

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
        utils._predictAffinity(self)

    def createOutputStep(self):
        outPath = self._getExtraPath('denovo')
        if self.optimizeLigands.get():
            sdfFiles = glob.glob(os.path.join(outPath, '*optimized*.sdf'))
            sdfFiles = [f for f in sdfFiles if os.path.getsize(f) > 0]
            if not sdfFiles:
                sdfFiles = glob.glob(os.path.join(outPath, '*.sdf'))
                sdfFiles = [f for f in sdfFiles if os.path.getsize(f) > 0]
        else:
            sdfFiles = glob.glob(os.path.join(outPath, '*.sdf'))
            sdfFiles = [f for f in sdfFiles if os.path.getsize(f) > 0]

        if not sdfFiles:
            self.warning("No valid (non-empty) SDF files found")
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

