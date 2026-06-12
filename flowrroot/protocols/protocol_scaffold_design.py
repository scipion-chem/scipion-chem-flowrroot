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
from .protocol_denovo import ProtDenovoGeneration

from .. import utils



class ProtScaffoldDesign(EMProtocol):
    """
    """
    _label = 'Scaffold-based design'

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
        form.addParam('option', params.EnumParam,
                      choices=['Scaffold hopping', 'Scaffold elaboration'], default=0,
                      label="Design option: ",
                      help='Scaffold hopping: preserves the functional groups from a reference ligand while generating a new molecular scaffold. This is useful for exploring novel chemotypes while maintaining key interactions. \n'
                           'Scaffold elaboration: preserves the core molecular scaffold from a reference ligand while generating new R-groups, decorations, and functional groups. This is useful for lead optimization where you want to keep the scaffold but explore different substituents.')
        ProtDenovoGeneration.mainParams(form)

        group = form.addGroup('Parameters')
        ProtDenovoGeneration.parameters(group)

        group.addParam('filterCondSubstructure', params.BooleanParam,
                       default=False,
                       label="Strict substructure filtering: ",
                       help="If enabled, generated molecules that do not contain the specified substructure will be discarded. This may fail if invalid molecules are produced during generation.")

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
        outPath = self._getExtraPath('scaffold')
        modelPath = os.path.join(Plugin.getVar(FLOWR_DIC['home']),'checkpoints/flowr_root_v2.1.ckpt')

        ligIdx = self.getLigandIndex()

        struct = self.inputAtomStruct.get()
        fileName = struct.getFileName()
        base = os.path.splitext(os.path.basename(fileName))[0]
        outFile = self._getExtraPath(base + '.pdb')
        if not os.path.exists(outFile):
            outFile = os.path.abspath(self.inputAtomStruct.get().getFileName())

        args = utils._createArgs(self, outFile, outPath)

        if self.filterCondSubstructure.get():
            args.append('--filter_cond_substructure')

        if self.option.get() == 0:
            args.append('--scaffold_hopping')
        elif self.option.get() == 1:
            args.append('--scaffold_elaboration')

        if self.cutPocket.get(): args.append('--cut_pocket')
        if self.sampleMolSizes.get(): args.append('--sample_mol_sizes')


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
        outPath = self._getExtraPath('scaffold')
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