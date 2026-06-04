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

import os
import pyworkflow.protocol.params as params
from pwem.protocols import EMProtocol
from pyworkflow.object import String
import shutil

from pwchem import Plugin
from flowrroot.constants import FLOWR_DIC

from pwem.objects import  AtomStruct, SetOfAtomStructs
from pwem.objects import Sequence, SetOfSequences
from pwchem import SetOfSmallMolecules, SmallMolecule
from pwchem.protocols.Sequences.protocol_define_sequences import ProtDefineSetOfSequences
from pwchem.utils.utilsFasta import parseFasta



class ProtDenovoGeneration(EMProtocol):
    """
    De novo generation creates new ligands from scratch,
     using only the protein pocket structure as input. The model learns to generate molecules that are complementary to the binding site
    """
    _label = 'De Novo ligand generation'
    protSeq = ProtDefineSetOfSequences()

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
                      label="Input reference ligand: ",
                      help='Select the AtomStruct object')
        iGroup.addParam('referenceMol', params.StringParam, #todo select the index of this ref to use in call to script
                        label='Reference ligand: ',
                        help='Reference ligand')

        group = form.addGroup('Parameters')
        group.addParam('pocketCutoff', params.FloatParam, default=6.0,
                       label='Pocket cutoff: ',
                       help="Number of step size. Its related to the temperature at which the diffusion process samples the distribution.")
        group.addParam('cutPocket', params.BooleanParam, default=True,
                      label="Cut pocket: ",
                      help='Choose whether the program sees whole protein or only the pocket.')
        group.addParam('nMolecules', params.IntParam, default=1,
                       label='Number of generated molecules: ', help="Number of generated molecules.")
        group.addParam('sampleIters', params.IntParam, default=20,
                       label='Max. iterations: ', help="Maximum number of sample iterations.")
        group.addParam('noiseScale', params.FloatParam, default=0.0,
                       label='Noise: ', expertLevel=params.LEVEL_ADVANCED,
                       help="How much noise added to generation to increase diversity.")
        group.addParam('sampleMolSizes', params.BooleanParam, default=True,
                        label="Sample molecule sizes: ", expertLevel=params.LEVEL_ADVANCED,
                        help="Enables stochastic sampling of molecular sizes, allowing the model to generate ligands with varying number of atoms based on learned size distribution.")

        form.addParallelSection(threads=4, mpi=1)

    # --------------------------- STEPS functions ------------------------------
    def _insertAllSteps(self):
        self._insertFunctionStep(self.runFlowrStep)
        self._insertFunctionStep(self.createOutputStep)

    def runFlowrStep(self):
        scriptPath = self._getVar(FLOWR_DIC['home'])

        #Plugin.runCondaCommand(
        #    self,
        #    program="python",
        #    args=f"{scriptPath} {jsonPath} {yamlPath}",
        #    condaDic=BOLTZ_DIC
        #)

    def createOutputStep(self):
        predictionsPath = os.path.join(
            os.path.abspath(self._getPath()),
            "boltz_results_input",
            "predictions"
        )

        inputFolders = [
            f for f in os.listdir(predictionsPath)
            if os.path.isdir(os.path.join(predictionsPath, f))
        ]
        if not inputFolders:
            raise Exception(f"No prediction folders found in {predictionsPath}")

        inputFolder = os.path.join(predictionsPath, inputFolders[0])

        cifFiles = sorted([f for f in os.listdir(inputFolder) if f.lower().endswith('.cif')])
        if not cifFiles:
            raise Exception(f"No CIF files found in {inputFolder}")

        # Create output directory
        outPath = os.path.join(self._getExtraPath(), 'outputs')
        os.makedirs(outPath, exist_ok=True)

        # Create set of structures
        outputSet = SetOfAtomStructs.create(self._getPath())

        scores = {}

        for cifName in cifFiles:
            cifPath = os.path.join(inputFolder, cifName)

            modelBase = os.path.splitext(cifName)[0]
            jsonName = f"confidence_{modelBase}.json"
            jsonFile = os.path.join(inputFolder, jsonName)
            score = 0

            if os.path.exists(jsonFile):
                try:
                    with open(jsonFile) as f:
                        data = json.load(f)
                    score = data.get("confidence_score", 0)
                except:
                    pass

            scores[cifName] = score

            # Copy to output folder
            dst = os.path.join(outPath, cifName)
            shutil.copy(cifPath, dst)

            atomStruct = AtomStruct(filename=dst)
            atomStruct.origin = String()
            atomStruct.setAttributeValue('origin', 'Boltz')

            outputSet.append(atomStruct)

        # Select best model
        bestModel = max(scores, key=scores.get)
        bestStructPath = os.path.join(outPath, bestModel)

        bestStruct = AtomStruct(filename=bestStructPath)
        bestStruct.origin = String()
        bestStruct.setAttributeValue('origin', 'Boltz')

        # Optional: write summary file
        resultsFile = os.path.join(self._getPath(), 'results.txt')
        with open(resultsFile, 'w') as f:
            f.write("Model\tConfidenceScore\n")
            for name in sorted(scores.keys()):
                f.write(f"{name}\t{scores[name]:.3f}\n")
            f.write(f"BEST\t{bestModel}\t{scores[bestModel]:.3f}\n")

        self._defineOutputs(
            outputBestAtomStruct=bestStruct,
            outputSetOfAtomStructs=outputSet
        )

    # --------------------------- INFO functions -----------------------------------
    def _summary(self):
        resultsFile = os.path.join(self._getPath(), 'results.txt')

        if not os.path.exists(resultsFile):
            return ["Results file does not exist."]

        summary = ["Boltz predictions summary:"]

        try:
            with open(resultsFile) as f:
                for line in f:
                    summary.append(line.strip())
        except Exception as e:
            summary.append(f"Error reading results file: {e}")

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
    def guessEntityType(self, sequence):
        """ Guess if a sequence is DNA, RNA or Protein """
        sequence = sequence.upper().strip()

        protein_only = re.compile(r'[DEFHIKLMPQRVWY]')

        if protein_only.search(sequence):
            return 'protein'

        if 'U' in sequence:
            return 'rna'

        return 'dna'

