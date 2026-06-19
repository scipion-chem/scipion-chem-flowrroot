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
from pyworkflow.object import String, Float
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
    AI Generated:

    This protocol performs scaffold-based ligand design using the FLOWR generative
    framework. Starting from a protein structure and a reference ligand, it
    generates novel compounds that preserve specific structural or functional
    characteristics of the input molecule while exploring new chemical space.

    The protocol supports two complementary design strategies:

    - Scaffold Hopping:
        Preserves the key functional groups responsible for protein interactions
        while replacing the central molecular scaffold. This enables exploration
        of alternative chemotypes that may retain biological activity while
        improving properties such as novelty, selectivity, or patentability.

    - Scaffold Elaboration:
        Preserves the core scaffold of the reference ligand while generating
        new substituents, decorations, and side chains. This strategy is useful
        for lead optimization and structure-activity relationship (SAR) studies.

    Core Concepts
    -------------
    FLOWR Generative Model:
        Deep learning framework that generates novel molecules conditioned on
        the geometry of a protein binding pocket and an input reference ligand.

    Protein-Guided Design:
        Molecules are generated using structural information from the receptor,
        allowing designs to remain compatible with the target binding site.

    Scaffold Hopping:
        Generates chemically distinct scaffolds while attempting to preserve
        important interaction patterns from the reference ligand.

    Scaffold Elaboration:
        Retains the central scaffold and modifies peripheral regions to improve
        potency, selectivity, physicochemical properties, or novelty.

    Workflow
    --------
    1. Import protein structure and reference ligand.
    2. Convert structures into FLOWR-compatible formats.
    3. Extract and prepare the protein binding pocket.
    4. Configure the desired design strategy:
           - Scaffold hopping
           - Scaffold elaboration
    5. Execute FLOWR conditional molecular generation.
    6. Optionally optimize generated ligand geometries.
    7. Optionally predict protein-ligand affinity.
    8. Split generated molecule collections into individual compounds.
    9. Export molecules as a SetOfSmallMolecules object.

    Generation Parameters
    ---------------------
    Pocket Processing:
        - Pocket cutoff distance
        - Pocket extraction around reference ligand
        - Minimum and maximum pocket sizes

    Sampling Controls:
        - Number of molecules to generate
        - Random seed
        - Number of diffusion iterations
        - Noise level for diversity
        - Stochastic molecule size sampling

    Post-processing:
        - Ligand geometry optimization
        - Optional affinity prediction
        - Optional strict substructure filtering

    Design Modes
    ------------
    Scaffold Hopping:
        Preserves functional groups and interaction motifs while replacing
        the underlying scaffold architecture.

    Scaffold Elaboration:
        Preserves the scaffold and generates alternative substituents,
        decorations, and side chains around the retained core.

    Outputs
    -------
    outputSmallMolecules:
        A SetOfSmallMolecules containing all successfully generated compounds.

    Each SmallMolecule includes:
        - Generated molecular structure (SDF)
        - Molecule identifier
        - Associated receptor structure
        - Optional affinity predictions when enabled

    Use Cases
    ---------
    - Lead optimization campaigns
    - Scaffold hopping for novel chemotype discovery
    - Exploration of structure-activity relationships
    - Generation of patentable analogs
    - Protein-guided ligand design
    - Hit expansion and compound diversification
    - Discovery of alternative scaffolds with retained activity
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
        self._insertFunctionStep(self.genIndivMoleculesStep)

        self._insertFunctionStep(self.createOutputStep)

    def convertFilesStep(self):
        return utils._convertFiles(self)

    def createLigandFileStep(self):
        utils._createLigandFile(self)

    def runFlowrStep(self):
        scriptPath = os.path.join(Plugin.getVar(FLOWR_DIC['home']),'flowr_root/flowr/gen/generate_from_pdb.py')
        outPath = self._getExtraPath('scaffold')

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

    def genIndivMoleculesStep(self):
        utils._individualMols(self, 'scaffold')


    def createOutputStep(self):
        sdfs = glob.glob(os.path.join(self._getPath(), '*.sdf'))
        outMols = SetOfSmallMolecules().create(outputPath=self._getPath())
        outMols.setProteinFile(self.inputAtomStruct.get().getFileName())
        for sdf in sdfs:
            molName = os.path.splitext(os.path.basename(sdf))[0]
            mol = SmallMolecule(smallMolFilename=sdf, molName=molName)

            if self.affinity.get():
                props = utils._extractSdfProperties(sdf)
                mol.pIC50 = Float()
                mol.pKi = Float()
                mol.pKd = Float()
                mol.pEC50 = Float()
                mol.setAttributeValue('pIC50', props.get("pic50"))
                mol.setAttributeValue('pKi', props.get("pki"))
                mol.setAttributeValue('pKd', props.get("pkd"))
                mol.setAttributeValue('pEC50', props.get("pec50"))

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