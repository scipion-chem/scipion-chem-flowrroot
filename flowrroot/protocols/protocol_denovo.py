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
from flowrroot import Plugin as flowrPlugin
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
    AI Generated:

De Novo Ligand Generation Protocol

This protocol performs structure-based de novo ligand generation using the
FLOWR generative model. Starting from a protein structure and a reference
ligand, it creates entirely new candidate molecules designed to fit the
binding pocket of the target protein.

Unlike virtual screening approaches that search existing compound libraries,
de novo generation constructs novel molecular structures directly from the
protein–ligand environment learned by the generative model.

Core Concepts
-------------
Protein-Guided Molecular Generation:
    The model uses the three-dimensional structure of a protein binding site
    to generate new ligands predicted to be compatible with the pocket.

Binding Pocket Representation:
    The region surrounding the reference ligand is extracted and provided
    to the generative model as structural context.

FLOWR Model:
    A diffusion-based molecular generation framework capable of creating
    chemically valid molecules directly from protein structural information.

Ligand Optimization:
    Generated molecules can optionally undergo post-processing and
    optimization steps to improve chemical validity and quality.

Affinity Prediction:
    Generated molecules may optionally be evaluated using the FLOWR affinity
    prediction model to estimate protein–ligand binding strength.

Workflow
--------
1. Input preparation:
    - A protein structure is provided.
    - A reference ligand identifies the target binding pocket.

2. Structure preprocessing:
    - Input structures are converted into formats required by FLOWR.
    - Pocket information is extracted around the reference ligand.

3. Ligand preparation:
    - The selected reference ligand is exported and used to define
      the generation region.

4. De novo generation:
    - FLOWR generates new candidate ligands inside the binding pocket.
    - Molecules are sampled according to the specified generation parameters.

5. Optional ligand optimization:
    - Generated structures can be refined and chemically optimized.

6. Optional affinity prediction:
    - Predicted binding affinity values are calculated for generated ligands.

7. Output creation:
    - Generated molecules are converted into individual SDF files.
    - Molecules are stored in a Scipion SetOfSmallMolecules object.

Input
-----
The receptor is never asked for: it is the structure the input object was
obtained on, taken from the object itself.

- inputType:
    Whether the binding pocket is defined by a reference ligand or by a
    structural ROI.

With a reference ligand:

- inputSetOfMols:
    SetOfSmallMolecules, docked, containing the reference ligands.

- referenceMol:
    Name of the ligand used to define the binding pocket. It also sets the
    size of the generated molecules.

With a structural ROI:

- inputStructROIs:
    SetOfStructROIs from a pocket search (fpocket, P2Rank, AutoSite...).

- referenceROI:
    Name of the ROI to generate molecules in.

- molSize:
    Number of heavy atoms of the generated molecules, which a ROI, unlike a
    reference ligand, does not imply. With sampleMolSizes enabled each
    molecule varies by 10% around it (25 gives molecules of 22-27 atoms).

Parameters
----------
Pocket Definition
~~~~~~~~~~~~~~~~~
- pocketCutoff:
    Distance cutoff (Å) used to define pocket residues around the
    reference ligand.

- cutPocket:
    If enabled, only the binding pocket is provided to the model.
    Otherwise, the complete protein structure is used.

Generation Control
~~~~~~~~~~~~~~~~~~
- nMolecules:
    Number of molecules to generate.

- seed:
    Random seed for reproducible generation.

- sampleIters:
    Maximum number of sampling iterations.

- noiseScale:
    Amount of stochastic noise introduced during generation.
    Higher values generally increase molecular diversity.

- sampleMolSizes:
    Varies the size of each generated molecule by 10% around the reference
    size (the reference ligand's number of heavy atoms, or molSize when
    generating in a ROI). Disabled, every molecule has exactly that size.

- batchCost:
    Internal generation batch size parameter used by FLOWR.

Pocket Constraints
~~~~~~~~~~~~~~~~~~
- minPocketSize:
    Minimum number of pocket atoms accepted.

- maxPocketSize:
    Maximum number of pocket atoms accepted.

Ligand Postprocessing
~~~~~~~~~~~~~~~~~~~~~
- optimizeLigands:
    Perform ligand optimization after generation.

- kekulize:
    Apply kekulization during molecular processing.

Affinity Evaluation
~~~~~~~~~~~~~~~~~~~
- affinity:
    Predict binding affinity for generated molecules.

GPU Support
~~~~~~~~~~~
- useGpu:
    Execute FLOWR using GPU acceleration.

- gpuList:
    GPU devices available for execution.

Output
------
- outputSmallMolecules:
    SetOfSmallMolecules containing all generated ligands.

Each generated molecule includes:
    - Molecular structure (SDF format)
    - Automatically assigned molecule identifier
    - Optional affinity predictions (if requested)

Use Cases
---------
- De novo drug design
- Lead discovery for novel protein targets
- Generation of pocket-specific ligand libraries
- Exploration of unexplored chemical space
- Creation of candidate molecules for docking and affinity evaluation
- Structure-based molecular design workflows
"""
    _label = 'De Novo ligand generation'

    @classmethod
    def mainParams(self, form, condition=None):
        '''Inputs shared by every generation protocol. The receptor is not asked for: it is the structure the
        input object was obtained on. condition hides the reference ligand where it is only one of the ways of
        defining the pocket (see ProtDenovoGeneration)'''
        cond = {'condition': condition} if condition else {}
        form.addParam('inputSetOfMols', params.PointerParam,
                      pointerClass='SetOfSmallMolecules',
                      label="Input reference ligands set: ", **cond,
                      help='Set of docked small molecules. The pocket is the region around the reference '
                           'ligand, and the receptor is the structure the set was docked on.')
        form.addParam('referenceMol', params.StringParam,
                      label='Reference ligand: ', **cond,
                      help='Ligand of the set defining the pocket to generate in.')
        form.addParam('model', params.EnumParam,
                      choices=['flowr_root_v2.1.ckpt', 'flowr_root_v2.2.ckpt'], default=1,
                      label="Model to use: ",
                      help='Select which model to use.')
        form.addParam('affinity', params.BooleanParam, default=True,
                      label="Predict affinity: ",
                      help='Choose whether to predict affinity of the new molecules with input protein')

    @classmethod
    def parameters(self, group):
        group.addParam('cutPocket', params.BooleanParam, default=True,
                       label="Cut pocket: ",
                       help='Choose whether the program sees whole protein or only the pocket.')
        group.addParam('pocketCutoff', params.FloatParam, default=6.0,
                       label='Pocket cutoff: ', condition='cutPocket',
                       help="Distance (A) around the reference ligand defining the pocket given to the model.")
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
                       help="Vary the size of each generated molecule by 10% around the reference size (the "
                            "reference ligand's number of heavy atoms, or 'Molecule size' when generating in a ROI). "
                            "Disable it to make every molecule exactly that size.")
        group.addParam('batchCost', params.IntParam, default=20,
                       label='Batch cost: ', expertLevel=params.LEVEL_ADVANCED,
                       help="How many molecules FLOWR samples at once. Lower it if the GPU runs out of memory.")

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
        form.addParam('inputType', params.EnumParam, display=params.EnumParam.DISPLAY_HLIST,
                      choices=['Reference ligand', 'Structural ROI'], default=0,
                      label="Define the pocket with: ",
                      help='Reference ligand: the pocket is the region around a docked molecule, and the size '
                           'of the generated molecules follows that of the reference.\n'
                           'Structural ROI: the pocket is a region of interest found by a pocket search '
                           '(fpocket, P2Rank, AutoSite...), and the size of the generated molecules is chosen '
                           'below.')
        form.addParam('inputStructROIs', params.PointerParam,
                      pointerClass='SetOfStructROIs', condition='inputType==1',
                      label="Input pockets: ",
                      help='Set of structural ROIs. The receptor is the structure they were found on.')
        form.addParam('referenceROI', params.StringParam, condition='inputType==1',
                      label='Pocket: ',
                      help='ROI of the set to generate molecules in.')
        form.addParam('molSize', params.IntParam, default=25, condition='inputType==1',
                      label='Molecule size: ',
                      help='Number of heavy atoms of the generated molecules. With a reference ligand this is '
                           'the size of that ligand, but a ROI defines no molecule, so it is asked for here. '
                           'With "Sample molecule sizes" enabled each molecule varies by 10% around it '
                           '(25 gives molecules of 22-27 atoms).')
        self.mainParams(form, condition='inputType==0')

        group = form.addGroup('Parameters')
        self.parameters(group)


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
        if utils.usesROI(self):
            utils._createROILigandFile(self)
        else:
            utils._createLigandFile(self)

    def runFlowrStep(self):
        outPath = self._getExtraPath('denovo')
        args = utils._createFlowrArgs(self, outPath)

        flowrPlugin.runFLOWRroot(
            self,
            args,
            cwd=self._getExtraPath()
        )

    def genIndivMoleculesStep(self):
        utils._individualMols(self, 'denovo')

    def createOutputStep(self):
        sdfs = glob.glob(os.path.join(self._getPath(), '*.sdf'))
        outMols = SetOfSmallMolecules().create(outputPath=self._getPath())
        outMols.setProteinFile(utils.getReceptorFile(self))
        for sdf in sdfs:
            molName = os.path.splitext(os.path.basename(sdf))[0]
            mol = SmallMolecule(smallMolFilename=sdf, molName=molName)

            if self.affinity.get():
                props = utils.getAffinities(self, sdf)
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
        if utils.usesROI(self):
            roi = utils.getSelectedROI(self)
            if roi is not None and self.molSize.get() > roi.getNumberOfPoints():
                validations.append(f'The pocket has only {roi.getNumberOfPoints()} points, which is fewer than '
                                   f'the {self.molSize.get()} atoms asked for: pick a smaller molecule size.')
        else:
            molSet = self.inputSetOfMols.get()
            if molSet is not None and not molSet.isDocked():
                validations.append(f'{molSet} is not docked yet: the reference ligand has to be placed in the '
                                   f'receptor, which is also where the receptor itself is taken from.')
        return validations

    def _warnings(self):
        warnings = []
        return warnings

    # --------------------------- UTILS functions -----------------------------------

