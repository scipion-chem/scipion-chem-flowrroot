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

from pwem.objects import  AtomStruct, SetOfAtomStructs
from pwem.objects import Sequence, SetOfSequences
from pwchem.objects import SmallMolecule, SetOfSmallMolecules
from pwchem.protocols.Sequences.protocol_define_sequences import ProtDefineSetOfSequences
from pwchem.utils.utilsFasta import parseFasta
from .protocol_denovo import ProtDenovoGeneration
from .. import utils


class ProtGrowth(EMProtocol):
    """
    AI Generated:

Fragment and Core Growth Protocol

This protocol performs structure-guided molecular growth using the FLOWR
generative framework. Starting from a reference ligand bound to a protein,
the protocol generates novel molecules by preserving a selected molecular
substructure and expanding it within the target binding pocket.

Two growth strategies are available:

1. Core Growth:
    Preserves a selected ring system (core scaffold) from the reference
    ligand and generates new substituents around it.

2. Fragment Growth:
    Preserves an input fragment and extends it by adding new atoms and
    chemical groups, producing larger molecules while maintaining the
    original fragment.

The generated compounds remain conditioned on the protein binding site,
allowing exploration of chemically diverse ligands compatible with the
target pocket.

Core Concepts
-------------
Structure-Based Ligand Design:
    Molecule generation is guided by the three-dimensional geometry of the
    protein binding pocket.

Core Growing:
    A selected ring system from the reference ligand is retained while
    surrounding regions are redesigned. This approach is useful for
    scaffold optimization and lead expansion.

Fragment Growing:
    A molecular fragment is fixed and the model generates additional
    chemical structure around it. This is commonly used in fragment-based
    drug discovery workflows.

FLOWR Model:
    A diffusion-based generative framework that creates chemically valid
    molecules conditioned on protein structural information.

Shape-Aware Prior:
    An anisotropic Gaussian prior can be used to bias growth along
    directions that better match the geometry of the binding site.

Affinity Prediction:
    Generated molecules may optionally be evaluated using FLOWR affinity
    prediction models.

Workflow
--------
1. Input preparation:
    - A protein structure is provided.
    - A reference ligand identifies the target binding site.

2. Structure preprocessing:
    - Protein and ligand files are converted into FLOWR-compatible formats.
    - Pocket information is extracted around the ligand.

3. Growth setup:
    - Core Growth:
        A ring system is selected and preserved.
    - Fragment Growth:
        A molecular fragment is selected and expanded.

4. Molecular generation:
    - FLOWR generates new structures while maintaining the selected
      substructure.
    - Sampling parameters control diversity and molecule size.

5. Optional optimization:
    - Generated molecules can be chemically refined and optimized.

6. Optional affinity prediction:
    - Binding affinity estimates are calculated for generated ligands.

7. Output generation:
    - Generated molecules are exported as individual SDF files.
    - Results are stored as a SetOfSmallMolecules object.

Input
-----
- inputAtomStruct:
    Protein structure containing the target binding site.

- inputSetOfMols:
    SetOfSmallMolecules containing reference ligands.

- referenceMol:
    Name of the ligand used to define the binding pocket and growth region.

Growth Modes
------------
Core Growth
~~~~~~~~~~~
Preserves a selected ring system from the reference ligand and redesigns
the surrounding molecular environment.

Parameters:
    - ringIndex:
        Ring system to preserve. By default, the first (typically largest)
        ring system is selected.

Fragment Growth
~~~~~~~~~~~~~~~
Uses an existing molecular fragment as a starting point and generates
additional atoms around it.

Parameters:
    - growSize:
        Number of heavy atoms to add. If left empty, molecular sizes are
        sampled automatically by the model.

General Parameters
------------------
Pocket Definition
~~~~~~~~~~~~~~~~~
- pocketCutoff:
    Distance cutoff (Å) used to define pocket residues surrounding the
    reference ligand.

- cutPocket:
    If enabled, only the binding pocket is used during generation.
    Otherwise, the complete protein structure is provided.

Generation Control
~~~~~~~~~~~~~~~~~~
- nMolecules:
    Number of molecules to generate.

- seed:
    Random seed for reproducible results.

- sampleIters:
    Maximum number of sampling iterations.

- noiseScale:
    Amount of stochastic noise introduced during generation.
    Higher values generally increase structural diversity.

- sampleMolSizes:
    Enables stochastic sampling of ligand sizes.

- batchCost:
    Internal generation batch-size parameter used by FLOWR.

Pocket Constraints
~~~~~~~~~~~~~~~~~~
- minPocketSize:
    Minimum number of atoms required in the extracted pocket.

- maxPocketSize:
    Maximum number of atoms allowed in the extracted pocket.

Growth Parameters
~~~~~~~~~~~~~~~~~
- anisotropic:
    Enables a shape-aware anisotropic Gaussian prior that biases growth
    according to the geometry of the binding site.

- filterCondSubstructure:
    Applies strict filtering to ensure generated molecules retain the
    specified substructure. Molecules that fail the constraint are removed.

Ligand Postprocessing
~~~~~~~~~~~~~~~~~~~~~
- optimizeLigands:
    Perform chemical optimization after generation.

- kekulize:
    Apply kekulization during molecular processing.

Affinity Evaluation
~~~~~~~~~~~~~~~~~~~
- affinity:
    Predict protein–ligand binding affinity for generated molecules.

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
    - Preserved core or fragment from the input ligand
    - Optional affinity predictions (if requested)

Use Cases
---------
- Lead optimization through scaffold expansion
- Fragment-based drug discovery
- Core hopping and scaffold decoration
- Structure-guided ligand elaboration
- Exploration of chemical space around known binders
- Generation of novel analogues of active compounds
- Protein-specific molecular design workflows
"""
    _label = 'Fragment and core growth'

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
                      choices=['Core growing', 'Fragment growing'], default=0,
                      label="Growth option: ",
                      help='Core growing: Core growing preserves a selected ring system (core) from the reference ligand and generates new substituents around it. \n'
                           'Fragment growing: Fragment growing takes an input fragment and grows additional atoms around it. ')
        ProtDenovoGeneration.mainParams(form)

        group = form.addGroup('Parameters')
        group.addParam('ringIndex', params.IntParam, default=0, condition='option==0',
                       label='Ring system to preserve: ', help="Select which ring system to preserve (default: 0, i.e. the first/largest ring system).")
        group.addParam('growSize', params.StringParam, default='', condition='option==1',
                       label='Heavy atoms to add: ',
                       help="Specify the number of heavy atoms to add (if not set, molecule sizes are sampled).")

        group.addParam('anisotropic', params.BooleanParam, default=True,
                       label="Shape-aware anisotropic Gaussian prior: ",
                       help='Choose whether to use it for a shape-aware prior that better matches the growth direction.')

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
        outPath = self._getExtraPath('growth')
        args = utils._createFlowrArgs(self, outPath)

        if self.filterCondSubstructure.get():
            args.append('--filter_cond_substructure')

        if self.anisotropic.get():
            args.append('--anisotropic_prior')

        if self.option.get() == 0:
            args.append('--core_growing')
            args.append('--ring_system_indexing')
            args.append(self.ringIndex.get())
        elif self.option.get() == 1:
            args.append('--fragment_growing')
            if self.growSize.get() != '':
                args.append('--grow_size')
                args.append(self.growSize.get())

        flowrPlugin.runFLOWRroot(
            self,
            args,
            cwd=self._getExtraPath()
        )

    def genIndivMoleculesStep(self):
        utils._individualMols(self, 'growth')

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