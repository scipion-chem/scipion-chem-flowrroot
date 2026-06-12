# -*- coding: utf-8 -*-
# **************************************************************************
# *
# * Authors:  Blanca Pueche (blanca.pueche@cnb.csic.es)
# *
# * Biocomputing Unit, CNB-CSIC
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

from pwchem.wizards import SelectElementWizard
from .protocols import ProtDenovoGeneration, ProtScaffoldDesign, ProtGrowth, ProtInpainting
import os
import sys
import subprocess
from pwem.wizards import VariableWizard

from pwchem.utils import getBaseName
from pwchem.viewers import PyMolViewer
from pwchem.constants import RDKIT_DIC
from pwchem import Plugin

SelectElementWizard().addTarget(protocol=ProtDenovoGeneration,
                               targets=['referenceMol'],
                               inputs=['inputSetOfMols'],
                               outputs=['referenceMol'])

SelectElementWizard().addTarget(protocol=ProtScaffoldDesign,
                               targets=['referenceMol'],
                               inputs=['inputSetOfMols'],
                               outputs=['referenceMol'])

SelectElementWizard().addTarget(protocol=ProtGrowth,
                               targets=['referenceMol'],
                               inputs=['inputSetOfMols'],
                               outputs=['referenceMol'])

SelectElementWizard().addTarget(protocol=ProtInpainting,
                               targets=['referenceMol'],
                               inputs=['inputSetOfMols'],
                               outputs=['referenceMol'])

class ViewInputLigandAtomsWizard(VariableWizard):
    """
    3D viewer for reference ligand with atom indices labeled.
    Used to select atom IDs for ProtInpainting.atoms parameter.
    """
    _targets, _inputs, _outputs = [], {}, {}

    def writePmlFile(self, pmlFile, molFile, molName, proteinFile=None):
        molFile = os.path.abspath(molFile)
        pml = ""
        if proteinFile:
            pml += f"load {os.path.abspath(proteinFile)}\n"
            pml += "hide everything, all\n"
            pml += "show cartoon, all\n"

        pml += f"load {molFile}, {molName}\n"
        pml += f"remove hydrogens\n"
        pml += f"show sticks, {molName}\n"

        pml += f"label {molName}, index\n"

        pml += "set label_size, 14\n"
        pml += "set label_color, red\n"

        with open(pmlFile, "w") as f:
            f.write(pml)

    def show(self, form, *params):
        inputParam, _ = self.getInputOutput(form)
        protocol = form.protocol
        project = protocol.getProject()

        inSet = getattr(protocol, inputParam[0]).get()
        molName = getattr(protocol, inputParam[1]).get()
        mol = None
        for m in inSet:
            if str(m) == molName:
                mol = m
                break

        if mol is None:
            print("Ligand not found")
            return
        molFile = mol.getFileName()
        rdkitMolFile = molFile.replace(".sdf", "_rdkit.sdf")
        scriptPath = os.path.join(os.path.dirname(__file__), "scripts", "addRDKITindices.py")
        if not os.path.exists(rdkitMolFile):
            env_name = str(RDKIT_DIC['name'])+'-'+str(RDKIT_DIC['version'])
            subprocess.run(
                [
                    "conda", "run",
                    "-n", env_name,
                    "python",
                    scriptPath,
                    molFile,
                    rdkitMolFile
                ],
                check=True
            )

        molFile = rdkitMolFile
        molName = getBaseName(molFile)
        proteinFile = None
        if hasattr(inSet, "getProteinFile"):
            proteinFile = inSet.getProteinFile()

        pmlDir = project.getTmpPath()
        pmlFile = os.path.join(pmlDir, f"{molName}_atoms.pml")

        self.writePmlFile(
            pmlFile,
            molFile,
            molName,
            proteinFile=proteinFile
        )
        pymolV = PyMolViewer(project=project)
        view = pymolV._visualize(os.path.abspath(pmlFile),
                                 cwd=os.path.dirname(pmlFile))[0]
        view.show()


ViewInputLigandAtomsWizard().addTarget(protocol=ProtInpainting,
    targets=['atoms'],
    inputs=['inputSetOfMols', 'referenceMol'],
    outputs=[]
)