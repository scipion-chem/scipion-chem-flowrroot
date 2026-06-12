# **************************************************************************
# *
# * Authors:   Blanca Pueche (blanca.pueche@cnb.csis.es)
# *
# * Unidad de  Bioinformatica of Centro Nacional de Biotecnologia , CSIC
# *
# * This program is free software; you can redistribute it and/or modify
# * it under the terms of the GNU General Public License as published by
# * the Free Software Foundation; either version 3 of the License, or
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
import subprocess
import unittest

from flowr.protocols import ProtDenovoGeneration
from pyworkflow.tests import BaseTest, setupTestProject, DataSet

STRING_MOL = 'SmallMolecule (ZINC00001453 molecule)'

class TestDenovo(BaseTest):
    @classmethod
    def setUpClass(cls):
        cls.ds = DataSet.getDataSet('model_building_tutorial')
        setupTestProject(cls)
        cls._runImportPDB()
        cls._runImportSmallMols()

    @classmethod
    def _runImportPDB(cls):
        protImportPDB = cls.newProtocol(
            ProtImportPdb,
            inputPdbData=1, pdbFile=cls.ds.getFile('PDBx_mmCIF/5ni1.pdb'))
        cls.launchProtocol(protImportPDB)
        cls.protImportPDB = protImportPDB

    @classmethod
    def _runImportSmallMols(cls):
        cls.protImportSmallMols = cls.newProtocol(
            ProtChemImportSmallMolecules,
            filesPath=cls.dsLig.getFile('mol2'))
        cls.launchProtocol(cls.protImportSmallMols)
        cls.protImportSmallMols = protImportSmallMols

    def _runDenovo(self):
        protDenovo = self.newProtocol(
            ProtDenovoGeneration,
            inputAtomStruct=cls.protImportPDB,
            inputSetOfMols=cls.protImportSmallMols,
            referenceMol=STRING_MOL
        )
        cls.launchProtocol(protDenovo)
        cls.protDenovo = protDenovo


    def test(self):
        denovo = self._runChai()
        mols = getattr(cls.denovo, 'outputSmallMolecules', None)
        self.assertIsNotNone(mols)


