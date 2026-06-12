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

from pwchem.protocols import *
from pwem.protocols import ProtImportPdb
from flowrroot.protocols import ProtDenovoGeneration
from pyworkflow.tests import BaseTest, setupTestProject, DataSet

STRING_MOL = 'SmallMolecule (ZINC00001453 molecule)'
chainStr = '{"model": 0, "chain": "C", "residues": 141}'

class TestDenovo(BaseTest):
    @classmethod
    def setUpClass(cls):
        cls.ds = DataSet.getDataSet('model_building_tutorial')
        cls.dsLig = DataSet.getDataSet("smallMolecules")
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
        protImportSmallMols = cls.newProtocol(
            ProtChemImportSmallMolecules,
            filesPath=cls.dsLig.getFile('sdf'))
        cls.launchProtocol(protImportSmallMols)
        cls.protImportSmallMols = protImportSmallMols

    @classmethod
    def _runExtractLigand(cls, inputProt, chainStr):
        protExtLig = cls.newProtocol(
            ProtExtractLigands,
            cleanPDB=True, rchains=True, chain_name=chainStr)

        protExtLig.inputStructure.set(inputProt)
        protExtLig.inputStructure.setExtended('outputPdb')

        cls.proj.launchProtocol(protExtLig)
        cls.protExtLig = protExtLig

    def _runDenovo(cls):
        protDenovo = cls.newProtocol(
            ProtDenovoGeneration,
            inputAtomStruct=cls.protImportPDB.outputPdb,
            inputSetOfMols=cls.protImportSmallMols.outputSmallMolecules,
            referenceMol=STRING_MOL
        )
        cls.launchProtocol(protDenovo)
        cls.protDenovo = protDenovo


    def test(self):
        protExtract = self._runExtractLigand(self.protImportPDB, chainStr)
        self._waitOutput(protExtract, 'outputSmallMolecules')

        #denovo = self._runDenovo()
        #mols = getattr(denovo, 'outputSmallMolecules', None)
        #self.assertIsNotNone(mols)


