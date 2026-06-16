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
import pwem.protocols as emprot
from flowrroot.protocols import ProtDenovoGeneration, ProtGrowth, ProtInpainting, ProtScaffoldDesign
from pyworkflow.tests import BaseTest, setupTestProject, DataSet

STRING_MOL = 'SmallMolecule (g1_9ktp_OGA-1_1 molecule)'
chainStr = '{"model": 0, "chain": "A", "residues": 328}'

class TestDenovo(BaseTest):
    @classmethod
    def setUpClass(cls):
        cls.dsLig = DataSet.getDataSet("smallMolecules")
        setupTestProject(cls)
        cls._runImportPDB()
        cls._runImportSmallMols()
        cls._runExtractLigand()
        cls._runJoinSets()

    @classmethod
    def _runImportPDB(cls):
        protImportPDB = cls.newProtocol(
            ProtImportPdb,
            pdbId='9ktp')
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
    def _runExtractLigand(cls):
        protExtLig = cls.newProtocol(
            ProtExtractLigands,
            cleanPDB=True, rchains=True, chain_name=chainStr)

        protExtLig.inputStructure.set(cls.protImportPDB)
        protExtLig.inputStructure.setExtended('outputPdb')
        cls.proj.launchProtocol(protExtLig, wait=True)
        cls.protExtLig = protExtLig

    @classmethod
    def _runJoinSets(cls):
        unionProt = cls.newProtocol(emprot.ProtUnionSet)

        unionProt.inputSets.append(cls.protExtLig.outputSmallMolecules)
        unionProt.inputSets.append(cls.protImportSmallMols.outputSmallMolecules)
        cls.launchProtocol(unionProt, wait=True)
        cls.unionProt = unionProt


    def _runDenovo(cls):
        protDenovo = cls.newProtocol(
            ProtDenovoGeneration,
            inputAtomStruct=cls.protImportPDB.outputPdb,
            inputSetOfMols=cls.unionProt.outputSet,
            referenceMol=STRING_MOL
        )
        cls.launchProtocol(protDenovo)
        return protDenovo


    def test(self):
        denovo = self._runDenovo()
        self._waitOutput(denovo, 'outputSmallMolecules', sleepTime=10)
        mols = getattr(denovo, 'outputSmallMolecules', None)
        self.assertIsNotNone(mols)

class TestGrowth(TestDenovo):
    @classmethod
    def setUpClass(cls):
        cls.dsLig = DataSet.getDataSet("smallMolecules")
        setupTestProject(cls)
        cls._runImportPDB()
        cls._runImportSmallMols()
        cls._runExtractLigand()
        cls._runJoinSets()

    def _runGrowthCore(cls):
        protGrowthCore = cls.newProtocol(
            ProtGrowth,
            inputAtomStruct=cls.protImportPDB.outputPdb,
            inputSetOfMols=cls.unionProt.outputSet,
            referenceMol=STRING_MOL,
            option=0
        )
        cls.launchProtocol(protGrowthCore)
        return protGrowthCore

    def _runGrowthFrag(cls):
        protGrowthFrag = cls.newProtocol(
            ProtGrowth,
            inputAtomStruct=cls.protImportPDB.outputPdb,
            inputSetOfMols=cls.unionProt.outputSet,
            referenceMol=STRING_MOL,
            option=1
        )
        cls.launchProtocol(protGrowthFrag)
        return protGrowthFrag

    def test(self):
        coreGrowth = self._runGrowthCore()
        self._waitOutput(coreGrowth, 'outputSmallMolecules', sleepTime=10)
        mols = getattr(coreGrowth, 'outputSmallMolecules', None)
        self.assertIsNotNone(mols)

        fragGrowth = self._runGrowthFrag()
        self._waitOutput(fragGrowth, 'outputSmallMolecules', sleepTime=10)
        mols = getattr(fragGrowth, 'outputSmallMolecules', None)
        self.assertIsNotNone(mols)

class TestInpainting(TestDenovo):
    @classmethod
    def setUpClass(cls):
        cls.dsLig = DataSet.getDataSet("smallMolecules")
        setupTestProject(cls)
        cls._runImportPDB()
        cls._runImportSmallMols()
        cls._runExtractLigand()
        cls._runJoinSets()

    def _runInpaint(cls):
        protInpaint = cls.newProtocol(
            ProtInpainting,
            inputAtomStruct=cls.protImportPDB.outputPdb,
            inputSetOfMols=cls.unionProt.outputSet,
            referenceMol=STRING_MOL,
            atoms='1,3,5'
        )
        cls.launchProtocol(protInpaint)
        return protInpaint

    def test(self):
        coreInpainting= self._runInpaint()
        self._waitOutput(coreInpainting, 'outputSmallMolecules', sleepTime=10)
        mols = getattr(coreInpainting, 'outputSmallMolecules', None)
        self.assertIsNotNone(mols)

class TestScaffold(TestDenovo):
    @classmethod
    def setUpClass(cls):
        cls.dsLig = DataSet.getDataSet("smallMolecules")
        setupTestProject(cls)
        cls._runImportPDB()
        cls._runImportSmallMols()
        cls._runExtractLigand()
        cls._runJoinSets()

    def _runScaffoldHopping(cls):
        protScaffold = cls.newProtocol(
            ProtScaffoldDesign,
            inputAtomStruct=cls.protImportPDB.outputPdb,
            inputSetOfMols=cls.unionProt.outputSet,
            referenceMol=STRING_MOL,
            option=0
        )
        cls.launchProtocol(protScaffold)
        return protScaffold

    def _runScaffoldElab(cls):
        protScaffold = cls.newProtocol(
            ProtScaffoldDesign,
            inputAtomStruct=cls.protImportPDB.outputPdb,
            inputSetOfMols=cls.unionProt.outputSet,
            referenceMol=STRING_MOL,
            option=1
        )
        cls.launchProtocol(protScaffold)
        return protScaffold

    def test(self):
        protScaffold= self._runScaffoldHopping()
        self._waitOutput(protScaffold, 'outputSmallMolecules', sleepTime=10)
        mols = getattr(protScaffold, 'outputSmallMolecules', None)
        self.assertIsNotNone(mols)

        protScaffoldElab = self._runScaffoldElab()
        self._waitOutput(protScaffoldElab, 'outputSmallMolecules', sleepTime=10)
        mols = getattr(protScaffoldElab, 'outputSmallMolecules', None)
        self.assertIsNotNone(mols)

