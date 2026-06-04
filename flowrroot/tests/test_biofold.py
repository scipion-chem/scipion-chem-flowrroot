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

from biofold.protocols import ProtChai, ProtBoltz, ProtIntelliFold, ProtProtenix
from pyworkflow.tests import BaseTest, setupTestProject, DataSet


defSetASChain, defSetPDBChain = 'A', 'B'
defSetPDBFile = 'Tmp/5ni1_{}_FIRST-LAST.fa'.format(defSetPDBChain)

names = ['5ni1']
defSetChains = [None, defSetASChain, defSetPDBChain]
defSetFiles = [defSetPDBFile]

defSetSeqs = '''1) {"name": "%s", "chain": "%s", "index": "FIRST-LAST", "seqFile": "%s"}''' % \
                         (names[0], defSetPDBChain, defSetPDBFile)


def gpu_available():
    try:
        # run nvidia-smi, check if it exists and returns 0
        subprocess.run(["nvidia-smi"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


@unittest.skipIf(not gpu_available(), "No GPU available, skipping Chai tests.")
class TestChai(BaseTest):
    @classmethod
    def setUpClass(cls):
        cls.ds = DataSet.getDataSet('model_building_tutorial')
        setupTestProject(cls)

    def _runChai(self):
        protChai = self.newProtocol(
            ProtChai,
            inputOrigin=2,
            file=self.ds.getFile('Sequences/3lqd_B_mutated.fasta')
        )

        self.launchProtocol(protChai)
        best = getattr(protChai, 'outputBestAtomStruct', None)
        self.assertIsNotNone(best)
        all = getattr(protChai, 'outputSetOfAtomStructs', None)
        self.assertIsNotNone(all)

    def test(self):
        self._runChai()

class TestBoltz(BaseTest):
    @classmethod
    def setUpClass(cls):
        cls.ds = DataSet.getDataSet('model_building_tutorial')

        setupTestProject(cls)

    def _runBoltz(self):
        protBoltz = self.newProtocol(
            ProtBoltz,
            useGpu=False,
            inputOrigin=2,
            entityType=1,
            recyclingSteps=1,
            samplingSteps=50,
            file=self.ds.getFile('Sequences/3lqd_B_mutated.fasta')
        )

        self.launchProtocol(protBoltz)
        best = getattr(protBoltz, 'outputBestAtomStruct', None)
        self.assertIsNotNone(best)
        all = getattr(protBoltz, 'outputSetOfAtomStructs', None)
        self.assertIsNotNone(all)

    def test(self):
        self._runBoltz()

class TestIntelliFold(BaseTest):
    @classmethod
    def setUpClass(cls):
        cls.ds = DataSet.getDataSet('model_building_tutorial')

        setupTestProject(cls)

    def _runIntelliFold(self):
        protIntelliFold = self.newProtocol(
            ProtIntelliFold,
            inputOrigin=2,
            entityType=1,
            recyclingSteps=1,
            samplingSteps=50,
            file=self.ds.getFile('Sequences/3lqd_B_mutated.fasta')
        )

        self.launchProtocol(protIntelliFold)
        best = getattr(protIntelliFold, 'outputBestAtomStruct', None)
        self.assertIsNotNone(best)
        all = getattr(protIntelliFold, 'outputSetOfAtomStructs', None)
        self.assertIsNotNone(all)

    def test(self):
        self._runIntelliFold()

class TestProtenix(BaseTest):
    @classmethod
    def setUpClass(cls):
        cls.ds = DataSet.getDataSet('model_building_tutorial')

        setupTestProject(cls)

    def _runProtenix(self):
        protProtenix = self.newProtocol(
            ProtProtenix,
            inputOrigin=2,
            entityType=1,
            recyclingSteps=1,
            samplingSteps=50,
            file=self.ds.getFile('Sequences/3lqd_B_mutated.fasta'),
            model=1
        )

        self.launchProtocol(protProtenix)
        best = getattr(protProtenix, 'outputBestAtomStruct', None)
        self.assertIsNotNone(best)
        all = getattr(protProtenix, 'outputSetOfAtomStructs', None)
        self.assertIsNotNone(all)

    def test(self):
        self._runProtenix()



