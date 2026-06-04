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

from scipion.install.funcs import InstallHelper

from pwchem import Plugin as pwchemPlugin
from .constants import *
import os

_references = ['']


class Plugin(pwchemPlugin):
    @classmethod
    def defineBinaries(cls, env):
        cls.addFLOWRrootPackage(env)

    @classmethod
    def _defineVariables(cls):
        """ Return and write a variable in the config file.
        """
        cls._defineEmVar(FLOWR_DIC['home'], cls.getEnvName(FLOWR_DIC))

    @classmethod
    def addFLOWRrootPackage(cls, env, default=True):
        installer = InstallHelper(
            FLOWR_DIC['name'],
            packageHome=cls.getVar(FLOWR_DIC['home']),
            packageVersion=FLOWR_DIC['version']
        )

        installer.getCondaEnvCommand(
            FLOWR_DIC['name'],
            binaryVersion=FLOWR_DIC['version'],
            pythonVersion='3.11'
        ).addCommand(
            f"{cls.getEnvActivationCommand(FLOWR_DIC)} && "
            "git clone --branch v1.0 --depth 1 https://github.com/jule-c/flowr_root.git "
        ).addCommand(
            f"cd flowr_root && conda env create -f environment.yml -n {FLOWR_DIC['name']}-{FLOWR_DIC['version']}",
            f"{FLOWR_DIC['name']}_installed"
        ).addCommand(
            f"mkdir -p {cls.getVar(FLOWR_DIC['home'])}/checkpoints && "
            f"conda run -n {FLOWR_DIC['name']}-{FLOWR_DIC['version']} pip install gdown && "
            f"conda run -n {FLOWR_DIC['name']}-{FLOWR_DIC['version']} gdown "
            f"'https://drive.google.com/uc?id=1eaazPXBL3Kpk5unXmQWFWtOfJKP1N9uj' "
            f"-O {cls.getVar(FLOWR_DIC['home'])}/checkpoints/flowr_root_v2.2.ckpt"
        )

        installer.addPackage(
            env,
            dependencies=['conda', 'pip', 'git'],
            default=default
        )
