import os

from pwem.viewers import Chimera
from pyworkflow.viewer import Viewer

from biofold.protocols import ProtImportPredictions, ProtBoltz, ProtChai, ProtIntelliFold, ProtProtenix


class ProtImportViewer(Viewer):
    """ Viewer for ProtImportPredictions protocol output. """
    _label = 'viewer discrepancies'
    _targets = [ProtImportPredictions, ProtBoltz, ProtChai, ProtIntelliFold, ProtProtenix]

    def visualize(self, obj, **args):
        # Create Chimera command file
        fnCmd = self.protocol._getExtraPath("chimera_output.cxc")
        with open(fnCmd, 'w') as f:
            # Process protocol outputs
            for output in self.protocol._outputs:
                # If the file is an atomic structure (.cif or .pdb), open it in Chimera
                fileName = os.path.abspath(eval(f'self.protocol.{output}.getFileName()'))
                if fileName.endswith(".cif") or fileName.endswith(".pdb"):
                    f.write(f"open {fileName}\n")
                    f.write("color bfactor palette alphafold\n")
                    f.write("hide atoms\n")
                    f.write("ribbon #1-10")

        # Run Chimera with the generated command file
        Chimera.runProgram(Chimera.getProgram(), fnCmd + "&")