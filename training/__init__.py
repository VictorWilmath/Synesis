"""Everything that runs on the training box and nothing that runs on the VR rig.

Deliberately outside the installable ``bodytracker`` package. The tracker needs
numpy, OpenCV and an ONNX runtime; this needs torch, smplx and a few hundred
gigabytes of dataset. Keeping them apart means the machine wearing the headset
never has to install any of that.
"""
