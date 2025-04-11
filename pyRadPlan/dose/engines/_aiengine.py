from ._base_pencilbeam import PencilBeamEngineAbstract
import numpy as np
# import torch
# import torch.nn as nn


class AIEngine(PencilBeamEngineAbstract):
    short_name = "AIEngine"
    name = "AI Engine"
    possible_radiation_modes = ["protons"]

    def _load_model(self, model_path):
        """
        Load the AI model from the given path.
        """
        # Assuming a PyTorch model for example
        model = torch.load(model_path)
        return model

    def _load_weights(self, weights_path):
        """
        Load the weights into the model from the given path.
        """
        self.model.load_state_dict(torch.load(weights_path))

    def _get_ray_geometry_from_beam(self, beam_info, ray_num):
        """
        Get the ray geometry from the beam information.
        """
        # Extract the cuboid
        # Placeholder for actual implementation
        return np.zeros((150, 23, 23))

    def _compute_bixel(self, bixel):
        """
        Compute the dose for a given bixel using the AI model.
        """
        # get the energy of the bixel
        # forward the cuboid stored in the ray to the model
        cuboid = self._get_ray_geometry_from_beam(bixel, bixel["ix"])
        cuboid_tensor = torch.tensor(cuboid, dtype=torch.float32).unsqueeze(
            0
        )  # Add batch dimension
        with torch.no_grad():  # Disable gradient calculation
            bixel["physical_dose"] = self.model(cuboid_tensor).numpy()[0]
        return bixel
