"""Utility functions for loading and processing structures."""

from typing import Dict
import numpy as np
import pydicom

from pyRadPlan.cst import StructureSet
from pyRadPlan.cst._cst import extract_structures_from_cst


def load_structures(structure_source: StructureSet) -> Dict:
    """
    Load structures from a pyRadPlan StructureSet object

    Args:
        structure_source: pyRadPlan StructureSet object

    Returns
    -------
        Dictionary of structures with masks and metadata
    """
    if not isinstance(structure_source, StructureSet):
        raise ValueError("Input must be a pyRadPlan StructureSet object")

    # Extract structures using the utility function
    structures = extract_structures_from_cst(structure_source)

    return structures


def load_structures_from_dicom(structure_file_path: str) -> Dict:
    """
    Load structures from an RTSTRUCT file

    Args:
        structure_file_path: Path to RTSTRUCT DICOM file

    Returns
    -------
        Dictionary of structures with masks and metadata
    """
    # Read RTSTRUCT file
    rtss = pydicom.dcmread(structure_file_path)

    # Extract structures
    structures = {}

    for structure in rtss.StructureSetROISequence:
        roi_number = structure.ROINumber
        roi_name = structure.ROIName

        # Find contour data for this ROI
        contour_data = None
        for roi_contour in rtss.ROIContourSequence:
            if roi_contour.ReferencedROINumber == roi_number:
                contour_data = roi_contour
                break

        if contour_data is None or not hasattr(contour_data, "ContourSequence"):
            print(f"Warning: No contour data found for ROI {roi_name}")
            continue

        # Extract contours
        contours = []
        for contour in contour_data.ContourSequence:
            points = np.array(contour.ContourData).reshape(-1, 3)
            contours.append(points)

        # Determine structure type (target vs OAR)
        structure_type = _determine_structure_type(roi_name, roi_number, rtss)

        # Create a placeholder for the mask
        # In a real implementation, you would convert contours to a 3D mask
        # This is a simplified representation
        mask = np.zeros((100, 100, 100))  # Placeholder
        volume_cc = 100.0  # Placeholder

        structures[roi_name] = {
            "mask": mask,
            "type": structure_type,
            "volume_cc": volume_cc,
            "contours": contours,
            "location": "Body",  # Placeholder
        }

    return structures


def _determine_structure_type(
    roi_name: str, roi_number: int, rtss: pydicom.dataset.FileDataset
) -> str:
    """
    Determine if a structure is a target or an organ at risk

    Args:
        roi_name: Name of the ROI
        roi_number: ROI number
        rtss: RTSTRUCT DICOM dataset

    Returns
    -------
        String indicating structure type ('TARGET', 'OAR', etc.)
    """
    # Look for ROI interpretations
    if hasattr(rtss, "RTROIInterpretedType"):
        return rtss.RTROIInterpretedType

    # Simple heuristic: If the name contains 'PTV', 'GTV', or 'CTV', it's likely a target
    lower_name = roi_name.lower()
    if any(target_type in lower_name for target_type in ["ptv", "gtv", "ctv", "target"]):
        return "TARGET"
    else:
        return "OAR"
