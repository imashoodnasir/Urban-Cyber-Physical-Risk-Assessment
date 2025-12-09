
"""
Placeholder utilities for multimodal data harmonization.

In a real pipeline, this module would contain code to read PlanetScope, Sentinel-1/2,
nighttime lights, population, traffic sensors, and OSM graphs, and to project them
onto a common spatio-temporal grid X(q,t).
"""

from typing import Dict, Any

def harmonize_raw_modalities(raw_inputs: Dict[str, Any]) -> Any:
    """
    Args:
        raw_inputs: dictionary with keys like "optical", "sar", "ntl", "population",
                    "traffic", "osm_graph" etc.

    Returns:
        A harmonized tensor or dictionary ready to be fed to RandomMultimodalDataset or
        a custom Dataset loading real data.
    """
    # This is intentionally left as a stub for the user to implement.
    raise NotImplementedError("Replace this stub with real harmonization code.")
