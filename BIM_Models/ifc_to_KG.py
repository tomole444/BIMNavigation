# This script parses an IFC file using IfcOpenShell and extracts nodes and relationships
# based on the schema defined for BIM-to-Graph conversion for robotics.
# It calculates absolute coordinates, bounding boxes, and identifies spatial connections.

import ifcopenshell
import ifcopenshell.geom
import numpy as np
import json

# --- Helper Functions for Geometry and Placement ---

def get_absolute_placement(ifc_placement):
    """
    Recursively gets the absolute placement matrix for an IfcProduct.
    IFC placements are often relative to the placement of parent elements.
    """
    if ifc_placement.is_a("IfcLocalPlacement"):
        parent_placement = get_absolute_placement(ifc_placement.PlacementRelTo)
        local_placement = ifcopenshell.util.placement.get_local_placement(ifc_placement.RelativePlacement)
        return np.dot(parent_placement, local_placement)
    else:
        # Base case: The Site or Building placement is the world origin
        return np.identity(4)

def get_bounding_box_and_centroid(ifc_product, settings):
    """
    Calculates the Axis-Aligned Bounding Box (AABB) and centroid for an IfcProduct.
    """
    try:
        shape = ifcopenshell.geom.create_shape(settings, ifc_product)
        verts = shape.geometry.verts
        
        # Vertices are a flat list of coordinates [x1, y1, z1, x2, y2, z2, ...]
        points = np.array(verts).reshape(-1, 3)
        
        min_coords = np.min(points, axis=0)
        max_coords = np.max(points, axis=0)
        
        bounding_box = {
            "min_x": float(min_coords[0]), "min_y": float(min_coords[1]), "min_z": float(min_coords[2]),
            "max_x": float(max_coords[0]), "max_y": float(max_coords[1]), "max_z": float(max_coords[2])
        }
        
        centroid = np.mean(points, axis=0)
        position = { "x": float(centroid[0]), "y": float(centroid[1]), "z": float(centroid[2]) }

        return bounding_box, position
    except Exception as e:
        # Some elements might not have a geometric representation
        # print(f"Could not create shape for {ifc_product.GlobalId}: {e}")
        return None, None

# --- Main Extraction Logic ---

def extract_graph_data(ifc_file_path):
    """
    Main function to extract nodes and relationships from an IFC file.
    """
    ifc_file = ifcopenshell.open(ifc_file_path)
    
    nodes = []
    relationships = []
    
    # Settings for geometry creation
    settings = ifcopenshell.geom.settings()
    settings.set(settings.USE_WORLD_COORDS, True) # IMPORTANT: Ensures we get world coordinates

    # -- Part 1: Node Extraction --
    # Iterate through all physical and spatial elements
    products = ifc_file.by_type("IfcProduct")
    
    print(f"Found {len(products)} products. Extracting nodes...")
    for product in products:
        node = {
            "ifcId": product.GlobalId,
            "ifcType": product.is_a(),
            "name": getattr(product, "Name", "Unnamed") or "Unnamed",
            "label": product.is_a()  # Use IFC type as label for all
        }
        # Calculate bounding box and position for elements with geometry
        bounding_box, position = get_bounding_box_and_centroid(product, settings)
        if bounding_box:
            node["boundingBox"] = bounding_box
            node["position"] = position
        
        # --- Material Extraction ---

        materials = []
        if hasattr(product, "HasAssociations"):
            for assoc in product.HasAssociations or []:
                if assoc.is_a("IfcRelAssociatesMaterial"):
                    relating_material = getattr(assoc, "RelatingMaterial", None)
                    if relating_material:
                        if hasattr(relating_material, "Name"):
                            materials.append(relating_material.Name)
                        elif hasattr(relating_material, "Materials"):
                            for m in relating_material.Materials:
                                if hasattr(m, "Name"):
                                    materials.append(m.Name)
        if materials:
            node["materials"] = materials
        # Add special properties for doors/windows
        if product.is_a("IfcDoor") or product.is_a("IfcWindow"):
            node["isPassable"] = True
            node["width"] = getattr(product, "OverallWidth", None)
            node["height"] = getattr(product, "OverallHeight", None)
        nodes.append(node)

    print(f"Finished extracting {len(nodes)} nodes.")

    # -- Part 2: Relationship Extraction --
    print("Extracting relationships...")

    # 1. Spatial Containment (CONTAINS)
    for rel in ifc_file.by_type("IfcRelContainedInSpatialStructure"):
        container = rel.RelatingStructure
        for element in rel.RelatedElements:
            relationships.append({
                "source": container.GlobalId,
                "target": element.GlobalId,
                "label": "CONTAINS"
            })

    # 2. Aggregation / Decomposition (CONTAINS) - NEWLY ADDED
    # This captures relationships like a Site containing a Building.
    for rel in ifc_file.by_type("IfcRelAggregates"):
        parent = rel.RelatingObject
        for child in rel.RelatedObjects:
            relationships.append({
                "source": parent.GlobalId,
                "target": child.GlobalId,
                "label": "CONTAINS" # Using CONTAINS for consistency
            })

    # 3. Space Boundaries (BOUNDED_BY and CONNECTS_TO)
    # This is the most critical part for navigation
    space_boundaries = {} # Key: Element ID, Value: List of Space IDs it bounds
    
    for rel in ifc_file.by_type("IfcRelSpaceBoundary"):
        space = rel.RelatingSpace
        element = rel.RelatedBuildingElement
        
        if not element: continue

        # Create BOUNDED_BY relationship
        relationships.append({
            "source": space.GlobalId,
            "target": element.GlobalId,
            "label": "BOUNDED_BY"
        })
        
        # Store for connection analysis
        if element.GlobalId not in space_boundaries:
            space_boundaries[element.GlobalId] = []
        space_boundaries[element.GlobalId].append(space.GlobalId)

    # Now, find connections
    for element_id, space_ids in space_boundaries.items():
        if len(space_ids) > 1: # This element connects two or more spaces
            # For simplicity, connect the first two. More complex logic can handle multi-connections.
            unique_space_ids = list(set(space_ids))
            if len(unique_space_ids) > 1:
                relationships.append({
                    "source": unique_space_ids[0],
                    "target": unique_space_ids[1],
                    "label": "CONNECTS_TO",
                    "properties": {"through": element_id}
                })
    
    # 4. Openings (HAS_OPENING) - NEWLY ADDED
    # This connects doors and windows to the walls that contain them.
    for rel in ifc_file.by_type("IfcRelFillsElement"):
        opening = rel.RelatingOpeningElement
        filler = rel.RelatedBuildingElement # This is the door or window
        
        # We need to find the wall that has the opening
        if hasattr(opening, 'VoidsElements') and opening.VoidsElements:
             # VoidsElements is a tuple, we take the first element
            wall = opening.VoidsElements[0].RelatingBuildingElement
            relationships.append({
                "source": wall.GlobalId,
                "target": filler.GlobalId,
                "label": "HAS_OPENING"
            })

    # Add other relationship extractions (VERTICAL_CONNECTION) here...

    print(f"Finished extracting {len(relationships)} relationships.")
    
    return nodes, relationships


# --- Main Execution ---
import csv

# --- Main Execution ---
if __name__ == "__main__":
    # IMPORTANT: Replace with the actual path to your IFC file
    ifc_file_path = "/home/grass/BIM_Navigation/src/BIM_Models/models/Projekt1.ifc"
    
    try:
        nodes, relationships = extract_graph_data(ifc_file_path)
        
        graph_data = {
            "nodes": nodes,
            "relationships": relationships
        }
        
        # --- Save JSON ---
        output_json = "bim_riedel.json"
        with open(output_json, 'w') as f:
            json.dump(graph_data, f, indent=4)
        print(f"JSON saved to {output_json}")
        
        # --- Save CSV: Nodes ---
        output_nodes = "nodes_riedel.csv"
        with open(output_nodes, 'w', newline='') as f:
            writer = csv.DictWriter(
                f, 
                fieldnames=["id:ID", "label:LABEL", "name", "ifcType", "x", "y", "z"]
            )
            writer.writeheader()
            for node in nodes:
                pos = node.get("position", {})
                writer.writerow({
                    "id:ID": node["ifcId"],
                    "label:LABEL": node.get("label", "BuildingElement"),
                    "name": node.get("name", ""),
                    "ifcType": node.get("ifcType", ""),
                    "x": pos.get("x", ""),
                    "y": pos.get("y", ""),
                    "z": pos.get("z", "")
                })
        print(f"Nodes CSV saved to {output_nodes}")
        
        # --- Save CSV: Relationships ---
        output_edges = "edges_riedel.csv"
        with open(output_edges, 'w', newline='') as f:
            writer = csv.DictWriter(
                f, 
                fieldnames=[":START_ID", ":END_ID", ":TYPE"]
            )
            writer.writeheader()
            for rel in relationships:
                writer.writerow({
                    ":START_ID": rel["source"],
                    ":END_ID": rel["target"],
                    ":TYPE": rel["label"]
                })
        print(f"Edges CSV saved to {output_edges}")

        print(f"\nSuccessfully extracted graph data.")
        print(f"-> {len(nodes)} nodes")
        print(f"-> {len(relationships)} relationships")

    except FileNotFoundError:
        print(f"Error: The file was not found at '{ifc_file_path}'")
        print("Please update the 'ifc_file_path' variable in the script.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
