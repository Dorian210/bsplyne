import numpy as np
import meshio as io
from typing import Iterable
from functools import reduce
import os
import xml


def writePVD(fileName: str, groups: dict[str, dict]):
    """
    Writes a Paraview Data (PVD) file that references multiple VTK files.

    Creates an XML-based PVD file that collects and organizes multiple VTK files into groups,
    allowing visualization of multi-part, time-series data in Paraview.

    Parameters
    ----------
    fileName : str
        Base name for the mesh files (without numbers and extension)
    groups : dict[str, dict]
        Nested dictionary specifying file groups with format:
        {"group_name": {"ext": "file_extension", "npart": num_parts, "nstep": num_timesteps}}

    Notes
    -----
    VTK files must follow naming pattern: {fileName}_{group}_{part}_{timestep}.{ext}
    Example: for fileName="mesh", group="fluid", part=1, timestep=5, ext="vtu":
            mesh_fluid_1_5.vtu

    Returns
    -------
    None
    """
    rep, fname = os.path.split(fileName)
    pvd = xml.dom.minidom.Document()
    pvd_root = pvd.createElementNS("VTK", "VTKFile")
    pvd_root.setAttribute("type", "Collection")
    pvd_root.setAttribute("version", "0.1")
    pvd_root.setAttribute("byte_order", "LittleEndian")
    pvd.appendChild(pvd_root)
    collection = pvd.createElementNS("VTK", "Collection")
    pvd_root.appendChild(collection)
    for name, grp in groups.items():
        for jp in range(grp["npart"]):
            for js in range(grp["nstep"]):
                dataSet = pvd.createElementNS("VTK", "DataSet")
                dataSet.setAttribute("timestep", str(js))
                dataSet.setAttribute("group", name)
                dataSet.setAttribute("part", str(jp))
                dataSet.setAttribute("file", f"{fname}_{name}_{jp}_{js}.{grp['ext']}")
                dataSet.setAttribute("name", f"{name}_{jp}")
                collection.appendChild(dataSet)
    outFile = open(fileName + ".pvd", "w")
    pvd.writexml(outFile, newl="\n")
    print("VTK: " + fileName + ".pvd written")
    outFile.close()


def merge_meshes(meshes: Iterable[io.Mesh]) -> io.Mesh:
    """
    Merges multiple meshio.Mesh objects into a single mesh.

    Parameters
    ----------
    meshes : Iterable[io.Mesh]
        An iterable of meshio.Mesh objects to merge.

    Returns
    -------
    io.Mesh
        A single meshio.Mesh object containing the merged meshes with combined
        vertices, cells and point data.
    """
    vertices = np.vstack([m.points for m in meshes])
    all_cell_types = reduce(lambda a, b: a & b, [m.cells_dict.keys() for m in meshes])
    cells = {}
    for cell_type in all_cell_types:
        counter = 0
        cell = []
        for m in meshes:
            if cell_type in m.cells_dict:
                cell.append(m.cells_dict[cell_type] + counter)
            counter += m.points.shape[0]
        cells[cell_type] = np.vstack(cell)
    point_data_names = reduce(lambda a, b: a & b, [m.point_data.keys() for m in meshes])
    point_datas = {}
    for point_data_name in point_data_names:
        for m in meshes:
            if point_data_name in m.point_data:
                shape = m.point_data[point_data_name].shape[1:]  # type: ignore
                break
        point_data = []
        for m in meshes:
            if point_data_name in m.point_data:
                point_data.append(m.point_data[point_data_name])
            else:
                point_data.append(np.zeros((m.points.shape[0], *shape)))
        point_datas[point_data_name] = np.concatenate(point_data, axis=0)
    return io.Mesh(vertices, cells, point_data=point_datas)


def merge_saves(
    path: str, name: str, nb_patchs: int, nb_steps: int, group_names: list[str]
) -> None:
    """
    Merge multiple mesh files and save the merged results.

    This function reads multiple mesh files for each group and time step,
    merges them into a single mesh, and writes the merged mesh to a new file.
    It also generates a PVD file to describe the collection of merged meshes.

    Parameters
    ----------
    path : str
        The directory path where the mesh files are located.
    name : str
        The base name of the mesh files.
    nb_patchs : int
        The number of patches to merge for each group and time step.
    nb_steps : int
        The number of time steps for which meshes are available.
    group_names : list[str]
        A list of group names to process.

    Returns
    -------
    None
    """
    filename = os.path.join(path, name)
    filename_out = filename + "_merged"
    groups = {}
    for group_name in group_names:
        groups[group_name] = {"ext": "vtu", "npart": 1, "nstep": nb_steps}
        for step in range(nb_steps):
            merge_meshes(
                [
                    io.read(f"{filename}_{group_name}_{p}_{step}.vtu")
                    for p in range(nb_patchs)
                ]
            ).write(f"{filename_out}_{group_name}_0_{step}.vtu")
    writePVD(filename_out, groups)
