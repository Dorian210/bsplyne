# bsplyne

<p align="center">
  <img src="docs/logo.png" width="500" />
</p>

**bsplyne** is a Python library for working with N-dimensional B-splines, with a focus on numerical mechanics and geometry.
It implements the Cox–de Boor algorithm for basis evaluation, order elevation, knot insertion, and provides tools for handling
multi-patch B-spline structures. Visualization and export utilities (e.g. Paraview) are also included.

---

## Installation

### Using pip

Install the core library:

```bash
pip install bsplyne
```

Install the library **with recommended visualization features** (additionally install `pyvista`):

```bash
pip install bsplyne[viz]
```

> Note: choose either the core (`bsplyne`) or the visualization (`bsplyne[viz]`) installation, not both.

### Using conda (conda-forge)

Install the core library:

```bash
conda install -c conda-forge bsplyne
```

Optional: install PyVista to enable visualization features:

```bash
conda install -c conda-forge pyvista
```

### From source (development mode)

Clone the repository and install:

```bash
git clone https://github.com/Dorian210/bsplyne
cd bsplyne
pip install -e .       # core
pip install -e .[viz]  # with visualization
```

---

## Dependencies

Core dependencies are handled automatically by `pip` or `conda`:

- numpy
- numba
- scipy
- matplotlib
- meshio
- tqdm

Optional visualization:

- pyvista (for 3D visualization)

---

## Main Modules

- **BSplineBasis**  
  Evaluation of B-spline basis functions using the Cox–de Boor recursion formula.

- **BSpline**  
  Construction and manipulation of N-dimensional B-splines, including order elevation and knot insertion.

- **MultiPatchBSplineConnectivity**  
  Management of connectivity between multiple B-spline patches.

- **CouplesBSplineBorder**  
  Utilities for coupling B-spline borders (experimental / less documented).

---

## Tutorials

A step-by-step introduction to **bsplyne** is provided in:

```
examples/tutorial/
```

These scripts are designed as a progressive entry point to the library and cover:

1. B-spline basis functions  
2. Curve construction  
3. Surface generation  
4. Least-squares fitting  
5. Multi-patch geometries  
6. Export to Paraview  

In addition, a **comprehensive PDF guide** (`tp_bsplyne.pdf`) is included in the tutorials directory, providing a hands-on introduction to the library for new users.  
It explains the workflow, the main modules, and practical usage examples.

---

## Examples

Additional standalone examples are available in the `examples/` directory, including:

- Curves and surfaces
- Order elevation and knot insertion
- Visualization with Matplotlib
- Export to Paraview

---

## Documentation

The full API documentation is available online:

https://dorian210.github.io/bsplyne/

The documentation is generated from the source code docstrings and reflects the latest published version.

---

## Contributions

This project is primarily developed for research purposes.
While I am not actively reviewing external contributions, bug reports and suggestions are welcome via the issue tracker.

---

## License

This project is licensed under the **CeCILL License**.  
See [LICENSE.txt](LICENSE.txt) for details.
