# bsplyne

<p align="center">
  <img src=docs/logo.png width="500" />
</p>

**bsplyne** is a Python library for working with N-dimensional B-splines. It implements the Cox-de Boor algorithm for basis evaluation, order elevation, knot insertion, and provides a connectivity class for multi-patch structures. Additionally, it includes visualization tools with export capabilities to Paraview.

> **Note:** This library is not yet available on PyPI. To install, please clone the repository and install it manually.

## Installation

Since **bsplyne** is not yet on PyPI, you can install it locally as follows:

```bash
git clone https://github.com/Dorian210/bsplyne
cd bsplyne
pip install -e .
```

### Dependencies
Make sure you have the following dependencies installed:
- `numpy`
- `numba`
- `scipy`
- `matplotlib`
- `meshio`
- `tqdm`

## Main Modules

- **BSplineBasis**  
  Implements B-spline basis function evaluation using the Cox-de Boor recursion formula.
  
- **BSpline**  
  Provides methods for creating and manipulating N-dimensional B-splines, including order elevation and knot insertion.
  
- **MultiPatchBSplineConnectivity**  
  Manages the connectivity between multiple N-dimensional B-spline patches.
  
- **CouplesBSplineBorder** (less documented)  
  Handles coupling between B-spline borders.

## Examples

Several example scripts demonstrating the usage of **bsplyne** can be found in the `examples/` directory. These scripts cover:
- Basis evaluation on a curved line
- Plotting with Matplotlib
- Order elevation
- Knot insertion
- Surface examples
- Exporting to Paraview

## Documentation

The full API documentation is available in the `docs/` directory of the project or via the [online documentation portal](https://dorian210.github.io/bsplyne/).

## Contributions

At the moment, I am not actively reviewing contributions. However, if you encounter issues or have suggestions, feel free to open an issue.

## License

This project is licensed under the [CeCILL License](LICENSE.txt).

