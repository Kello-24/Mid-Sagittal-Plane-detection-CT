# Mid‑Sagittal Plane Detection Pipeline

This repository provides a complete pipeline for robust mid‑sagittal plane (MSP) detection in 3D CT volumes, with a focus on head‑and‑neck or mandibular scans. It includes:

* Loading and preprocessing of NIfTI & gzipped NIfTI images
* Threshold‑based bone segmentation
* Construction of a cached RegularGridInterpolator on physical coordinates
* Grid‑search initialization over spherical angles
* Huber‑loss optimization of plane parameters via BFGS
* Interactive visualization (axial & coronal) with widget sliders
* Animated GIF generation of plane evolution

---

## Installation

1. Clone this repository:

   ```bash
   git clone <repo_url>
   cd <repo>
   ```
2. Create a Python virtual environment and install dependencies:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

> **Dependencies** include: `numpy`, `nibabel`, `scipy`, `matplotlib`, `ipywidgets`, `scikit-image`, `joblib`, `imageio`.

---

## File Structure

* `msp_pipeline.py` – main pipeline and helper functions.
* `requirements.txt` – Python dependencies.

---

## Usage Example

```python
from msp_pipeline import MSP_pipeline

MSP_pipeline(
    base_path='data/patients',
    output_path='results',
    structure_names=['Image','Body','GTVp'],
    slice_axis=2,
    HU_range=[300,1500],
    azimuthal=(0,90),
    polar=(90,45),
    initialization_steps=10,
    delta=300,
    widget=True
)
```

This will process all patient subfolders under `data/patients`, save intermediate files in `results/<patient>/`, and display interactive viewers if `widget=True`.

---

## Function Summaries

### Loading & I/O

* **`load_nifti_file(file_path)`**

  * Loads a NIfTI file, reoriented to `(coronal, sagittal, axial)` axes.
  * Returns `(data_array, voxel_size_mm)`.

* **`open_gzip_file(gzip_file_path)`**

  * Reads raw bytes from a `.nii.gz` file.

* **`get_image_and_voxel_size_from_gzip(path)`**

  * Unpacks gzipped NIfTI, calls `load_nifti_file`, and returns data & voxel size.

* **`load_patient_structures(folder, structure_names)`**

  * Walks a patient folder, loads `.nii`/`.nii.gz` files matching `structure_names`, returns a dict `{'Name': (array, vox_sz)}`.

### Preprocessing & Masking

* **`mask_via_threshold(ct_image, HU_range=(300,1500))`**

  * Returns a binary mask of voxels within the specified HU window.

* **`preprocess_bone_image(struct_dict, HU_range)`**

  * Applies optional body‐mask, thresholds to bone range, returns `(proc_image, bone_ct)`.

* **`crop_patient_volumes(struct_dict, slice_axis=2, slice_range=None)`**

  * Crops each structure along an axis, auto‐using `GTVp` if available.

### Interpolation

* **`get_cached_interpolator(output_dir, image, voxel_size, filename, method)`**

  * Builds or loads a `RegularGridInterpolator` on physical grids `(y_mm,x_mm,z_mm)`, caches via Joblib.

### Plane Parameterization

* **`generate_normal(theta,phi)`**<br>
  Converts spherical angles to a 3D unit normal.

* **`compute_signed_distances(params, image, voxel_size)`**

  * Computes signed distance `d_i` from every nonzero voxel to the candidate plane.
  * Returns `(d, n, coords_phy, indices)`.

### Loss & Optimization

* **`huber_loss_function(diff, delta=300)`**

  * Implements piecewise Huber loss: quadratic inside `±δ`, linear outside.

* **`compute_objective(params, bone, interpolator, voxel_size, delta)`**

  * Samples mirror intensities at reflected physical coordinates.
  * Filters out-of-range mirrors (`<300` or `>2500 HU`).
  * Returns mean Huber loss over valid pairs.

* **`parameter_initialization(...)`**

  * Grid-search over `(θ, φ)` around specified centers, computes objective, plots heatmap, returns best `(θ,φ,L)`.

* **`optimize_plane(initial, image, interpolator, voxel_size, delta)`**

  * Refines plane via `scipy.optimize.minimize` (BFGS), records callback history.

* **`run_or_load_optimization(...)`**

  * Wrapper: loads saved params/objectives if exist, else runs full optimize, generates GIF, saves history.

### Visualization

* **`make_plane_gif(image_3d, voxel_size, plane_params, objective_vals, output_path, duration)`**

  * Creates an animation of plane contours on the mid‐axial slice, saves as GIF.

* **`display_scrollable_views(struct_dict, voxel_size, plane_coeffs_list, optimization_methods_list)`**

  * Interactive dual‐panel viewer: axial & coronal slices with structure contours and plane overlays.

---

## Mathematical Formulation

Let \$I(\mathbf{x})\$ be the raw CT intensity at location \$\mathbf{x}\$ (in physical mm).  Define the candidate plane by a unit normal \$\mathbf{n}\$ and offset \$L\$:  \$\mathbf{n}!\cdot!\mathbf{x}=L\$.  For each bone‐masked voxel \$\mathbf{x}\_i\$, let \$d\_i=\mathbf{n}!\cdot!\mathbf{x}\_i-L\$ and the mirror point \$\mathbf{x}\_i'=\mathbf{x}\_i-2d\_i,\mathbf{n}\$.

The residual is\~\$r\_i=I(\mathbf{x}\_i)-I(\mathbf{x}\_i')\$.  We only keep pairs where \$I(\mathbf{x}\_i')\in\[300,2500],\$HU.  The Huber loss with \$\delta=300\$ transforms each residual:

$$
L_\delta(r)=\begin{cases}
\tfrac12\,r^2, & |r|\le\delta,\\
\delta\bigl(|r|-\tfrac12\delta\bigr), & |r|>\delta.
\end{cases}
$$

The objective is the mean Huber loss over all valid pairs, optimised via BFGS.

---

## License

For Freeeeeeeeeee

---

## Contact

For questions or issues, please open an issue on GitHub or contact the maintainer.
