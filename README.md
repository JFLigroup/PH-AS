# PH-SA

## Introduction

PH-SA is a topology-based automatic active phase construction framework, enabling thorough configuration sampling and efficient computation.

## Requirements

The code in this repository has been tested with the following software and hardware requirements:

### Software Dependencies

- **Operating System:** Linux (Rocky Linux 8.8） ,or Windows 10，11
- **Python:** 3.8.13
- **ASE:**  3.22
- **Gudhi:** 3.8
- **Numpy:** 1.23
- **NetworkX:**  2.8.8
- **SciPy:** 1.10.1

### Hardware Requirements

- **Processor:** Multi-core CPU (e.g., Intel i5 or AMD equivalent)
- **Memory:** Minimum 8 GB RAM (16 GB recommended for large datasets)

## Installation Guide

1. **Install Python and Dependencies:**

   - Ensure Python  is installed. You can use Anaconda or pyenv for Python version management.

   - Create a virtual environment (recommended):

     ```
     python3 -m venv ph-sa-env
     source ph-sa-env/bin/activate  # For Linux/macOS
     ph-sa-env\Scripts\activate   # For Windows
     ```

   - Install dependencies:

     ```
     pip install ase==3.22 gudhi==3.8 numpy==1.23 networkx==2.8.8 scipy==1.10.1
     ```

2. **Clone the Repository:**

   ```
   git clone https://github.com/JFLigroup/PH-SA.git
   cd PH-SA
   ```

3. **Check Installation:** Test the installation by running a provided example (see the "Examples" section below).

### Typical Installation Time

On a standard desktop computer with an i5 processor and 16 GB RAM, the installation typically takes about 1-2 minutes.

## Usage Guide

### Files

#### Methods

- `adsorption_sites.py`: Searches for surface and embedding sites in periodic or  aperiodic structures.
- `utils.py`: Some utilities for site enumeration and for configuration generation.
- `structural_optimization.py`:Simple example of structural optimization using a model trained by dpa-1

#### Workflow

Two Jupyter notebooks provide a quick start to the workflow for finding unique configurations for both clusters and slabs:

- `cluster_workflow.ipynb`: Demonstrates workflow for aperiodic structures.
- `slab_workflow.ipynb`: Demonstrates workflow for periodic structures.

#### Example

A simple structure is provided for running the workflow and verifying installation.

 `input.json`: Simple sample input file for dpa-1 fine tuning.

 `OC_10M.pb`: Pre-trained weight for DPA-1 fine-tuning

### Running the Software

#### Quick Start with Example Data

1. Launch the Jupyter Notebook:

   ```
   jupyter notebook
   ```

2. Open the `cluster_workflow.ipynb` or `slab_workflow.ipynb` notebook.

3. Follow the step-by-step instructions in the notebook to run the workflow on example data.

#### Expected Output

- **Output:** Unique atomic configurations for clusters or slabs.
- **Typical Runtime:**
  - Small datasets: ~1-2 minutes on a standard desktop computer.
  - Large datasets: Runtime depends on the data size but typically completes within 10-30 minutes.

### Using Your Own Data

1. Prepare your input data in the appropriate format (e.g., XYZ or POSCAR files for atomic structures).

2. Modify the input paths in the Jupyter notebook to point to your data.

3. Run the workflow.

4. Analyze the output configurations generated in the results directory.

5. Use DFT for structure optimization or for molecular dynamics calculations

6. Transform the structure of the computed data into dpdata and train it

   `dp train input.json -- finetune OC_10M.pb`

7. Structural optimization using trained weights

### Typical Demonstration Runtime

- On a standard desktop computer, running the demonstration typically takes ~2-5 minutes.

