# Differentiable Fluid Dynamics on TPUs

![Status](https://img.shields.io/badge/Status-Personal_Experiment-orange?style=for-the-badge)

## Overview

This repository demonstrates a modern workflow for [Computational Fluid Dynamics (CFD)](https://en.wikipedia.org/wiki/Computational_fluid_dynamics) that bridges the gap between rapid Python prototyping and supercomputer-scale production.

Targeted at scientists and engineers, this project implements a 2D Lattice [Boltzmann Method (LBM)](https://en.wikipedia.org/wiki/Lattice_Boltzmann_methods) solver to showcase how Google's [JAX](https://docs.jax.dev/) library and [Tensor Processing Units(TPUs)](https://en.wikipedia.org/wiki/Tensor_Processing_Unit) can solve two of the biggest pain points in scientific computing:

1. The "Two-Language" Problem: No more prototyping in Python and rewriting in C++/CUDA for speed.

1. The Optimization Barrier: Automatically differentiating through the physics to solve inverse design problems without writing complex adjoint solvers by hand.

---

## Key Features

**1. Scaling Potential:** 

A side-by-side comparison of a standard NumPy implementation running on CPU versus a JAX implementation running on TPU.
  - Code Familiarity: The JAX code is 95% identical to standard NumPy syntax.
  - Performance: Experimentation with different grid resolution and iteration on a real-time fluid simulation.
  - Unified Workflow: The exact same Python code scales from a laptop CPU to a full TPU Pod slice.

**2. Inverse Design & Automatic Differentiation**

Beyond raw speed, this repository demonstrates the power of differentiable physics.
  - Instead of trial-and-error to find optimal shapes, we use jax.grad to differentiate through the entire fluid simulation backwards in time.
  - Demo: We calculate the "Sensitivity Map" (Adjoint Gradient) for a cylinder in cross-flow, automatically visualizing where to add or remove material to minimize turbulence.

> **Read the full story:** https://medium.com/@o.bernie/beyond-ai-supercharging-computational-fluid-dynamics-with-tpus-and-jax-dabef1dff928

---
## Disclaimer: Hardware & Cost

**Hardware Requirements:** To replicate the high-performance results ($4096 \times 4096$ real-time simulation), access to a Google Cloud TPU VM or similar accelerator is recommended. The code will run on a standard CPU or GPU, but performance will scale down accordingly.

**Cost Warning:** If running this on Google Cloud Platform (GCP), please be aware of the hourly rates for TPU VMs and always remember to stop or delete your TPU instance when not in use to avoid unexpected charges.

## 🛠️ Workflow

### 1. Prerequisite: Google Cloud SDK
Ensure you have the `gcloud` CLI installed and authenticated.
```bash
gcloud auth login
gcloud config set project [YOUR_PROJECT_ID]
```

### 2. Create the TPU VM

We use the TPU v5e (Lite Pod) in the example below.

```bash
gcloud compute tpus tpu-vm create lb-jax-vm \
    --zone=us-west1-c \
    --accelerator-type=v5litepod-4 \
    --version=tpu-ubuntu2204-base \
    --spot
```

### 3. Configure Network (Allow Jupyter) 

By default, GCP blocks external access. You need to open port 8888 specifically for your IP address only.

Get your IP at https://ifconfig.co/ip

Create a firewall rule allowing Jupyter access from YOUR IP

```bash
gcloud compute firewall-rules create allow-jupyter-lab \
    --direction=INGRESS \
    --priority=1000 \
    --network=default \
    --action=ALLOW \
    --rules=tcp:8888 \
    --source-ranges=$MY_IP/32
```

### 4. SSH into the TPU VM

```bash
gcloud compute tpus tpu-vm ssh lb-jax-vm --zone=us-west1-c
```


### 5. Install Jupyter (if not in base image)

```bash
pip install jupyterlab
```

### 6. Launch Jupyter 

```bash
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --NotebookApp.token='secure-password'
```

### 7. Access via browser

Open your browser and navigate to: `http://[EXTERNAL_IP_OF_TPU_VM]:8888`

### 8. Run Experiments

- Upload `lattice-boltzmann-jax.ipynb` and run through all the cells. 
- <span style="color: red;">**Note: On the first run you may have to restart the kernel after running the first cell.**</span>
- Don't forget to experiment with lattice fineness and number of iterations to truly appreciate the capabilities of JAX for this simulation.

**Disclaimer: This project is a personal experiment applying JAX to physics simulations. It is not an officially supported Google Cloud product.**


