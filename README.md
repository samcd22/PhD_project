# 🚀 BayeSpace: Installation Guide

This guide walks you through setting up **BayeSpace** inside a Docker container with full VS Code integration. You'll be able to run Python scripts and Jupyter notebooks in a reproducible environment with minimal setup.

---

## ✅ Prerequisites

Make sure you have the following installed:

- [Docker Desktop](https://www.docker.com/products/docker-desktop/)
- [Visual Studio Code](https://code.visualstudio.com/)
- The **Remote - Containers** extension in VS Code

---

## 📦 Setup Steps

### 1. Choose your working directory

Decide where you want to work from — this is where your `data/`, `results/`, and notebooks will live.

---

### 2. Download the `run.ps1` script

Place the `run.ps1` PowerShell script **directly into your working directory**. This script:

- Builds the BayeSpace Docker image from GitHub
- Creates and starts a container named `bayespace-container`
- Mounts local `data/` and `results/` folders for persistent I/O

---

### 3. Open PowerShell in the working directory

Navigate to the folder where `run.ps1` is located. You can do this by right-clicking in the folder and selecting:

> **"Open in Terminal"** or **"Open in PowerShell"**

---

### 4. Run the setup script

In PowerShell, run the following command to allow the script to execute and start the container:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\run.ps1


