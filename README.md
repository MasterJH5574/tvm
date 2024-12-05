# 15-745 Project - MagPy with TVM

This branch is the integration of MagPy with TVM, for eager-mode DNN
program compilation in TVM.

## Project Reports

* [Project proposal](reports/15745_Project_Proposal.pdf)
* [Project milestone report](reports/15745_Project_Milestone_Report.pdf)
* [Project final report](reports/15745_Project_Final_Report.pdf)

## Reproduce the results

Please follow the steps below to reproduce the evaluation results.

**Step 1.** Install TVM via

```bash
python -m pip install --pre -U -f https://mlc.ai/wheels mlc-ai-nightly-cu123
```

**Step 2.** Install MagPy by following [the MagPy installation](#installation).

**Step 3.** Run evaluation via

```bash
./eval_script/run_15.sh
```



---

(Below is the original README content of the MagPy project.)

# MagPy
MagPy is a JIT compiler for PyTorch programs. It can extract the operator graph from PyTorch programs and optimize the graph with a wide range of deep learning graph compilers.

# Installation
MagPy now supports Python 3.9. The support of other Python versions is working in progress.

1. Install CUDA. CUDA 11.8 is recommended.
2. Install dependencies:
    ```bash
    pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
    ```
3. Install MagPy:
    ```bash
    pip install -e .
    ```
4. Compile a shared library to disable Python integer cache by LD_PRELOAD. This script will generates a ``ldlong.v3.9.12.so'' file in build/ directory. You need to set the LD_PRELOAD environment variable to this file when running the PyTorch program.
    ```bash
    cd scripts
    ./compile_longobj.sh
    ```

# Example Usage

The following script compiles and runs a simple PyTorch program with MagPy.

```python
LD_PRELOAD=build/ldlong.v3.9.12.so python test/example.py
```

# Citation
If you find MagPy useful in your research, please consider citing the following paper:

> MagPy: Effective Operator Graph Instantiation for Deep Learning by Execution State Monitoring; Chen Zhang, Rongchao Dong, Haojie Wang, Runxin Zhong, Jike Chen, and Jidong Zhai, Tsinghua University; will be appeared in USENIX ATC'24.

