# GitHub replacement commands

Use these commands after unzipping this package.

## Option A: clone fresh and replace the original notebook

```bash
git clone https://github.com/terry0809000/sleepingedf.git
cd sleepingedf

# Copy the BSPC-strengthened notebook over the original tracked notebook path
cp ../github_ready_sleepingedf_bspc_final/notebooks/sleep_edf_pipeline_record_boundary_fixed.ipynb notebooks/sleep_edf_pipeline_record_boundary_fixed.ipynb

# Optional: also keep a clearly named copy
cp ../github_ready_sleepingedf_bspc_final/notebooks/sleep_edf_pipeline_BSPC_FINAL_STRENGTHENED.ipynb notebooks/sleep_edf_pipeline_BSPC_FINAL_STRENGTHENED.ipynb

# Replace/update the exported script and documentation
cp ../github_ready_sleepingedf_bspc_final/scripts/sleep_edf_pipeline_record_boundary_fixed.py scripts/sleep_edf_pipeline_record_boundary_fixed.py
cp ../github_ready_sleepingedf_bspc_final/README.md README.md
cp ../github_ready_sleepingedf_bspc_final/requirements.txt requirements.txt
cp ../github_ready_sleepingedf_bspc_final/.gitignore .gitignore
mkdir -p docs
cp ../github_ready_sleepingedf_bspc_final/docs/SECTION_INDEX.md docs/SECTION_INDEX.md

git status
git add notebooks/sleep_edf_pipeline_record_boundary_fixed.ipynb \
        notebooks/sleep_edf_pipeline_BSPC_FINAL_STRENGTHENED.ipynb \
        scripts/sleep_edf_pipeline_record_boundary_fixed.py \
        README.md requirements.txt .gitignore docs/SECTION_INDEX.md

git commit -m "Replace notebook with BSPC final strengthened Sleep-EDF pipeline"
git push origin main
```

## Option B: use the included shell script

```bash
bash replace_original_notebook.sh /path/to/local/sleepingedf
```

The script copies the notebook, script export, README, requirements, `.gitignore`, and section index into the local repository, then prints the final `git add`, `commit`, and `push` commands.
