# Spatial analysis of chromatin fibers of *Thermococcus kodakarensis*

Python code for the geometric analysis of hypernucleosome particles in cryo-electron tomograms of *T. kodakarensis* KU216 and ΔhtkB
(nearest-neighbour distances, neighbour orientations, fibers, random control and statistics).

## Requirements
Python ≥ 3.10 and `pip install -r requirements.txt` (numpy, scipy, pandas, starfile, mrcfile).

## Input
A folder with the STAR files `WT_<n>dimer.star` and `deltaB_<n>dimer.star` and the matching class-average maps
`*.mrc`. Class and tomogram definitions are in `classes.py`.

## Usage
Set the input and output folders once (or pass `--star-dir` / `--out-dir` to each script) and run the numbered scripts 01-09 in order in 
```
export STAR_DIR=/path/to/input
export OUT_DIR=/path/to/results
cd scripts
python 01_extract_particles.py       # particle table and superhelical axes
python 02_length_model.py            # particle length from the class maps
python 03_duplicates.py              # duplicate picks
python 04_candidate_contacts.py      # edge-to-edge distances of close pairs (slow, resumes if interrupted)
python 05_nearest_neighbour.py       # nearest neighbours, touching, isolated particles
python 06_heatmap_pairs.py           # neighbour pairs and relative orientations
python 07_fibers.py                  # fibers, DNA condensed, fiber end-to-end distance
python 08_null_model.py              # random control (slow, resumes if interrupted)
python 09_statistics.py              # Mann-Whitney U, effect sizes, bootstrap confidence intervals
```
Each step needs the output of the previous ones. The whole run takes about 35 minutes on one CPU core.

## License
GNU General Public License v3.0 or later (see `LICENSE`).
