# Copyright (C) 2026  EMBL/Maximilian Dreimann, EMBL/Svetlana Dodonova
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""Shared path handling for all scripts. Set once via environment variables or command-line flags:
   STAR_DIR  folder holding the RELION class STAR files and reference MRC maps
   OUT_DIR   folder where every script writes its results (created if missing)"""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def get_args(extra=None, description=""):
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--star-dir", default=os.environ.get("STAR_DIR", "./input"), help="folder with STAR files / MRC maps")
    ap.add_argument("--out-dir", default=os.environ.get("OUT_DIR", "./output"), help="results folder")
    if extra:
        extra(ap)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    return a
