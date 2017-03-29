# PyMatchInterface

## License

__This Software is GPL v 3.0__

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

Marcelo B. de Bianchi / m.bianchi@iag.usp.br

## What it is?

This is an interface to the Match package developed by Lorraine Lisiecki which implements a method for alignement of data series with similar shape. For a complete description of the method please check:

Lisiecki, L. E., and P. A. Lisiecki, Application of dynamic programming to the correlation of paleoclimate records, Paleoceanography, 17(D4), 1049, doi:10.1029/2001PA000733, 2002.

Match package is available from: http://www.lorraine-lisiecki.com/match.html

## What is inside?

 * This is a Python __Wrapper__; 
 * This is __not__ Match;
 * But it can help you a lot ;) !

The package comes with:

 1. A setup script that fetchs and compiles match tool in a Linux box (check SRC/setup.sh)
 2. A Python package (Match.py) with classes that represents the
   1. Data Serie file as used by Match (Class Serie)
   2. A match config file with capabilities to run match from within Python ambient (MatchConfFile)
 3. A Jupyter Notebook demonstrating all capabilities of the MatchPyGui classes for easy startup

## How do I start?

 1. Clone the repo:

```bash
git clone https://github.com/marcelobianchi/pymatch.git
```

2. Obtain and compile Match Software

```shell
cd pymatch/SRC
bash setup.sh
```

3. Check that match file is compiled and in place. It should resides inside pymatch folder, the same one with the Jupyter Notebook.

4. Start the notebook

```shell
jupyter-notebook
```
