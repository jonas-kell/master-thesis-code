# Installation of the `EspressoII` program to minimize logical expressions

```shell
cd computation-scripts/logic-optimization
git clone https://github.com/classabbyamp/espresso-logic.git program-generation/espresso
cd program-generation/espresso/espresso-src
make
cd ../bin
mv espresso ./../../bin
cd ../..
rm -rf espresso
cd ..
```

Bit of documentation and newer release, but doesn't compile on my linux machine (probably a skill issue):
https://github.com/Gigantua/Espresso

## Usage:

```shell
python3 vpartsDifferenceFlippingA.py > tmpfA.pla
./program-generation/bin/espresso tmpfA.pla > outputFlippingA.txt
```
