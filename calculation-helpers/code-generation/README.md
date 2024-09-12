# Installation of the `EspressoII` program to minimize logical expressions

```shell
cd calculation-helpers/code-generation
git clone https://github.com/classabbyamp/espresso-logic.git program-generation/espresso
cd program-generation/espresso/espresso-src
make
cd ../bin
mv espresso ./../../bin
cd ../..
rm -rf espresso
cd ..
```
