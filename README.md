# Free Refill

Free Refill is the free REcursive FILtering Library.  
It provides functionality to create recursive filters simply and hassle free.

## Contributing

Coding style: [Google Style Guide](https://google.github.io/styleguide/cppguide.html)

Setting up git hooks for development:
`python devtools/init-git-hooks.py`

## Ubuntu Installation Notes

The installation requires:
`sudo apt-get install libeigen3-dev libgtest-dev`

A manual compilation of gtest may be necessary:

```shell
cd /usr/src/gtest
sudo cmake .
sudo make
sudo mv libg* /usr/lib/
```

A discussion about this can be found [here](https://askubuntu.com/questions/145887/why-no-library-files-installed-for-google-test/14591). In the future gtest will be shipped as part of the library following Google's [advice](https://github.com/google/googletest/blob/master/googletest/docs/FAQ.md) not to install a pre-compiled copy of Google Test.

## Documentation

To generate documentation for this library, [doxygen](http://www.doxygen.org/) has to be installed. Then run the following commands:

```shell
cd doc
doxygen config.DoxyFile
```
