## Compression for AI

### Environment setup

1. Fork this repository;
2. Create a directory called ExternalDependencies;

- Install SZ3 with python binding
3. Execute `cd ExternalDependencies && git clone git@github.com:szcompressor/SZ3.git`;
4. Edit `build_SZ3.sh` by changing the PROJHOME to a suitable directory;
5. Execute `source build_SZ3.sh` from the home directory;

- Install zfpy python library
6. Execute `pip install zfpy`;

### Compression experiments
Go to notebooks/comp4ai.ipynb

1. Make sure the data folder, containing all the feature maps, are in the home directory;
2. Change `HOME` in `comp4ai.ipynb` to the same directory;
3. Run `comp4ai.ipynb` and measure the performance of SZ and zfp compression.
