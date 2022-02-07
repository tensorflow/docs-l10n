
## Development

The document contains the necessary information for setting up the development environment
and building the `tensorflow-io` package from source on various platforms. Once the setup is completed please refer to the [STYLE_GUIDE](https://github.com/tensorflow/io/blob/master/STYLE_GUIDE.md) for guidelines on adding new ops.

### IDE Setup

For instructions on how to configure Visual Studio Code for developing TensorFlow I/O, please refer to this [doc](https://github.com/tensorflow/io/blob/master/docs/vscode.md).

### Lint

TensorFlow I/O's code conforms to Bazel Buildifier, Clang Format, Black, and Pyupgrade.
Please use the following command to check the source code and identify lint issues:

```
# Install Bazelisk (manage bazel version implicitly)
$ curl -sSOL https://github.com/bazelbuild/bazelisk/releases/download/v1.11.0/bazelisk-linux-amd64
$ sudo mv bazelisk-linux-amd64 /usr/local/bin/bazel
$ sudo chmod +x /usr/local/bin/bazel
$ bazel run //tools/lint:check
```

For Bazel Buildifier and Clang Format, the following command will automatically identify
and fix any lint errors:
```
$ bazel run //tools/lint:lint
```

Alternatively, if you only want to perform lint check using individual linters,
then you can selectively pass `black`, `pyupgrade`, `bazel`, or `clang` to the above commands.

For example, a `black` specific lint check can be done using:
```
$ bazel run //tools/lint:check -- black
```

Lint fix using Bazel Buildifier and Clang Format can be done using:
```
$ bazel run //tools/lint:lint -- bazel clang
```

Lint check using `black` and `pyupgrade` for an individual python file can be done using:
```
$ bazel run //tools/lint:check -- black pyupgrade -- tensorflow_io/python/ops/version_ops.py
```

Lint fix an individual python file with black and pyupgrade using:
```
$ bazel run //tools/lint:lint -- black pyupgrade --  tensorflow_io/python/ops/version_ops.py
```

### Python

#### macOS

On macOS Catalina 10.15.7, it is possible to build tensorflow-io with
system provided python 3.8.2. Both `tensorflow` and `bazel` are needed to do so.

NOTE: The system default python 3.8.2 on macOS 10.15.7 will cause `regex` installation
error caused by compiler option of `-arch arm64 -arch x86_64` (similar to the issue
mentioned in https://github.com/giampaolo/psutil/issues/1832). To overcome this issue
`export ARCHFLAGS="-arch x86_64"` will be needed to remove arm64 build option.

```sh
#!/usr/bin/env bash

# Disable arm64 build by specifying only x86_64 arch.
# Only needed for macOS's system default python 3.8.2 on macOS 10.15.7
export ARCHFLAGS="-arch x86_64"

# Use following command to check if Xcode is correctly installed:
xcodebuild -version

# Show macOS's default python3
python3 --version

# Install Bazelisk (manage bazel version implicitly)
brew install bazelisk

# Install tensorflow and configure bazel
sudo ./configure.sh

# Add any optimization on bazel command, e.g., --compilation_mode=opt,
#   --copt=-msse4.2, --remote_cache=, etc.
# export BAZEL_OPTIMIZATION=

# Build shared libraries
bazel build -s --verbose_failures $BAZEL_OPTIMIZATION //tensorflow_io/... //tensorflow_io_gcs_filesystem/...

# Once build is complete, shared libraries will be available in
# `bazel-bin/tensorflow_io/core`, `bazel-bin/tensorflow_io/python/ops` and
# it is possible to run tests with `pytest`, e.g.:
sudo python3 -m pip install pytest
TFIO_DATAPATH=bazel-bin python3 -m pytest -s -v tests/test_serialization.py
```

NOTE: When running pytest, `TFIO_DATAPATH=bazel-bin` has to be passed so that python can utilize the generated shared libraries after the build process.

##### Troubleshoot

If Xcode is installed, but `$ xcodebuild -version` is not displaying the expected output, you might need to enable Xcode command line with the command:

`$ xcode-select -s /Applications/Xcode.app/Contents/Developer`.

A terminal restart might be required for the changes to take effect.

Sample output:

```
$ xcodebuild -version
Xcode 12.2
Build version 12B45b
```

#### Linux

Development of tensorflow-io on Linux is similar to macOS. The required packages
are gcc, g++, git, bazel, and python 3. Newer versions of gcc or python, other than the default system installed
versions might be required though.

##### Ubuntu 20.04

Ubuntu 20.04 requires gcc/g++, git, and python 3. The following will install dependencies and build
the shared libraries on Ubuntu 20.04:
```sh
#!/usr/bin/env bash

# Install gcc/g++, git, unzip/curl (for bazel), and python3
sudo apt-get -y -qq update
sudo apt-get -y -qq install gcc g++ git unzip curl python3-pip

# Install Bazelisk (manage bazel version implicitly)
curl -sSOL https://github.com/bazelbuild/bazelisk/releases/download/v1.11.0/bazelisk-linux-amd64
sudo mv bazelisk-linux-amd64 /usr/local/bin/bazel
sudo chmod +x /usr/local/bin/bazel

# Upgrade pip
sudo python3 -m pip install -U pip

# Install tensorflow and configure bazel
sudo ./configure.sh

# Alias python3 to python, needed by bazel
sudo ln -s /usr/bin/python3 /usr/bin/python

# Add any optimization on bazel command, e.g., --compilation_mode=opt,
#   --copt=-msse4.2, --remote_cache=, etc.
# export BAZEL_OPTIMIZATION=

# Build shared libraries
bazel build -s --verbose_failures $BAZEL_OPTIMIZATION //tensorflow_io/... //tensorflow_io_gcs_filesystem/...

# Once build is complete, shared libraries will be available in
# `bazel-bin/tensorflow_io/core`, `bazel-bin/tensorflow_io/python/ops` and
# it is possible to run tests with `pytest`, e.g.:
sudo python3 -m pip install pytest
TFIO_DATAPATH=bazel-bin python3 -m pytest -s -v tests/test_serialization.py
```

##### CentOS 8

The steps to build shared libraries for CentOS 8 is similar to Ubuntu 20.04 above
except that
```
sudo yum install -y python3 python3-devel gcc gcc-c++ git unzip which make
```
should be used instead to install gcc/g++, git, unzip/which (for bazel), and python3.

##### CentOS 7

On CentOS 7, the default python and gcc version are too old to build tensorflow-io's shared
libraries (.so). The gcc provided by Developer Toolset and rh-python36 should be used instead.
Also, the libstdc++ has to be linked statically to avoid discrepancy of libstdc++ installed on
CentOS vs. newer gcc version by devtoolset.

Furthermore, a special flag `--//tensorflow_io/core:static_build` has to be passed to Bazel
in order to avoid duplication of symbols in statically linked libraries for file system
plugins.

The following will install bazel, devtoolset-9, rh-python36, and build the shared libraries:
```sh
#!/usr/bin/env bash

# Install centos-release-scl, then install gcc/g++ (devtoolset), git, and python 3
sudo yum install -y centos-release-scl
sudo yum install -y devtoolset-9 git rh-python36 make

# Install Bazelisk (manage bazel version implicitly)
curl -sSOL https://github.com/bazelbuild/bazelisk/releases/download/v1.11.0/bazelisk-linux-amd64
sudo mv bazelisk-linux-amd64 /usr/local/bin/bazel
sudo chmod +x /usr/local/bin/bazel

# Upgrade pip
scl enable rh-python36 devtoolset-9 \
    'python3 -m pip install -U pip'

# Install tensorflow and configure bazel with rh-python36
scl enable rh-python36 devtoolset-9 \
    './configure.sh'

# Add any optimization on bazel command, e.g., --compilation_mode=opt,
#   --copt=-msse4.2, --remote_cache=, etc.
# export BAZEL_OPTIMIZATION=

# Build shared libraries, notice the passing of --//tensorflow_io/core:static_build
BAZEL_LINKOPTS="-static-libstdc++ -static-libgcc" BAZEL_LINKLIBS="-lm -l%:libstdc++.a" \
  scl enable rh-python36 devtoolset-9 \
    'bazel build -s --verbose_failures $BAZEL_OPTIMIZATION --//tensorflow_io/core:static_build //tensorflow_io/...'

# Once build is complete, shared libraries will be available in
# `bazel-bin/tensorflow_io/core`, `bazel-bin/tensorflow_io/python/ops` and
# it is possible to run tests with `pytest`, e.g.:
scl enable rh-python36 devtoolset-9 \
    'python3 -m pip install pytest'

TFIO_DATAPATH=bazel-bin \
  scl enable rh-python36 devtoolset-9 \
    'python3 -m pytest -s -v tests/test_serialization.py'
```

#### Docker

For Python development, a reference Dockerfile [here](tools/docker/devel.Dockerfile) can be
used to build the TensorFlow I/O package (`tensorflow-io`) from source. Additionally, the
pre-built devel images can be used as well:
```sh
# Pull (if necessary) and start the devel container
$ docker run -it --rm --name tfio-dev --net=host -v ${PWD}:/v -w /v tfsigio/tfio:latest-devel bash

# Inside the docker container, ./configure.sh will install TensorFlow or use existing install
(tfio-dev) root@docker-desktop:/v$ ./configure.sh

# Clean up exisiting bazel build's (if any)
(tfio-dev) root@docker-desktop:/v$ rm -rf bazel-*

# Build TensorFlow I/O C++. For compilation optimization flags, the default (-march=native)
# optimizes the generated code for your machine's CPU type.
# Reference: https://www.tensorflow.orginstall/source#configuration_options).

# NOTE: Based on the available resources, please change the number of job workers to:
# -j 4/8/16 to prevent bazel server terminations and resource oriented build errors.

(tfio-dev) root@docker-desktop:/v$ bazel build -j 8 --copt=-msse4.2 --copt=-mavx --compilation_mode=opt --verbose_failures --test_output=errors --crosstool_top=//third_party/toolchains/gcc7_manylinux2010:toolchain //tensorflow_io/... //tensorflow_io_gcs_filesystem/...


# Run tests with PyTest, note: some tests require launching additional containers to run (see below)
(tfio-dev) root@docker-desktop:/v$ pytest -s -v tests/
# Build the TensorFlow I/O package
(tfio-dev) root@docker-desktop:/v$ python setup.py bdist_wheel
```

A package file `dist/tensorflow_io-*.whl` will be generated after a build is successful.

NOTE: When working in the Python development container, an environment variable
`TFIO_DATAPATH` is automatically set to point tensorflow-io to the shared C++
libraries built by Bazel to run `pytest` and build the `bdist_wheel`. Python
`setup.py` can also accept `--data [path]` as an argument, for example
`python setup.py --data bazel-bin bdist_wheel`.

NOTE: While the tfio-dev container gives developers an easy to work with
environment, the released whl packages are built differently due to manylinux2010
requirements. Please check [Build Status and CI] section for more details
on how the released whl packages are generated.

#### Python Wheels

It is possible to build python wheels after bazel build is complete with the following command:
```
$ python setup.py bdist_wheel --data bazel-bin
```
The .whl file will be available in dist directory. Note the bazel binary directory `bazel-bin`
has to be passed with `--data` args in order for setup.py to locate the necessary share objects,
as `bazel-bin` is outside of the `tensorflow_io` package directory.

Alternatively, source install could be done with:
```
$ TFIO_DATAPATH=bazel-bin python -m pip install .
```
with `TFIO_DATAPATH=bazel-bin` passed for the same reason.

Note installing with `-e` is different from the above. The
```
$ TFIO_DATAPATH=bazel-bin python -m pip install -e .
```
will not install shared object automatically even with `TFIO_DATAPATH=bazel-bin`. Instead,
`TFIO_DATAPATH=bazel-bin` has to be passed everytime the program is run after the install:
```
$ TFIO_DATAPATH=bazel-bin python

>>> import tensorflow_io as tfio
>>> ...
```

#### Testing

Some tests require launching a test container or start a local instance
of the associated tool before running. For example, to run kafka
related tests which will start a local instance of kafka, zookeeper and schema-registry,
use:

```sh
# Start the local instances of kafka, zookeeper and schema-registry
$ bash -x -e tests/test_kafka/kafka_test.sh

# Run the tests
$ TFIO_DATAPATH=bazel-bin pytest -s -vv tests/test_kafka.py
```

Testing `Datasets` associated with tools such as `Elasticsearch` or `MongoDB`
require docker to be available on the system. In such scenarios, use:


```sh
# Start elasticsearch within docker container
$ bash tests/test_elasticsearch/elasticsearch_test.sh start

# Run the tests
$ TFIO_DATAPATH=bazel-bin pytest -s -vv tests/test_elasticsearch.py

# Stop and remove the container
$ bash tests/test_elasticsearch/elasticsearch_test.sh stop
```

Additionally, testing some features of `tensorflow-io` doesn't require you to spin up
any additional tools as the data has been provided in the `tests` directory itself.
For example, to run tests related to `parquet` dataset's, use:

```sh
# Just run the test
$ TFIO_DATAPATH=bazel-bin pytest -s -vv tests/test_parquet.py
```


### R

We provide a reference Dockerfile [here](R-package/scripts/Dockerfile) for you
so that you can use the R package directly for testing. You can build it via:
```sh
$ docker build -t tfio-r-dev -f R-package/scripts/Dockerfile .
```

Inside the container, you can start your R session, instantiate a `SequenceFileDataset`
from an example [Hadoop SequenceFile](https://wiki.apache.org/hadoop/SequenceFile)
[string.seq](R-package/tests/testthat/testdata/string.seq), and then use any [transformation functions](https://tensorflow.rstudio.com/tools/tfdatasets/articles/introduction.html#transformations) provided by [tfdatasets package](https://tensorflow.rstudio.com/tools/tfdatasets/) on the dataset like the following:

```r
library(tfio)
dataset <- sequence_file_dataset("R-package/tests/testthat/testdata/string.seq") %>%
    dataset_repeat(2)

sess <- tf$Session()
iterator <- make_iterator_one_shot(dataset)
next_batch <- iterator_get_next(iterator)

until_out_of_range({
  batch <- sess$run(next_batch)
  print(batch)
})
```
