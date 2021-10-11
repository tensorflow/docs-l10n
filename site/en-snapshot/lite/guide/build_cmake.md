# Build TensorFlow Lite with CMake

This page describes how to build and use the TensorFlow Lite library with
[CMake](https://cmake.org/) tool.

The following instructions have been tested on Ubuntu 16.04.3 64-bit PC (AMD64)
, macOS Catalina (x86_64), Windows 10 and TensorFlow devel Docker image
[tensorflow/tensorflow:devel](https://hub.docker.com/r/tensorflow/tensorflow/tags/).

**Note:** This feature is available since version 2.4.

### Step 1. Install CMake tool

It requires CMake 3.16 or higher. On Ubuntu, you can simply run the following
command.

```sh
sudo apt-get install cmake
```

Or you can follow
[the official cmake installation guide](https://cmake.org/install/)

### Step 2. Clone TensorFlow repository

```sh
git clone https://github.com/tensorflow/tensorflow.git tensorflow_src
```

**Note:** If you're using the TensorFlow Docker image, the repo is already
provided in `/tensorflow_src/`.

### Step 3. Create CMake build directory

```sh
mkdir tflite_build
cd tflite_build
```

### Step 4. Run CMake tool with configurations

#### Release build

It generates an optimized release binary by default. If you want to build for
your workstation, simply run the following command.

```sh
cmake ../tensorflow_src/tensorflow/lite
```

#### Debug build

If you need to produce a debug build which has symbol information, you need to
provide `-DCMAKE_BUILD_TYPE=Debug` option.

```sh
cmake ../tensorflow_src/tensorflow/lite -DCMAKE_BUILD_TYPE=Debug
```

#### Build with kernel unit tests

In order to be able to run kernel tests, you need to provide
'-DTFLITE_KERNEL_TEST=on' flag. Unit test cross-compilation specifics can be
found in the next subsection.

```sh
cmake ../tensorflow_src/tensorflow/lite -DTFLITE_KERNEL_TEST=on
```

#### Cross-compilation

You can use CMake to build binaries for ARM64 or Android target architectures.

In order to cross-compile the TF Lite, you namely need to provide the path to
the SDK (e.g. ARM64 SDK or NDK in Android's case) with `-DCMAKE_TOOLCHAIN_FILE`
flag.

```sh
cmake -DCMAKE_TOOLCHAIN_FILE=<CMakeToolchainFileLoc> ../tensorflow/lite/
```

##### Specifics of Android cross-compilation

For Android cross-compilation, you need to install
[Android NDK](https://developer.android.com/ndk) and provide the NDK path with
`-DCMAKE_TOOLCHAIN_FILE` flag mentioned above. You also need to set target ABI
with`-DANDROID_ABI` flag.

```sh
cmake -DCMAKE_TOOLCHAIN_FILE=<NDK path>/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a ../tensorflow_src/tensorflow/lite
```

##### Specifics of kernel (unit) tests cross-compilation

Cross-compilation of the unit tests requires flatc compiler for the host
architecture. For this purpose, there is a CMakeLists located in
`tensorflow/lite/tools/cmake/native_tools/flatbuffers` to build the flatc
compiler with CMake in advance in a separate build directory using the host
toolchain.

```sh
mkdir flatc-native-build && cd flatc-native-build
cmake ../tensorflow_src/tensorflow/lite/tools/cmake/native_tools/flatbuffers
cmake --build .
```

It is also possible **to install** the *flatc* to a custom installation location
(e.g. to a directory containing other natively-built tools instead of the CMake
build directory):

```sh
cmake -DCMAKE_INSTALL_PREFIX=<native_tools_dir> ../tensorflow_src/tensorflow/lite/tools/cmake/native_tools/flatbuffers
cmake --build .
```

For the TF Lite cross-compilation itself, additional parameter
`-DTFLITE_HOST_TOOLS_DIR=<flatc_dir_path>` pointing to the directory containing
the native *flatc* binary needs to be provided along with the
`-DTFLITE_KERNEL_TEST=on` flag mentioned above.

```sh
cmake -DCMAKE_TOOLCHAIN_FILE=${OE_CMAKE_TOOLCHAIN_FILE} -DTFLITE_KERNEL_TEST=on -DTFLITE_HOST_TOOLS_DIR=<flatc_dir_path> ../tensorflow/lite/
```

#### OpenCL GPU delegate

If your target machine has OpenCL support, you can use
[GPU delegate](https://www.tensorflow.org/lite/performance/gpu) which can
leverage your GPU power.

To configure OpenCL GPU delegate support:

```sh
cmake ../tensorflow_src/tensorflow/lite -DTFLITE_ENABLE_GPU=ON
```

**Note:** It's experimental and available starting from TensorFlow 2.5. There
could be compatibility issues. It's only verified with Android devices and
NVidia CUDA OpenCL 1.2.

### Step 5. Build TensorFlow Lite

In the tflite_build directory,

```sh
cmake --build . -j
```

**Note:** This generates a static library `libtensorflow-lite.a` in the current
directory but the library isn't self-contained since all the transitive
dependencies are not included. To use the library properly, you need to create a
CMake project. Please refer the
["Create a CMake project which uses TensorFlow Lite"](#create_a_cmake_project_which_uses_tensorflow_lite)
section.

### Step 6. Build TensorFlow Lite Benchmark Tool and Label Image Example (Optional)

In the tflite_build directory,

```sh
cmake --build . -j -t benchmark_model
```

```sh
cmake --build . -j -t label_image
```

## Available Options to build TensorFlow Lite

Here is the list of available options. You can override it with
`-D<option_name>=[ON|OFF]`. For example, `-DTFLITE_ENABLE_XNNPACK=OFF` to
disable XNNPACK which is enabled by default.

| Option Name           | Feature        | Android | Linux | macOS | Windows |
| --------------------- | -------------- | ------- | ----- | ----- | ------- |
| TFLITE_ENABLE_RUY     | Enable RUY     | ON      | OFF   | OFF   | OFF     |
:                       : matrix         :         :       :       :         :
:                       : multiplication :         :       :       :         :
:                       : library        :         :       :       :         :
| TFLITE_ENABLE_NNAPI   | Enable NNAPI   | ON      | OFF   | N/A   | N/A     |
:                       : delegate       :         :       :       :         :
| TFLITE_ENABLE_GPU     | Enable GPU     | OFF     | OFF   | N/A   | N/A     |
:                       : delegate       :         :       :       :         :
| TFLITE_ENABLE_XNNPACK | Enable XNNPACK | ON      | ON    | ON    | ON      |
:                       : delegate       :         :       :       :         :
| TFLITE_ENABLE_MMAP    | Enable MMAP    | ON      | ON    | ON    | N/A     |

## Create a CMake project which uses TensorFlow Lite

Here is the CMakeLists.txt of
[TFLite minimal example](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/examples/minimal).

You need to have add_subdirectory() for TensorFlow Lite directory and link
`tensorflow-lite` with target_link_libraries().

```
cmake_minimum_required(VERSION 3.16)
project(minimal C CXX)

set(TENSORFLOW_SOURCE_DIR "" CACHE PATH
  "Directory that contains the TensorFlow project" )
if(NOT TENSORFLOW_SOURCE_DIR)
  get_filename_component(TENSORFLOW_SOURCE_DIR
    "${CMAKE_CURRENT_LIST_DIR}/../../../../" ABSOLUTE)
endif()

add_subdirectory(
  "${TENSORFLOW_SOURCE_DIR}/tensorflow/lite"
  "${CMAKE_CURRENT_BINARY_DIR}/tensorflow-lite" EXCLUDE_FROM_ALL)

add_executable(minimal minimal.cc)
target_link_libraries(minimal tensorflow-lite)
```

## Build TensorFlow Lite C library

If you want to build TensorFlow Lite shared library for
[C API](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/c/README.md),
follow [step 1](#step-1-install-cmake-tool) to
[step 3](#step-3-create-cmake-build-directory) first. After that, run the
following commands.

```sh
cmake ../tensorflow_src/tensorflow/lite/c
cmake --build . -j
```

This command generates the following shared library in the current directory.

Platform | Library name
-------- | -------------------------
Linux    | libtensorflowlite_c.so
macOS    | libtensorflowlite_c.dylib
Windows  | tensorflowlite_c.dll

**Note:** You need necessary headers (c_api.h, c_api_experimental.h and
common.h) to use the generated shared library.
