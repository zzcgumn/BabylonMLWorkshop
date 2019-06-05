# Iris Classification Model with Swift for TensorFlow.

To install Swift for Tensor Flow head over the GitHub repository and download the [Xcode buildchain](https://github.com/tensorflow/swift/blob/master/Installation.md). By default the package installer will install a version of the swift compiler in

```
/Library/Developer/Toolchains/
```

Make sure that the `swift-latest` soft link points to the toolchain that was downloaded.

According to the instructions one should be able to set the toolchain in Xcode preferences but this does not work correctly. Instead, one has to use the swift package manager to create a project

```
mkdir MyAwesomeMLProject
cd MyAwesomeMLProject
/Library/Developer/Toolchains/swift-latest/usr/bin/swift-package init --type executable
```
