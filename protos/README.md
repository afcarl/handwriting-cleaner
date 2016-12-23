# protos

This directory contains the protocol buffers for handwriting that are used in this project.

## Building `handwriting.proto`

After updating the handwriting proto, it can be rebuilt using Google's protobuf tool, which can be downloaded from [here](https://github.com/google/protobuf/releases). The complete tutorial on how to do this can be found [here](https://developers.google.com/protocol-buffers/docs/pythontutorial). After downloading the protoc binary, the command for creating the proto should look something like this:

    /path/to/bin/protoc proto/handwriting.proto --python_out proto/

