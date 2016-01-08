@echo Config.pb.h is being generated

".\protoc" -I="." --cpp_out=".\proto" ".\Config.proto"

@pause


