if exist ".\Config.pb.h" (
    echo Config.pb.h remains the same as before
) else (
    echo Config.pb.h is being generated
    ".\protoc" -I="." --cpp_out=".\proto" ".\Config.proto"
)

pause


