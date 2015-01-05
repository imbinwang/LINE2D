#include <stdint.h>
#include <fcntl.h>
#include <io.h>

#include <google\protobuf\text_format.h>
#include <google\protobuf\io\zero_copy_stream_impl.h>
#include <google\protobuf\io\coded_stream.h>

#include <algorithm>
#include <string>
#include <vector>
#include <fstream>

#include "..\..\include\util\ProtobufIO.h"
#include "..\..\include\util\Config.pb.h"

using std::fstream;
using std::ios;
using std::string;
using google::protobuf::io::FileInputStream;
using google::protobuf::io::FileOutputStream;
using google::protobuf::io::ZeroCopyInputStream;
using google::protobuf::io::CodedInputStream;
using google::protobuf::io::ZeroCopyOutputStream;
using google::protobuf::io::CodedOutputStream;
using google::protobuf::Message;

namespace rl2d 
{
	bool ReadProtoFromTextFile(const char* filename, Message* proto) 
	{
		int fd = open(filename, O_RDONLY);
		CHECK_NE(fd, -1) << "File not found: " << filename;
		FileInputStream* input = new FileInputStream(fd);
		bool success = google::protobuf::TextFormat::Parse(input, proto);
		delete input;
		close(fd);
		return success;
	}

	void WriteProtoToTextFile(const Message& proto, const char* filename) 
	{
		int fd = open(filename, O_WRONLY);
		FileOutputStream* output = new FileOutputStream(fd);
		CHECK(google::protobuf::TextFormat::Print(proto, output));
		delete output;
		close(fd);
	}

	bool ReadProtoFromBinaryFile(const char* filename, Message* proto) 
	{
		int fd = open(filename, O_RDONLY | O_BINARY);
		CHECK_NE(fd, -1) << "File not found: " << filename;
		ZeroCopyInputStream* raw_input = new FileInputStream(fd);
		CodedInputStream* coded_input = new CodedInputStream(raw_input);
		coded_input->SetTotalBytesLimit(1073741824, 536870912);

		bool success = proto->ParseFromCodedStream(coded_input);

		delete coded_input;
		delete raw_input;
		close(fd);
		return success;
	}

	void WriteProtoToBinaryFile(const Message& proto, const char* filename)
	{
		fstream output(filename, ios::out | ios::trunc | ios::binary);
		CHECK(proto.SerializeToOstream(&output));
	}

}  // namespace rl2d
