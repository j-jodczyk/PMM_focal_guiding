
#define VCL_NAMESPACE vcl
#include "../vectorclass/vectorclass.h"

namespace pmm {

typedef vcl::Vec4f float4;
typedef vcl::Vec8f float8;

float sum(const float8 &a) { return vcl::horizontal_add(a); }
float sum(const float4 &a) {	return vcl::horizontal_add(a); }

}

