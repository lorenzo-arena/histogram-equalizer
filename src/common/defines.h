#ifndef __DEFINES_H__
#define __DEFINES_H__

#include "errors.h"
#include "cexception/lib/CException.h"

#define N_BINS 512

#define check_pointer(x) { if(NULL == (x)) { Throw(UNALLOCATED_MEMORY); } }

#endif
