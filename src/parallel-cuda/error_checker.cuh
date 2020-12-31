#ifndef __ERROR_CHECKER_H__
#define __ERROR_CHECKER_H__

extern "C" {
    #include "cexception/lib/CException.h"
    #include "log.h"
}

#define gpuErrorCheck(code) { gpuAssert((code), __FILE__, __LINE__); }

#define gpuErrorCheckLastCode() { gpuAssert((cudaGetLastError()), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess) 
    {
        log_error("gpuErrorCheck: %s %s:%d", cudaGetErrorString(code), file, line);

        if(abort)
        {
            Throw(code);
        }
    }
}

#endif
