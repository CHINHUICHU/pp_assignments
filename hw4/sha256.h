#ifndef __SHA256_HEADER__
#define __SHA256_HEADER__

#include <stddef.h>

#ifdef __cplusplus
extern "C"{
#endif  //__cplusplus

//--------------- DATA TYPES --------------
typedef unsigned int WORD;
typedef unsigned char BYTE;

typedef union _sha256_ctx{
	WORD h[8];
	BYTE b[32];
}SHA256;

//----------- FUNCTION DECLARATION --------
__host__ __device__ void sha256_transform(SHA256 *ctx, const BYTE *msg);
__host__ __device__ void sha256(SHA256 *ctx, const BYTE *msg, size_t len);

// Midstate functions for Bitcoin mining optimization
__host__ __device__ void sha256_init(SHA256 *ctx);
__host__ __device__ void sha256_update_64(SHA256 *ctx, const BYTE *msg);
__host__ __device__ void sha256_finalize_80(SHA256 *ctx, const BYTE *last_16_bytes);


#ifdef __cplusplus
}
#endif  //__cplusplus

#endif  //__SHA256_HEADER__
