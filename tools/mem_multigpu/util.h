/* Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <iostream>
#include <stdint.h>
#include <string>
#include <zstd.h>
#include <sstream>
#include <cstdio>
#include <mutex>

int fread_ss(void *buffIn, size_t toRead, std::stringstream &ss);
void write_to_stream(ZSTD_CStream *cs, std::string msg, void *buffIn,
    void *buffOut, size_t buffInSize, size_t buffOutSize,
    FILE *fout);

class Logger {
  private:
  FILE *fout;
  size_t buffInSize;
  size_t buffOutSize;
  void *buffIn;
  void *buffOut;
  std::mutex log_mutex;
  ZSTD_CStream *cs;
  public:

  Logger(std::string filename) {
    cs = ZSTD_createCStream();

    // TODO: Try changing compression level to 3
    ZSTD_initCStream(cs, 1);

    buffInSize = ZSTD_CStreamInSize();
    buffIn = malloc(buffInSize);
    buffOutSize = ZSTD_CStreamOutSize();
    buffOut = malloc(buffOutSize);
    fout = fopen(filename.c_str(), "wb");
  }

  ~Logger() {
    ZSTD_outBuffer output = {buffOut, buffOutSize, 0};
    size_t const remainingToFlush = ZSTD_endStream(cs, &output); /* close frame */
    if (remainingToFlush) {
      fprintf(stderr, "not fully flushed");
      exit(13);
    }

    fwrite(buffOut, 1, output.pos, fout);
    fclose(fout);

    free(buffOut);
    free(buffIn);
  }

  void log(std::string msg) {
    log_mutex.lock();
    write_to_stream(cs, msg, buffIn, buffOut, buffInSize, buffOutSize, fout);
    log_mutex.unlock();
  }
};


int fread_ss(void *buffIn, size_t toRead, std::stringstream &ss) {
  ss.read((char *)buffIn, toRead);
  return ss.gcount();
}

void write_to_stream(ZSTD_CStream *cs, std::string msg, void *buffIn,
    void *buffOut, size_t buffInSize, size_t buffOutSize,
    FILE *fout) {

  std::stringstream ss(msg);
  size_t read, toRead = buffInSize;

  while (!ss.eof()) {
    read = fread_ss(buffIn, toRead, ss);
    if (read == 0)
      continue;

    ZSTD_inBuffer input = {buffIn, read, 0};
    while (input.pos < input.size) {
      ZSTD_outBuffer output = {buffOut, buffOutSize, 0};
      toRead = ZSTD_compressStream(cs, &output, &input);
      if (ZSTD_isError(toRead)) {
        fprintf(stderr, "ZSTD_compressStream() error : %s \n",
            ZSTD_getErrorName(toRead));
        exit(12);
      }

      if (toRead > buffInSize)
        toRead = buffInSize;

      fwrite(buffOut, 1, output.pos, fout);
    }
  }
}


const char* find_cbid_name(nvbit_api_cuda_t cbid) {
  switch (cbid) {
    case   1:   return "cuInit";
    case   2:   return "cuDriverGetVersion";
    case   3:   return "cuDeviceGet";
    case   4:   return "cuDeviceGetCount";
    case   5:   return "cuDeviceGetName";
    case   6:   return "cuDeviceComputeCapability";
    case   7:   return "cuDeviceTotalMem";
    case   8:   return "cuDeviceGetProperties";
    case   9:   return "cuDeviceGetAttribute";
    case  10:   return "cuCtxCreate";
    case  11:   return "cuCtxDestroy";
    case  12:   return "cuCtxAttach";
    case  13:   return "cuCtxDetach";
    case  14:   return "cuCtxPushCurrent";
    case  15:   return "cuCtxPopCurrent";
    case  16:   return "cuCtxGetDevice";
    case  17:   return "cuCtxSynchronize";
    case  18:   return "cuModuleLoad";
    case  19:   return "cuModuleLoadData";
    case  20:   return "cuModuleLoadDataEx";
    case  21:   return "cuModuleLoadFatBinary";
    case  22:   return "cuModuleUnload";
    case  23:   return "cuModuleGetFunction";
    case  24:   return "cuModuleGetGlobal";
    case  25:   return "cu64ModuleGetGlobal";
    case  26:   return "cuModuleGetTexRef";
    case  27:   return "cuMemGetInfo";
    case  28:   return "cu64MemGetInfo";
    case  29:   return "cuMemAlloc";
    case  30:   return "cu64MemAlloc";
    case  31:   return "cuMemAllocPitch";
    case  32:   return "cu64MemAllocPitch";
    case  33:   return "cuMemFree";
    case  34:   return "cu64MemFree";
    case  35:   return "cuMemGetAddressRange";
    case  36:   return "cu64MemGetAddressRange";
    case  37:   return "cuMemAllocHost";
    case  38:   return "cuMemFreeHost";
    case  39:   return "cuMemHostAlloc";
    case  40:   return "cuMemHostGetDevicePointer";
    case  41:   return "cu64MemHostGetDevicePointer";
    case  42:   return "cuMemHostGetFlags";
    case  43:   return "cuMemcpyHtoD";
    case  44:   return "cu64MemcpyHtoD";
    case  45:   return "cuMemcpyDtoH";
    case  46:   return "cu64MemcpyDtoH";
    case  47:   return "cuMemcpyDtoD";
    case  48:   return "cu64MemcpyDtoD";
    case  49:   return "cuMemcpyDtoA";
    case  50:   return "cu64MemcpyDtoA";
    case  51:   return "cuMemcpyAtoD";
    case  52:   return "cu64MemcpyAtoD";
    case  53:   return "cuMemcpyHtoA";
    case  54:   return "cuMemcpyAtoH";
    case  55:   return "cuMemcpyAtoA";
    case  56:   return "cuMemcpy2D";
    case  57:   return "cuMemcpy2DUnaligned";
    case  58:   return "cuMemcpy3D";
    case  59:   return "cu64Memcpy3D";
    case  60:   return "cuMemcpyHtoDAsync";
    case  61:   return "cu64MemcpyHtoDAsync";
    case  62:   return "cuMemcpyDtoHAsync";
    case  63:   return "cu64MemcpyDtoHAsync";
    case  64:   return "cuMemcpyDtoDAsync";
    case  65:   return "cu64MemcpyDtoDAsync";
    case  66:   return "cuMemcpyHtoAAsync";
    case  67:   return "cuMemcpyAtoHAsync";
    case  68:   return "cuMemcpy2DAsync";
    case  69:   return "cuMemcpy3DAsync";
    case  70:   return "cu64Memcpy3DAsync";
    case  71:   return "cuMemsetD8";
    case  72:   return "cu64MemsetD8";
    case  73:   return "cuMemsetD16";
    case  74:   return "cu64MemsetD16";
    case  75:   return "cuMemsetD32";
    case  76:   return "cu64MemsetD32";
    case  77:   return "cuMemsetD2D8";
    case  78:   return "cu64MemsetD2D8";
    case  79:   return "cuMemsetD2D16";
    case  80:   return "cu64MemsetD2D16";
    case  81:   return "cuMemsetD2D32";
    case  82:   return "cu64MemsetD2D32";
    case  83:   return "cuFuncSetBlockShape";
    case  84:   return "cuFuncSetSharedSize";
    case  85:   return "cuFuncGetAttribute";
    case  86:   return "cuFuncSetCacheConfig";
    case  87:   return "cuArrayCreate";
    case  88:   return "cuArrayGetDescriptor";
    case  89:   return "cuArrayDestroy";
    case  90:   return "cuArray3DCreate";
    case  91:   return "cuArray3DGetDescriptor";
    case  92:   return "cuTexRefCreate";
    case  93:   return "cuTexRefDestroy";
    case  94:   return "cuTexRefSetArray";
    case  95:   return "cuTexRefSetAddress";
    case  96:   return "cu64TexRefSetAddress";
    case  97:   return "cuTexRefSetAddress2D";
    case  98:   return "cu64TexRefSetAddress2D";
    case  99:   return "cuTexRefSetFormat";
    case 100:   return "cuTexRefSetAddressMode";
    case 101:   return "cuTexRefSetFilterMode";
    case 102:   return "cuTexRefSetFlags";
    case 103:   return "cuTexRefGetAddress";
    case 104:   return "cu64TexRefGetAddress";
    case 105:   return "cuTexRefGetArray";
    case 106:   return "cuTexRefGetAddressMode";
    case 107:   return "cuTexRefGetFilterMode";
    case 108:   return "cuTexRefGetFormat";
    case 109:   return "cuTexRefGetFlags";
    case 110:   return "cuParamSetSize";
    case 111:   return "cuParamSeti";
    case 112:   return "cuParamSetf";
    case 113:   return "cuParamSetv";
    case 114:   return "cuParamSetTexRef";
    case 115:   return "cuLaunch";
    case 116:   return "cuLaunchGrid";
    case 117:   return "cuLaunchGridAsync";
    case 118:   return "cuEventCreate";
    case 119:   return "cuEventRecord";
    case 120:   return "cuEventQuery";
    case 121:   return "cuEventSynchronize";
    case 122:   return "cuEventDestroy";
    case 123:   return "cuEventElapsedTime";
    case 124:   return "cuStreamCreate";
    case 125:   return "cuStreamQuery";
    case 126:   return "cuStreamSynchronize";
    case 127:   return "cuStreamDestroy";
    case 128:   return "cuGraphicsUnregisterResource";
    case 129:   return "cuGraphicsSubResourceGetMappedArray";
    case 130:   return "cuGraphicsResourceGetMappedPointer";
    case 131:   return "cu64GraphicsResourceGetMappedPointer";
    case 132:   return "cuGraphicsResourceSetMapFlags";
    case 133:   return "cuGraphicsMapResources";
    case 134:   return "cuGraphicsUnmapResources";
    case 135:   return "cuGetExportTable";
    case 136:   return "cuCtxSetLimit";
    case 137:   return "cuCtxGetLimit";
    case 138:   return "cuD3D10GetDevice";
    case 139:   return "cuD3D10CtxCreate";
    case 140:   return "cuGraphicsD3D10RegisterResource";
    case 141:   return "cuD3D10RegisterResource";
    case 142:   return "cuD3D10UnregisterResource";
    case 143:   return "cuD3D10MapResources";
    case 144:   return "cuD3D10UnmapResources";
    case 145:   return "cuD3D10ResourceSetMapFlags";
    case 146:   return "cuD3D10ResourceGetMappedArray";
    case 147:   return "cuD3D10ResourceGetMappedPointer";
    case 148:   return "cuD3D10ResourceGetMappedSize";
    case 149:   return "cuD3D10ResourceGetMappedPitch";
    case 150:   return "cuD3D10ResourceGetSurfaceDimensions";
    case 151:   return "cuD3D11GetDevice";
    case 152:   return "cuD3D11CtxCreate";
    case 153:   return "cuGraphicsD3D11RegisterResource";
    case 154:   return "cuD3D9GetDevice";
    case 155:   return "cuD3D9CtxCreate";
    case 156:   return "cuGraphicsD3D9RegisterResource";
    case 157:   return "cuD3D9GetDirect3DDevice";
    case 158:   return "cuD3D9RegisterResource";
    case 159:   return "cuD3D9UnregisterResource";
    case 160:   return "cuD3D9MapResources";
    case 161:   return "cuD3D9UnmapResources";
    case 162:   return "cuD3D9ResourceSetMapFlags";
    case 163:   return "cuD3D9ResourceGetSurfaceDimensions";
    case 164:   return "cuD3D9ResourceGetMappedArray";
    case 165:   return "cuD3D9ResourceGetMappedPointer";
    case 166:   return "cuD3D9ResourceGetMappedSize";
    case 167:   return "cuD3D9ResourceGetMappedPitch";
    case 168:   return "cuD3D9Begin";
    case 169:   return "cuD3D9End";
    case 170:   return "cuD3D9RegisterVertexBuffer";
    case 171:   return "cuD3D9MapVertexBuffer";
    case 172:   return "cuD3D9UnmapVertexBuffer";
    case 173:   return "cuD3D9UnregisterVertexBuffer";
    case 174:   return "cuGLCtxCreate";
    case 175:   return "cuGraphicsGLRegisterBuffer";
    case 176:   return "cuGraphicsGLRegisterImage";
    case 177:   return "cuWGLGetDevice";
    case 178:   return "cuGLInit";
    case 179:   return "cuGLRegisterBufferObject";
    case 180:   return "cuGLMapBufferObject";
    case 181:   return "cuGLUnmapBufferObject";
    case 182:   return "cuGLUnregisterBufferObject";
    case 183:   return "cuGLSetBufferObjectMapFlags";
    case 184:   return "cuGLMapBufferObjectAsync";
    case 185:   return "cuGLUnmapBufferObjectAsync";
    case 186:   return "cuVDPAUGetDevice";
    case 187:   return "cuVDPAUCtxCreate";
    case 188:   return "cuGraphicsVDPAURegisterVideoSurface";
    case 189:   return "cuGraphicsVDPAURegisterOutputSurface";
    case 190:   return "cuModuleGetSurfRef";
    case 191:   return "cuSurfRefCreate";
    case 192:   return "cuSurfRefDestroy";
    case 193:   return "cuSurfRefSetFormat";
    case 194:   return "cuSurfRefSetArray";
    case 195:   return "cuSurfRefGetFormat";
    case 196:   return "cuSurfRefGetArray";
    case 197:   return "cu64DeviceTotalMem";
    case 198:   return "cu64D3D10ResourceGetMappedPointer";
    case 199:   return "cu64D3D10ResourceGetMappedSize";
    case 200:   return "cu64D3D10ResourceGetMappedPitch";
    case 201:   return "cu64D3D10ResourceGetSurfaceDimensions";
    case 202:   return "cu64D3D9ResourceGetSurfaceDimensions";
    case 203:   return "cu64D3D9ResourceGetMappedPointer";
    case 204:   return "cu64D3D9ResourceGetMappedSize";
    case 205:   return "cu64D3D9ResourceGetMappedPitch";
    case 206:   return "cu64D3D9MapVertexBuffer";
    case 207:   return "cu64GLMapBufferObject";
    case 208:   return "cu64GLMapBufferObjectAsync";
    case 209:   return "cuD3D11GetDevices";
    case 210:   return "cuD3D11CtxCreateOnDevice";
    case 211:   return "cuD3D10GetDevices";
    case 212:   return "cuD3D10CtxCreateOnDevice";
    case 213:   return "cuD3D9GetDevices";
    case 214:   return "cuD3D9CtxCreateOnDevice";
    case 215:   return "cu64MemHostAlloc";
    case 216:   return "cuMemsetD8Async";
    case 217:   return "cu64MemsetD8Async";
    case 218:   return "cuMemsetD16Async";
    case 219:   return "cu64MemsetD16Async";
    case 220:   return "cuMemsetD32Async";
    case 221:   return "cu64MemsetD32Async";
    case 222:   return "cuMemsetD2D8Async";
    case 223:   return "cu64MemsetD2D8Async";
    case 224:   return "cuMemsetD2D16Async";
    case 225:   return "cu64MemsetD2D16Async";
    case 226:   return "cuMemsetD2D32Async";
    case 227:   return "cu64MemsetD2D32Async";
    case 228:   return "cu64ArrayCreate";
    case 229:   return "cu64ArrayGetDescriptor";
    case 230:   return "cu64Array3DCreate";
    case 231:   return "cu64Array3DGetDescriptor";
    case 232:   return "cu64Memcpy2D";
    case 233:   return "cu64Memcpy2DUnaligned";
    case 234:   return "cu64Memcpy2DAsync";
    case 235:   return "cuCtxCreate_v2";
    case 236:   return "cuD3D10CtxCreate_v2";
    case 237:   return "cuD3D11CtxCreate_v2";
    case 238:   return "cuD3D9CtxCreate_v2";
    case 239:   return "cuGLCtxCreate_v2";
    case 240:   return "cuVDPAUCtxCreate_v2";
    case 241:   return "cuModuleGetGlobal_v2";
    case 242:   return "cuMemGetInfo_v2";
    case 243:   return "cuMemAlloc_v2";
    case 244:   return "cuMemAllocPitch_v2";
    case 245:   return "cuMemFree_v2";
    case 246:   return "cuMemGetAddressRange_v2";
    case 247:   return "cuMemHostGetDevicePointer_v2";
    case 248:   return "cuMemcpy_v2";
    case 249:   return "cuMemsetD8_v2";
    case 250:   return "cuMemsetD16_v2";
    case 251:   return "cuMemsetD32_v2";
    case 252:   return "cuMemsetD2D8_v2";
    case 253:   return "cuMemsetD2D16_v2";
    case 254:   return "cuMemsetD2D32_v2";
    case 255:   return "cuTexRefSetAddress_v2";
    case 256:   return "cuTexRefSetAddress2D_v2";
    case 257:   return "cuTexRefGetAddress_v2";
    case 258:   return "cuGraphicsResourceGetMappedPointer_v2";
    case 259:   return "cuDeviceTotalMem_v2";
    case 260:   return "cuD3D10ResourceGetMappedPointer_v2";
    case 261:   return "cuD3D10ResourceGetMappedSize_v2";
    case 262:   return "cuD3D10ResourceGetMappedPitch_v2";
    case 263:   return "cuD3D10ResourceGetSurfaceDimensions_v2";
    case 264:   return "cuD3D9ResourceGetSurfaceDimensions_v2";
    case 265:   return "cuD3D9ResourceGetMappedPointer_v2";
    case 266:   return "cuD3D9ResourceGetMappedSize_v2";
    case 267:   return "cuD3D9ResourceGetMappedPitch_v2";
    case 268:   return "cuD3D9MapVertexBuffer_v2";
    case 269:   return "cuGLMapBufferObject_v2";
    case 270:   return "cuGLMapBufferObjectAsync_v2";
    case 271:   return "cuMemHostAlloc_v2";
    case 272:   return "cuArrayCreate_v2";
    case 273:   return "cuArrayGetDescriptor_v2";
    case 274:   return "cuArray3DCreate_v2";
    case 275:   return "cuArray3DGetDescriptor_v2";
    case 276:   return "cuMemcpyHtoD_v2";
    case 277:   return "cuMemcpyHtoDAsync_v2";
    case 278:   return "cuMemcpyDtoH_v2";
    case 279:   return "cuMemcpyDtoHAsync_v2";
    case 280:   return "cuMemcpyDtoD_v2";
    case 281:   return "cuMemcpyDtoDAsync_v2";
    case 282:   return "cuMemcpyAtoH_v2";
    case 283:   return "cuMemcpyAtoHAsync_v2";
    case 284:   return "cuMemcpyAtoD_v2";
    case 285:   return "cuMemcpyDtoA_v2";
    case 286:   return "cuMemcpyAtoA_v2";
    case 287:   return "cuMemcpy2D_v2";
    case 288:   return "cuMemcpy2DUnaligned_v2";
    case 289:   return "cuMemcpy2DAsync_v2";
    case 290:   return "cuMemcpy3D_v2";
    case 291:   return "cuMemcpy3DAsync_v2";
    case 292:   return "cuMemcpyHtoA_v2";
    case 293:   return "cuMemcpyHtoAAsync_v2";
    case 294:   return "cuMemAllocHost_v2";
    case 295:   return "cuStreamWaitEvent";
    case 296:   return "cuCtxGetApiVersion";
    case 297:   return "cuD3D10GetDirect3DDevice";
    case 298:   return "cuD3D11GetDirect3DDevice";
    case 299:   return "cuCtxGetCacheConfig";
    case 300:   return "cuCtxSetCacheConfig";
    case 301:   return "cuMemHostRegister";
    case 302:   return "cuMemHostUnregister";
    case 303:   return "cuCtxSetCurrent";
    case 304:   return "cuCtxGetCurrent";
    case 305:   return "cuMemcpy";
    case 306:   return "cuMemcpyAsync";
    case 307:   return "cuLaunchKernel";
    case 308:   return "cuProfilerStart";
    case 309:   return "cuProfilerStop";
    case 310:   return "cuPointerGetAttribute";
    case 311:   return "cuProfilerInitialize";
    case 312:   return "cuDeviceCanAccessPeer";
    case 313:   return "cuCtxEnablePeerAccess";
    case 314:   return "cuCtxDisablePeerAccess";
    case 315:   return "cuMemPeerRegister";
    case 316:   return "cuMemPeerUnregister";
    case 317:   return "cuMemPeerGetDevicePointer";
    case 318:   return "cuMemcpyPeer";
    case 319:   return "cuMemcpyPeerAsync";
    case 320:   return "cuMemcpy3DPeer";
    case 321:   return "cuMemcpy3DPeerAsync";
    case 322:   return "cuCtxDestroy_v2";
    case 323:   return "cuCtxPushCurrent_v2";
    case 324:   return "cuCtxPopCurrent_v2";
    case 325:   return "cuEventDestroy_v2";
    case 326:   return "cuStreamDestroy_v2";
    case 327:   return "cuTexRefSetAddress2D_v3";
    case 328:   return "cuIpcGetMemHandle";
    case 329:   return "cuIpcOpenMemHandle";
    case 330:   return "cuIpcCloseMemHandle";
    case 331:   return "cuDeviceGetByPCIBusId";
    case 332:   return "cuDeviceGetPCIBusId";
    case 333:   return "cuGLGetDevices";
    case 334:   return "cuIpcGetEventHandle";
    case 335:   return "cuIpcOpenEventHandle";
    case 336:   return "cuCtxSetSharedMemConfig";
    case 337:   return "cuCtxGetSharedMemConfig";
    case 338:   return "cuFuncSetSharedMemConfig";
    case 339:   return "cuTexObjectCreate";
    case 340:   return "cuTexObjectDestroy";
    case 341:   return "cuTexObjectGetResourceDesc";
    case 342:   return "cuTexObjectGetTextureDesc";
    case 343:   return "cuSurfObjectCreate";
    case 344:   return "cuSurfObjectDestroy";
    case 345:   return "cuSurfObjectGetResourceDesc";
    case 346:   return "cuStreamAddCallback";
    case 347:   return "cuMipmappedArrayCreate";
    case 348:   return "cuMipmappedArrayGetLevel";
    case 349:   return "cuMipmappedArrayDestroy";
    case 350:   return "cuTexRefSetMipmappedArray";
    case 351:   return "cuTexRefSetMipmapFilterMode";
    case 352:   return "cuTexRefSetMipmapLevelBias";
    case 353:   return "cuTexRefSetMipmapLevelClamp";
    case 354:   return "cuTexRefSetMaxAnisotropy";
    case 355:   return "cuTexRefGetMipmappedArray";
    case 356:   return "cuTexRefGetMipmapFilterMode";
    case 357:   return "cuTexRefGetMipmapLevelBias";
    case 358:   return "cuTexRefGetMipmapLevelClamp";
    case 359:   return "cuTexRefGetMaxAnisotropy";
    case 360:   return "cuGraphicsResourceGetMappedMipmappedArray";
    case 361:   return "cuTexObjectGetResourceViewDesc";
    case 362:   return "cuLinkCreate";
    case 363:   return "cuLinkAddData";
    case 364:   return "cuLinkAddFile";
    case 365:   return "cuLinkComplete";
    case 366:   return "cuLinkDestroy";
    case 367:   return "cuStreamCreateWithPriority";
    case 368:   return "cuStreamGetPriority";
    case 369:   return "cuStreamGetFlags";
    case 370:   return "cuCtxGetStreamPriorityRange";
    case 371:   return "cuMemAllocManaged";
    case 372:   return "cuGetErrorString";
    case 373:   return "cuGetErrorName";
    case 374:   return "cuOccupancyMaxActiveBlocksPerMultiprocessor";
    case 375:   return "cuCompilePtx";
    case 376:   return "cuBinaryFree";
    case 377:   return "cuStreamAttachMemAsync";
    case 378:   return "cuPointerSetAttribute";
    case 379:   return "cuMemHostRegister_v2";
    case 380:   return "cuGraphicsResourceSetMapFlags_v2";
    case 381:   return "cuLinkCreate_v2";
    case 382:   return "cuLinkAddData_v2";
    case 383:   return "cuLinkAddFile_v2";
    case 384:   return "cuOccupancyMaxPotentialBlockSize";
    case 385:   return "cuGLGetDevices_v2";
    case 386:   return "cuDevicePrimaryCtxRetain";
    case 387:   return "cuDevicePrimaryCtxRelease";
    case 388:   return "cuDevicePrimaryCtxSetFlags";
    case 389:   return "cuDevicePrimaryCtxReset";
    case 390:   return "cuGraphicsEGLRegisterImage";
    case 391:   return "cuCtxGetFlags";
    case 392:   return "cuDevicePrimaryCtxGetState";
    case 393:   return "cuEGLStreamConsumerConnect";
    case 394:   return "cuEGLStreamConsumerDisconnect";
    case 395:   return "cuEGLStreamConsumerAcquireFrame";
    case 396:   return "cuEGLStreamConsumerReleaseFrame";
    case 397:   return "cuMemcpyHtoD_v2_ptds";
    case 398:   return "cuMemcpyDtoH_v2_ptds";
    case 399:   return "cuMemcpyDtoD_v2_ptds";
    case 400:   return "cuMemcpyDtoA_v2_ptds";
    case 401:   return "cuMemcpyAtoD_v2_ptds";
    case 402:   return "cuMemcpyHtoA_v2_ptds";
    case 403:   return "cuMemcpyAtoH_v2_ptds";
    case 404:   return "cuMemcpyAtoA_v2_ptds";
    case 405:   return "cuMemcpy2D_v2_ptds";
    case 406:   return "cuMemcpy2DUnaligned_v2_ptds";
    case 407:   return "cuMemcpy3D_v2_ptds";
    case 408:   return "cuMemcpy_ptds";
    case 409:   return "cuMemcpyPeer_ptds";
    case 410:   return "cuMemcpy3DPeer_ptds";
    case 411:   return "cuMemsetD8_v2_ptds";
    case 412:   return "cuMemsetD16_v2_ptds";
    case 413:   return "cuMemsetD32_v2_ptds";
    case 414:   return "cuMemsetD2D8_v2_ptds";
    case 415:   return "cuMemsetD2D16_v2_ptds";
    case 416:   return "cuMemsetD2D32_v2_ptds";
    case 417:   return "cuGLMapBufferObject_v2_ptds";
    case 418:   return "cuMemcpyAsync_ptsz";
    case 419:   return "cuMemcpyHtoAAsync_v2_ptsz";
    case 420:   return "cuMemcpyAtoHAsync_v2_ptsz";
    case 421:   return "cuMemcpyHtoDAsync_v2_ptsz";
    case 422:   return "cuMemcpyDtoHAsync_v2_ptsz";
    case 423:   return "cuMemcpyDtoDAsync_v2_ptsz";
    case 424:   return "cuMemcpy2DAsync_v2_ptsz";
    case 425:   return "cuMemcpy3DAsync_v2_ptsz";
    case 426:   return "cuMemcpyPeerAsync_ptsz";
    case 427:   return "cuMemcpy3DPeerAsync_ptsz";
    case 428:   return "cuMemsetD8Async_ptsz";
    case 429:   return "cuMemsetD16Async_ptsz";
    case 430:   return "cuMemsetD32Async_ptsz";
    case 431:   return "cuMemsetD2D8Async_ptsz";
    case 432:   return "cuMemsetD2D16Async_ptsz";
    case 433:   return "cuMemsetD2D32Async_ptsz";
    case 434:   return "cuStreamGetPriority_ptsz";
    case 435:   return "cuStreamGetFlags_ptsz";
    case 436:   return "cuStreamWaitEvent_ptsz";
    case 437:   return "cuStreamAddCallback_ptsz";
    case 438:   return "cuStreamAttachMemAsync_ptsz";
    case 439:   return "cuStreamQuery_ptsz";
    case 440:   return "cuStreamSynchronize_ptsz";
    case 441:   return "cuEventRecord_ptsz";
    case 442:   return "cuLaunchKernel_ptsz";
    case 443:   return "cuGraphicsMapResources_ptsz";
    case 444:   return "cuGraphicsUnmapResources_ptsz";
    case 445:   return "cuGLMapBufferObjectAsync_v2_ptsz";
    case 446:   return "cuEGLStreamProducerConnect";
    case 447:   return "cuEGLStreamProducerDisconnect";
    case 448:   return "cuEGLStreamProducerPresentFrame";
    case 449:   return "cuGraphicsResourceGetMappedEglFrame";
    case 450:   return "cuPointerGetAttributes";
    case 451:   return "cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags";
    case 452:   return "cuOccupancyMaxPotentialBlockSizeWithFlags";
    case 453:   return "cuEGLStreamProducerReturnFrame";
    case 454:   return "cuDeviceGetP2PAttribute";
    case 455:   return "cuTexRefSetBorderColor";
    case 456:   return "cuTexRefGetBorderColor";
    case 457:   return "cuMemAdvise";
    case 458:   return "cuStreamWaitValue32";
    case 459:   return "cuStreamWaitValue32_ptsz";
    case 460:   return "cuStreamWriteValue32";
    case 461:   return "cuStreamWriteValue32_ptsz";
    case 462:   return "cuStreamBatchMemOp";
    case 463:   return "cuStreamBatchMemOp_ptsz";
    case 464:   return "cuNVNbufferGetPointer";
    case 465:   return "cuNVNtextureGetArray";
    case 466:   return "cuNNSetAllocator";
    case 467:   return "cuMemPrefetchAsync";
    case 468:   return "cuMemPrefetchAsync_ptsz";
    case 469:   return "cuEventCreateFromNVNSync";
    case 470:   return "cuEGLStreamConsumerConnectWithFlags";
    case 471:   return "cuMemRangeGetAttribute";
    case 472:   return "cuMemRangeGetAttributes";
    case 473:   return "cuStreamWaitValue64";
    case 474:   return "cuStreamWaitValue64_ptsz";
    case 475:   return "cuStreamWriteValue64";
    case 476:   return "cuStreamWriteValue64_ptsz";
    case 477:   return "cuLaunchCooperativeKernel";
    case 478:   return "cuLaunchCooperativeKernel_ptsz";
    case 479:   return "cuEventCreateFromEGLSync";
    case 480:   return "cuLaunchCooperativeKernelMultiDevice";
    case 481:   return "cuFuncSetAttribute";
    case 482:   return "cuDeviceGetUuid";
    case 483:   return "cuStreamGetCtx";
    case 484:   return "cuStreamGetCtx_ptsz";
    case 485:   return "cuImportExternalMemory";
    case 486:   return "cuExternalMemoryGetMappedBuffer";
    case 487:   return "cuExternalMemoryGetMappedMipmappedArray";
    case 488:   return "cuDestroyExternalMemory";
    case 489:   return "cuImportExternalSemaphore";
    case 490:   return "cuSignalExternalSemaphoresAsync";
    case 491:   return "cuSignalExternalSemaphoresAsync_ptsz";
    case 492:   return "cuWaitExternalSemaphoresAsync";
    case 493:   return "cuWaitExternalSemaphoresAsync_ptsz";
    case 494:   return "cuDestroyExternalSemaphore";
    case 495:   return "cuStreamBeginCapture";
    case 496:   return "cuStreamBeginCapture_ptsz";
    case 497:   return "cuStreamEndCapture";
    case 498:   return "cuStreamEndCapture_ptsz";
    case 499:   return "cuStreamIsCapturing";
    case 500:   return "cuStreamIsCapturing_ptsz";
    case 501:   return "cuGraphCreate";
    case 502:   return "cuGraphAddKernelNode";
    case 503:   return "cuGraphKernelNodeGetParams";
    case 504:   return "cuGraphAddMemcpyNode";
    case 505:   return "cuGraphMemcpyNodeGetParams";
    case 506:   return "cuGraphAddMemsetNode";
    case 507:   return "cuGraphMemsetNodeGetParams";
    case 508:   return "cuGraphMemsetNodeSetParams";
    case 509:   return "cuGraphNodeGetType";
    case 510:   return "cuGraphGetRootNodes";
    case 511:   return "cuGraphNodeGetDependencies";
    case 512:   return "cuGraphNodeGetDependentNodes";
    case 513:   return "cuGraphInstantiate";
    case 514:   return "cuGraphLaunch";
    case 515:   return "cuGraphLaunch_ptsz";
    case 516:   return "cuGraphExecDestroy";
    case 517:   return "cuGraphDestroy";
    case 518:   return "cuGraphAddDependencies";
    case 519:   return "cuGraphRemoveDependencies";
    case 520:   return "cuGraphMemcpyNodeSetParams";
    case 521:   return "cuGraphKernelNodeSetParams";
    case 522:   return "cuGraphDestroyNode";
    case 523:   return "cuGraphClone";
    case 524:   return "cuGraphNodeFindInClone";
    case 525:   return "cuGraphAddChildGraphNode";
    case 526:   return "cuGraphAddEmptyNode";
    case 527:   return "cuLaunchHostFunc";
    case 528:   return "cuLaunchHostFunc_ptsz";
    case 529:   return "cuGraphChildGraphNodeGetGraph";
    case 530:   return "cuGraphAddHostNode";
    case 531:   return "cuGraphHostNodeGetParams";
    case 532:   return "cuDeviceGetLuid";
    case 533:   return "cuGraphHostNodeSetParams";
    case 534:   return "cuGraphGetNodes";
    case 535:   return "cuGraphGetEdges";
    case 536:   return "cuStreamGetCaptureInfo";
    case 537:   return "cuStreamGetCaptureInfo_ptsz";
    case 538:   return "cuGraphExecKernelNodeSetParams";
    case 539:   return "cuStreamBeginCapture_v2";
    case 540:   return "cuStreamBeginCapture_v2_ptsz";
    case 541:   return "cuThreadExchangeStreamCaptureMode";
    case 542:   return "cuDeviceGetNvSciSyncAttributes";
    case 543:   return "cuOccupancyAvailableDynamicSMemPerBlock";
    case 544:   return "cuDevicePrimaryCtxRelease_v2";
    case 545:   return "cuDevicePrimaryCtxReset_v2";
    case 546:   return "cuDevicePrimaryCtxSetFlags_v2";
    case 547:   return "cuMemAddressReserve";
    case 548:   return "cuMemAddressFree";
    case 549:   return "cuMemCreate";
    case 550:   return "cuMemRelease";
    case 551:   return "cuMemMap";
    case 552:   return "cuMemUnmap";
    case 553:   return "cuMemSetAccess";
    case 554:   return "cuMemExportToShareableHandle";
    case 555:   return "cuMemImportFromShareableHandle";
    case 556:   return "cuMemGetAllocationGranularity";
    case 557:   return "cuMemGetAllocationPropertiesFromHandle";
    case 558:   return "cuMemGetAccess";
    case 559:   return "cuStreamSetFlags";
    case 560:   return "cuStreamSetFlags_ptsz";
    case 561:   return "cuGraphExecUpdate";
    case 562:   return "cuGraphExecMemcpyNodeSetParams";
    case 563:   return "cuGraphExecMemsetNodeSetParams";
    case 564:   return "cuGraphExecHostNodeSetParams";
    case 565:   return "cuMemRetainAllocationHandle";
    case 566:   return "cuFuncGetModule";
    case 567:   return "cuIpcOpenMemHandle_v2";
    case 568:   return "cuCtxResetPersistingL2Cache";
    case 569:   return "cuGraphKernelNodeCopyAttributes";
    case 570:   return "cuGraphKernelNodeGetAttribute";
    case 571:   return "cuGraphKernelNodeSetAttribute";
    case 572:   return "cuStreamCopyAttributes";
    case 573:   return "cuStreamCopyAttributes_ptsz";
    case 574:   return "cuStreamGetAttribute";
    case 575:   return "cuStreamGetAttribute_ptsz";
    case 576:   return "cuStreamSetAttribute";
    case 577:   return "cuStreamSetAttribute_ptsz";
    case 578:   return "cuGraphInstantiate_v2";
    case 579:   return "cuDeviceGetTexture1DLinearMaxWidth";
    case 580:   return "cuGraphUpload";
    case 581:   return "cuGraphUpload_ptsz";
    case 582:   return "cuArrayGetSparseProperties";
    case 583:   return "cuMipmappedArrayGetSparseProperties";
    case 584:   return "cuMemMapArrayAsync";
    case 585:   return "cuMemMapArrayAsync_ptsz";
    case 586:   return "cuGraphExecChildGraphNodeSetParams";
    case 587:   return "cuEventRecordWithFlags";
    case 588:   return "cuEventRecordWithFlags_ptsz";
    case 589:   return "cuGraphAddEventRecordNode";
    case 590:   return "cuGraphAddEventWaitNode";
    case 591:   return "cuGraphEventRecordNodeGetEvent";
    case 592:   return "cuGraphEventWaitNodeGetEvent";
    case 593:   return "cuGraphEventRecordNodeSetEvent";
    case 594:   return "cuGraphEventWaitNodeSetEvent";
    case 595:   return "cuGraphExecEventRecordNodeSetEvent";
    case 596:   return "cuGraphExecEventWaitNodeSetEvent";
    case 597:   return "cuArrayGetPlane";
    case 598:   return "cuMemAllocAsync";
    case 599:   return "cuMemAllocAsync_ptsz";
    case 600:   return "cuMemFreeAsync";
    case 601:   return "cuMemFreeAsync_ptsz";
    case 602:   return "cuMemPoolTrimTo";
    case 603:   return "cuMemPoolSetAttribute";
    case 604:   return "cuMemPoolGetAttribute";
    case 605:   return "cuMemPoolSetAccess";
    case 606:   return "cuDeviceGetDefaultMemPool";
    case 607:   return "cuMemPoolCreate";
    case 608:   return "cuMemPoolDestroy";
    case 609:   return "cuDeviceSetMemPool";
    case 610:   return "cuDeviceGetMemPool";
    case 611:   return "cuMemAllocFromPoolAsync";
    case 612:   return "cuMemAllocFromPoolAsync_ptsz";
    case 613:   return "cuMemPoolExportToShareableHandle";
    case 614:   return "cuMemPoolImportFromShareableHandle";
    case 615:   return "cuMemPoolExportPointer";
    case 616:   return "cuMemPoolImportPointer";
    case 617:   return "cuMemPoolGetAccess";
    case 618:   return "cuGraphAddExternalSemaphoresSignalNode";
    case 619:   return "cuGraphExternalSemaphoresSignalNodeGetParams";
    case 620:   return "cuGraphExternalSemaphoresSignalNodeSetParams";
    case 621:   return "cuGraphAddExternalSemaphoresWaitNode";
    case 622:   return "cuGraphExternalSemaphoresWaitNodeGetParams";
    case 623:   return "cuGraphExternalSemaphoresWaitNodeSetParams";
    case 624:   return "cuGraphExecExternalSemaphoresSignalNodeSetParams";
    case 625:   return "cuGraphExecExternalSemaphoresWaitNodeSetParams";
    case 626:   return "cuGetProcAddress";
    case 627:   return "cuFlushGPUDirectRDMAWrites";
    case 628:   return "cuGraphDebugDotPrint";
    case 629:   return "cuStreamGetCaptureInfo_v2";
    case 630:   return "cuStreamGetCaptureInfo_v2_ptsz";
    case 631:   return "cuStreamUpdateCaptureDependencies";
    case 632:   return "cuStreamUpdateCaptureDependencies_ptsz";
    case 633:   return "cuUserObjectCreate";
    case 634:   return "cuUserObjectRetain";
    case 635:   return "cuUserObjectRelease";
    case 636:   return "cuGraphRetainUserObject";
    case 637:   return "cuGraphReleaseUserObject";
    case 638:   return "cuGraphAddMemAllocNode";
    case 639:   return "cuGraphAddMemFreeNode";
    case 640:   return "cuDeviceGraphMemTrim";
    case 641:   return "cuDeviceGetGraphMemAttribute";
    case 642:   return "cuDeviceSetGraphMemAttribute";
    case 643:   return "cuGraphInstantiateWithFlags";
    case 644:   return "cuDeviceGetExecAffinitySupport";
    case 645:   return "cuCtxCreate_v3";
    case 646:   return "cuCtxGetExecAffinity";
    case 647:   return "cuDeviceGetUuid_v2";
    case 648:   return "cuGraphMemAllocNodeGetParams";
    case 649:   return "cuGraphMemFreeNodeGetParams";
    default:    return "Invalid CUDA API";
  }
}
