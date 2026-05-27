
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_SPARSE_CONV3D_H_
#define ACLNN_SPARSE_CONV3D_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnSparseConv3dGetWorkspaceSize
 * parameters :
 * indices : required
 * kernelSize : required
 * outSpatialShape : required
 * stride : required
 * padding : required
 * indicesOutOut : required
 * indicesPairOut : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnSparseConv3dGetWorkspaceSize(
    const aclTensor *indices,
    const aclIntArray *kernelSize,
    const aclIntArray *outSpatialShape,
    const aclIntArray *stride,
    const aclIntArray *padding,
    const aclTensor *indicesOutOut,
    const aclTensor *indicesPairOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnSparseConv3d
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnSparseConv3d(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
