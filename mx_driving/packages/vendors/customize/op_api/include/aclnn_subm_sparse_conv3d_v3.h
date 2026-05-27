
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_SUBM_SPARSE_CONV3D_V3_H_
#define ACLNN_SUBM_SPARSE_CONV3D_V3_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnSubmSparseConv3dV3GetWorkspaceSize
 * parameters :
 * feature : required
 * weight : required
 * indices : required
 * indicesOffset : required
 * map1 : required
 * map2 : required
 * kernelSize : required
 * inChannels : required
 * outChannels : required
 * outSpatialShape : required
 * batchSize : required
 * withKey : required
 * featureOutOut : required
 * outIndicesOffsetOut : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnSubmSparseConv3dV3GetWorkspaceSize(
    const aclTensor *feature,
    const aclTensor *weight,
    const aclTensor *indices,
    const aclTensor *indicesOffset,
    const aclTensor *map1,
    const aclTensor *map2,
    const aclIntArray *kernelSize,
    int64_t inChannels,
    int64_t outChannels,
    const aclIntArray *outSpatialShape,
    int64_t batchSize,
    int64_t withKey,
    const aclTensor *featureOutOut,
    const aclTensor *outIndicesOffsetOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnSubmSparseConv3dV3
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnSubmSparseConv3dV3(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
