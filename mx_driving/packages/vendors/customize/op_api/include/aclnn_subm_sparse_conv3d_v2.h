
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_SUBM_SPARSE_CONV3D_V2_H_
#define ACLNN_SUBM_SPARSE_CONV3D_V2_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnSubmSparseConv3dV2GetWorkspaceSize
 * parameters :
 * feature : required
 * indices : required
 * map1 : required
 * map2 : required
 * kernelSize : required
 * inChannels : required
 * outSpatialShape : required
 * batchSize : required
 * sparseRate : required
 * featureOutOut : required
 * indicesOffsetOut : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnSubmSparseConv3dV2GetWorkspaceSize(
    const aclTensor *feature,
    const aclTensor *indices,
    const aclTensor *map1,
    const aclTensor *map2,
    const aclIntArray *kernelSize,
    int64_t inChannels,
    const aclIntArray *outSpatialShape,
    int64_t batchSize,
    double sparseRate,
    const aclTensor *featureOutOut,
    const aclTensor *indicesOffsetOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnSubmSparseConv3dV2
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnSubmSparseConv3dV2(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
