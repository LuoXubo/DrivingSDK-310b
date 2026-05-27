
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_SUBM_SPARSE_CONV3D_H_
#define ACLNN_SUBM_SPARSE_CONV3D_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnSubmSparseConv3dGetWorkspaceSize
 * parameters :
 * feature : required
 * indices : required
 * weight : required
 * temp : required
 * kernelSize : required
 * outChannel : required
 * outSpatialShape : required
 * batchSize : required
 * featureOutOut : required
 * indicesOffsetOut : required
 * indicesPairOut : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnSubmSparseConv3dGetWorkspaceSize(
    const aclTensor *feature,
    const aclTensor *indices,
    const aclTensor *weight,
    const aclTensor *temp,
    const aclIntArray *kernelSize,
    int64_t outChannel,
    const aclIntArray *outSpatialShape,
    int64_t batchSize,
    const aclTensor *featureOutOut,
    const aclTensor *indicesOffsetOut,
    const aclTensor *indicesPairOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnSubmSparseConv3d
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnSubmSparseConv3d(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
