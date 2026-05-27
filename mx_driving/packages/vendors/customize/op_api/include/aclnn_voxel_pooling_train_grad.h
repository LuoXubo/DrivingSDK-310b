
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_VOXEL_POOLING_TRAIN_GRAD_H_
#define ACLNN_VOXEL_POOLING_TRAIN_GRAD_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnVoxelPoolingTrainGradGetWorkspaceSize
 * parameters :
 * gradOut : required
 * posMemo : required
 * batchSize : required
 * numPoints : required
 * numChannels : required
 * h : required
 * w : required
 * out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnVoxelPoolingTrainGradGetWorkspaceSize(
    const aclTensor *gradOut,
    const aclTensor *posMemo,
    int64_t batchSize,
    int64_t numPoints,
    int64_t numChannels,
    int64_t h,
    int64_t w,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnVoxelPoolingTrainGrad
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnVoxelPoolingTrainGrad(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
