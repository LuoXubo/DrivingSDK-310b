
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_VOXEL_POOLING_TRAIN_H_
#define ACLNN_VOXEL_POOLING_TRAIN_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnVoxelPoolingTrainGetWorkspaceSize
 * parameters :
 * geom : required
 * inputFeatures : required
 * batchSize : required
 * numPoints : required
 * numChannels : required
 * numVoxelX : required
 * numVoxelY : required
 * numVoxelZ : required
 * outputFeaturesOut : required
 * posMemoOut : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnVoxelPoolingTrainGetWorkspaceSize(
    const aclTensor *geom,
    const aclTensor *inputFeatures,
    int64_t batchSize,
    int64_t numPoints,
    int64_t numChannels,
    int64_t numVoxelX,
    int64_t numVoxelY,
    int64_t numVoxelZ,
    const aclTensor *outputFeaturesOut,
    const aclTensor *posMemoOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnVoxelPoolingTrain
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnVoxelPoolingTrain(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
