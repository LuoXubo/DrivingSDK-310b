
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_DEFORMABLE_AGGREGATION_H_
#define ACLNN_DEFORMABLE_AGGREGATION_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnDeformableAggregationGetWorkspaceSize
 * parameters :
 * mcMsFeat : required
 * spatialShape : required
 * scaleStartIndex : required
 * samplingLocation : required
 * weights : required
 * batchSize : required
 * numFeat : required
 * numEmbeds : required
 * numAnchors : required
 * numPts : required
 * numCams : required
 * numScale : required
 * numGroups : required
 * out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnDeformableAggregationGetWorkspaceSize(
    const aclTensor *mcMsFeat,
    const aclTensor *spatialShape,
    const aclTensor *scaleStartIndex,
    const aclTensor *samplingLocation,
    const aclTensor *weights,
    int64_t batchSize,
    int64_t numFeat,
    int64_t numEmbeds,
    int64_t numAnchors,
    int64_t numPts,
    int64_t numCams,
    int64_t numScale,
    int64_t numGroups,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnDeformableAggregation
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnDeformableAggregation(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
