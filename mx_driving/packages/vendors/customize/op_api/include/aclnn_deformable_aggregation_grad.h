
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_DEFORMABLE_AGGREGATION_GRAD_H_
#define ACLNN_DEFORMABLE_AGGREGATION_GRAD_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnDeformableAggregationGradGetWorkspaceSize
 * parameters :
 * mcMsFeat : required
 * spatialShape : required
 * scaleStartIndex : required
 * samplingLocation : required
 * weights : required
 * gradOutput : required
 * gradMcMsFeatOut : required
 * gradSamplingLocationOut : required
 * gradWeightsOut : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnDeformableAggregationGradGetWorkspaceSize(
    const aclTensor *mcMsFeat,
    const aclTensor *spatialShape,
    const aclTensor *scaleStartIndex,
    const aclTensor *samplingLocation,
    const aclTensor *weights,
    const aclTensor *gradOutput,
    const aclTensor *gradMcMsFeatOut,
    const aclTensor *gradSamplingLocationOut,
    const aclTensor *gradWeightsOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnDeformableAggregationGrad
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnDeformableAggregationGrad(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
