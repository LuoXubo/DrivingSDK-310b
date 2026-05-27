
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_ASSIGN_SCORE_WITHK_H_
#define ACLNN_ASSIGN_SCORE_WITHK_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnAssignScoreWithkGetWorkspaceSize
 * parameters :
 * points : required
 * centers : required
 * scores : required
 * knnIdx : required
 * batchSize : required
 * nsource : required
 * npoint : required
 * numWeights : required
 * numNeighbors : required
 * numFeatures : required
 * aggregate : required
 * out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnAssignScoreWithkGetWorkspaceSize(
    const aclTensor *points,
    const aclTensor *centers,
    const aclTensor *scores,
    const aclTensor *knnIdx,
    int64_t batchSize,
    int64_t nsource,
    int64_t npoint,
    int64_t numWeights,
    int64_t numNeighbors,
    int64_t numFeatures,
    int64_t aggregate,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnAssignScoreWithk
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnAssignScoreWithk(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
