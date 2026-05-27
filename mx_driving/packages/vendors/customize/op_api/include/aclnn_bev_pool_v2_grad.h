
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_BEVPOOL_V2GRAD_H_
#define ACLNN_BEVPOOL_V2GRAD_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnBEVPoolV2GradGetWorkspaceSize
 * parameters :
 * gradOut : required
 * depth : required
 * feat : required
 * ranksDepth : required
 * ranksFeat : required
 * ranksBev : required
 * intervalLengths : required
 * intervalStarts : required
 * b : required
 * d : required
 * h : required
 * w : required
 * c : required
 * gradDepthOut : required
 * gradFeatOut : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnBEVPoolV2GradGetWorkspaceSize(
    const aclTensor *gradOut,
    const aclTensor *depth,
    const aclTensor *feat,
    const aclTensor *ranksDepth,
    const aclTensor *ranksFeat,
    const aclTensor *ranksBev,
    const aclTensor *intervalLengths,
    const aclTensor *intervalStarts,
    int64_t b,
    int64_t d,
    int64_t h,
    int64_t w,
    int64_t c,
    const aclTensor *gradDepthOut,
    const aclTensor *gradFeatOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnBEVPoolV2Grad
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnBEVPoolV2Grad(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
