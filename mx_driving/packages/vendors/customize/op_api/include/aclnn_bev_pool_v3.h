
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_BEVPOOL_V3_H_
#define ACLNN_BEVPOOL_V3_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnBEVPoolV3GetWorkspaceSize
 * parameters :
 * depthOptional : optional
 * feat : required
 * ranksDepthOptional : optional
 * ranksFeatOptional : optional
 * ranksBev : required
 * withDepth : required
 * b : required
 * d : required
 * h : required
 * w : required
 * c : required
 * out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnBEVPoolV3GetWorkspaceSize(
    const aclTensor *depthOptional,
    const aclTensor *feat,
    const aclTensor *ranksDepthOptional,
    const aclTensor *ranksFeatOptional,
    const aclTensor *ranksBev,
    bool withDepth,
    int64_t b,
    int64_t d,
    int64_t h,
    int64_t w,
    int64_t c,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnBEVPoolV3
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnBEVPoolV3(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
