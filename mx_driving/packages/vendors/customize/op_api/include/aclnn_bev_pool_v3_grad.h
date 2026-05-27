
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_BEVPOOL_V3GRAD_H_
#define ACLNN_BEVPOOL_V3GRAD_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnBEVPoolV3GradGetWorkspaceSize
 * parameters :
 * gradOut : required
 * depthOptional : optional
 * feat : required
 * ranksDepthOptional : optional
 * ranksFeatOptional : optional
 * ranksBev : required
 * withDepth : required
 * gradDepthOutOptional : optional
 * gradFeatOut : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnBEVPoolV3GradGetWorkspaceSize(
    const aclTensor *gradOut,
    const aclTensor *depthOptional,
    const aclTensor *feat,
    const aclTensor *ranksDepthOptional,
    const aclTensor *ranksFeatOptional,
    const aclTensor *ranksBev,
    bool withDepth,
    const aclTensor *gradDepthOutOptional,
    const aclTensor *gradFeatOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnBEVPoolV3Grad
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnBEVPoolV3Grad(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
