
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_NMS3D_ON_SIGHT_H_
#define ACLNN_NMS3D_ON_SIGHT_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnNms3dOnSightGetWorkspaceSize
 * parameters :
 * boxes : required
 * threshold : required
 * out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnNms3dOnSightGetWorkspaceSize(
    const aclTensor *boxes,
    double threshold,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnNms3dOnSight
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnNms3dOnSight(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
