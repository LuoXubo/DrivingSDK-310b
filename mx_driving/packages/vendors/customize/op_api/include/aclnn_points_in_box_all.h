
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_POINTS_IN_BOX_ALL_H_
#define ACLNN_POINTS_IN_BOX_ALL_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnPointsInBoxAllGetWorkspaceSize
 * parameters :
 * boxes : required
 * pts : required
 * out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnPointsInBoxAllGetWorkspaceSize(
    const aclTensor *boxes,
    const aclTensor *pts,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnPointsInBoxAll
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnPointsInBoxAll(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
