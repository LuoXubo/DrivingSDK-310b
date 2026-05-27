
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_BOX_IOU_H_
#define ACLNN_BOX_IOU_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnBoxIouGetWorkspaceSize
 * parameters :
 * boxesA : required
 * boxesB : required
 * modeFlag : optional
 * aligned : optional
 * out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnBoxIouGetWorkspaceSize(
    const aclTensor *boxesA,
    const aclTensor *boxesB,
    int64_t modeFlag,
    bool aligned,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnBoxIou
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnBoxIou(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
